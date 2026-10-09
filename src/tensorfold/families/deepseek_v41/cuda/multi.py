"""DeepSeek-V4.1's concurrent rounds (``tensorfold serve --parallel N``): N streams in N cache slots, each round's rows
of every stream verified in one forward (``SerialEngine.step_multi``), so the weights a round reads serve them all.

Rows are row-invariant, so a stream's tokens equal its serial decoding however many streams share its rounds. Rank 0
decides (admissions, prefill steps, each stream's draft count, completions) and sends every decision to rank 1 before
acting on it; both ranks then compute the same tokens (argmax or position-keyed samples of the gathered logits).

The shared-pool extent lifecycle (``_make_room``, ``_grow``, ``_room``, ``_settle``, ``yield_for``: grow in place,
else move to a free run, else evict kept prompts; the newest replayable stream gives its rows back) and rank 0
sending ADMIT / EVICT / MOVE / GROW for rank 1 to apply follow ``glm5_next/cuda/multi.py`` of MiaAI-Lab's
GLM-5.3-Flash TensorFold recipe (patches 0030, 0040-0042; Apache License 2.0, Copyright 2026 MiaAI-Lab).
``_settle`` is adapted from its ``_settle``; the rest is rewritten for this engine (sid-keyed messages, the
fewest whole kept extents chosen up front, no compaction, NoRoom). Copy rounds (TF_MULTI_COPY) use
``tensorfold.cuda.copy_drafts`` and follow the same recipe's copy-draft wiring (patches 0007, 0032). See
THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from tensorfold.cuda.copy_drafts import WINDOW as COPY_WINDOW
from tensorfold.cuda.copy_drafts import CopyDrafts, CopySettings
from tensorfold.cuda.sampling import sample_rows
from tensorfold.cuda.streams import Stream, next_fill

from .serial import DRING, MAX_ROWS, PREFILL_AHEAD, WINDOW_ROWS, SerialEngine

ADMIT, FILL, ROUND, DONE, EVICT, GROW, MOVE, RESTORE, PERSIST, IDLE = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10  # rank 0's messages
FRESH, TAKEOVER, COPY = 0, 1, 2            # how an admitted stream gets its extent (a kept prompt's, or new rows)
from .serial import PROMPT_ROWS as ROWS

KEEP_MIN = int(os.environ.get("TF_DSV41_KEEP_MIN") or 256)    # shorter states are not kept
# a stream's extent covers its prompt and this many reply tokens at first, and grows by as much while it decodes
# (in place, else moved to a free run, kept states evicted for it); 0: the whole reply up front
GROW_AHEAD = int(os.environ.get("TF_DSV41_GROW_AHEAD") or 4096)
RELEASE_AFTER = 4096                       # prompts prefilling more rows than this release cached blocks after
RELEASE_BELOW = int(float(os.environ.get("TF_DSV41_RELEASE_BELOW_GIB") or 2.5) * 2 ** 30)   # ... when memory is low
# the NVMe tier (``kvdisk``): a kept state on disk is restored when it covers this many more tokens than the best
# kept state in the pool (a restore costs the read; fewer tokens prefill about as fast)
DISK_GAIN = int(os.environ.get("TF_DSV41_DISK_GAIN") or 1024)
# a request's draft acceptance estimates (by draft position) start at DRAFT_PRIOR when DRAFT_RESET is on: identical
# requests then plan identical rounds, whatever ran before them (in one process: the cost curves are timed at start).
# TF_DSV41_DRAFT_RESET=0: they start from the running estimate over earlier requests (``MultiDecoder.prior``); the
# relax still pulls toward DRAFT_PRIOR (the old behaviour also needs DRAFT_RELAX=0 DRAFT_PRIOR=0.6). Draft counts never
# change a token.
DRAFT_RESET = os.environ.get("TF_DSV41_DRAFT_RESET", "1") != "0"
DRAFT_PRIOR = float(os.environ.get("TF_DSV41_DRAFT_PRIOR") or 0.8)
# a stream's estimates move this share of the way back to DRAFT_PRIOR each round it verifies no drafts: one that
# stopped drafting (its estimates are only updated by drafted rounds) tries again later instead of never (0: off).
# 2026-10-06, single requests (tools/dsv41_serial_run.py --jaybench): prior 0.6 without it (the old start) code 78.0,
# prose 40.4, structured 109.1, c1 98.7 tok/s; 0.8 + 0.02: 77.3-79.8 / 42.3 / 114.0 / 99.5, hard 38.9 -> 47.4
DRAFT_RELAX = float(os.environ.get("TF_DSV41_DRAFT_RELAX") or 0.02)
# a round that kept all its k drafts also moves the estimates of the positions it did not draft (k ..) toward the k-th's
# (TF_DSV41_DRAFT_BORROW=1; 0, the default: they stay where they are): otherwise a stream that starts at k drafts never
# learns position k+1's rate and stays at k (2026-10-09: some processes ran jaybench structured at 4 drafts a round
# with 99% kept, 112 tok/s, others at 5, 124; prior 0.8 prices 4 and 5 drafts within 1%). Draft counts only
DRAFT_BORROW = os.environ.get("TF_DSV41_DRAFT_BORROW", "0") == "1"
# streams under a grammar (response_format, tool grammars) draft like the rest (TF_DSV41_DRAFT_GRAMMAR=1): each verify
# row is masked by the grammar's state after the row's path (``Constraint.window``: through </think>, a draft the
# grammar rejects ends the path), so the kept tokens are the serial reply's. 0 (the default until the A/B): such
# streams verify one row a round, their reasoning included
DRAFT_GRAMMAR = os.environ.get("TF_DSV41_DRAFT_GRAMMAR", "0") == "1"
# a measured verify curve that replaces the calibrated one for every round (TF_DSV41_COST_CURVE="ms1,ms2,..": ms of
# 1, 2, .. rows; rows past the list add the calibrated curve's increments; empty, the default: the calibration's).
# An experiment knob: draft counts only, never tokens (rank 0 plans, agree() sends its k)
# replays a calibration timing of 1 .. CALIB_FEW rows takes the best of (TF_DSV41_CALIB_REPS, default 16; wider rows:
# 4): at 4 the 4-6-row costs varied by process (2026-10-09, 16 processes: row 4 38.98-40.31 ms, row 5 44.39-46.79)
# enough to flip one stream's draft count for the process's life (jaybench structured 112 / 124 tok/s, the same
# tokens); at 16, over 4 processes: 39.14-39.19, 44.55-44.75. ~5 s more calibration
CALIB_REPS = max(1, int(os.environ.get("TF_DSV41_CALIB_REPS") or "16"))
CALIB_FEW = 8
COST_CURVE = [float(v) for v in (os.environ.get("TF_DSV41_COST_CURVE") or "").split(",") if v.strip()]
# a prompt of KEEP_SHRINK_MIN tokens or more is also kept at 3/4 and 7/8 of its length (exact states), and room is made
# by dropping loose extents' longest kept states before whole extents are evicted (``_trim``): a long document's
# prefix stays resumable when the pool needs its tail. Policy after bertholomus/TensorFold v0.5 (508bfb3, kept prompts
# shrink to a 3/4 or 7/8 boundary; Apache License 2.0, Copyright 2026 BertholomusAI); done here with this engine's
# several kept states an extent and ``_settle``. TF_DSV41_KEEP_SHRINK=0: off
KEEP_SHRINK = os.environ.get("TF_DSV41_KEEP_SHRINK", "1") != "0"
KEEP_SHRINK_MIN = int(os.environ.get("TF_DSV41_KEEP_SHRINK_MIN") or 131072)
# copy (prompt-lookup) drafts in concurrent rounds (TF_MULTI_COPY, on by default since the 2026-10-08 A/B: jaybench edit
# 107.5 -> 178.1, docs 94.9 -> 115.1, code unchanged, the same reply shas; 0: none, as before): a stream whose
# context's last tokens occurred before verifies what followed them (``copy_drafts``) instead of DSpark's drafts, as the
# serial path does. Rank 0 plans a copy round as (sid, k + COPY_FLAG); both ranks then compute the same proposal from
# the stream's prompt and reply. A stream copies while the running mean (each copy round weighs half) of the share of its copy
# drafts kept stays at least MULTI_COPY_MIN; below it, it backs off 2, 4 .. 64 rounds before trying again; the round's
# length is the cost curve's pick, as a draft count is. TF_COPY_DRAFTS=0: off everywhere; TF_COPY_MATCH, TF_COPY_MAX
# (default 15 here) tune it. Copy-draft wiring after MiaAI-Lab's GLM recipe patches 0007 / 0032 (Apache-2.0); see
# THIRD_PARTY_NOTICES.md
MULTI_COPY = (os.environ.get("TF_MULTI_COPY") or "1") != "0"
MULTI_COPY_MIN = float(os.environ.get("TF_MULTI_COPY_MIN") or 0.3)
COPY_FLAG = 1 << 16                        # a plan's copy round: drafts + this (drafts < ROWS <= 2048)
COPY_BACKOFF = 64                          # most rounds a stream waits after copy rounds that missed
M_DSPARK, M_COPY = 1, 2                    # a stream's drafts this round (not the admission modes FRESH / TAKEOVER / COPY)


def costs_depth(value: str | None) -> int:
    """TF_DSV41_COSTS: empty / calib (the default, as before): every round is priced by the startup curve of random
    rows over the slots at position 200, which over-prices one stream's 5-6 row windows (2026-10-08, 2K: 24 29 34 38
    43 49 ms against decode-bench's 23.6 28.0 32.1 35.8 38.0 40.7); depth (or depth:N): calibration also times one
    stream's 1 .. drafts + 1 rows of natural consecutive tokens after a natural 2048 (N) token prompt, through
    ``step_multi`` as a round verifies them, and one-stream rounds are priced by that curve. Returns N (0: off).
    Draft counts only: never a token."""

    v = (value or "").strip().lower()
    if v in ("", "calib", "0"):
        return 0
    if v == "depth":
        return 2048
    if v.startswith("depth:") and v[6:].isdigit() and int(v[6:]) > 0:
        return int(v[6:])
    raise ValueError(f"TF_DSV41_COSTS={value!r}: calib, depth or depth:N")


COSTS_DEPTH = costs_depth(os.environ.get("TF_DSV41_COSTS"))


def natural_ids(tokenizer_json, n: int) -> list[int]:
    """``n`` natural tokens (code and English: this package's own source, tokenized), repeated if short; without a
    tokenizer, the Markov table's tokens in frequency order (frequent, but not a text)."""

    from pathlib import Path

    ids: list[int] = []
    if tokenizer_json is not None and Path(tokenizer_json).is_file():
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer_json))
        for p in sorted(Path(__file__).parent.glob("*.py")):
            ids += tok.encode(p.read_text(), add_special_tokens=False).ids
            if len(ids) >= n:
                break
    if not ids:
        from .markov_tokens import TOKENS

        ids = list(TOKENS)
    return (ids * (1 + n // len(ids)))[:n]


def copy_settings() -> CopySettings | None:
    """TF_MULTI_COPY's settings (both ranks must agree on them), or None: off."""

    if not MULTI_COPY:
        return None
    return CopySettings.from_env(ROWS - 1 if os.environ.get("TF_COPY_MAX") else min(15, ROWS - 1))


@dataclass(eq=False)
class Kept:
    """A kept prompt inside the shared pool: the per-token caches of ``ids`` in extent ``x``'s first rows and its
    window rings in bank entry ``bank``, which hold positions from ``vs`` on: a prompt sharing its first m tokens
    resumes there when the window before m (and the rows a short tail backs up into) is held.

    Only prefilled rows are kept: every prompt chunk gives a row the same values whatever the chunk's length or
    start (``SerialEngine.prefill``), so a resumed prompt's caches and reply equal a fresh prefill's. A reply's
    decoded rows are not kept (a decode graph's arithmetic differs in the last bits)."""

    kid: int
    x: Any
    ids: np.ndarray
    vs: int
    bank: int
    vd: int = 0         # from where every layer's window is exact (vs, or later after early bounded-tail chunks)
    disk: bytes | None = None   # its key on the NVMe tier once written there or restored from it

    @property
    def n(self) -> int:
        return len(self.ids)


def common_prefix(a: np.ndarray, b: np.ndarray) -> int:
    n = min(len(a), len(b))
    diff = np.flatnonzero(a[:n] != b[:n])
    return int(diff[0]) if len(diff) else n

SAMPLING_WORDS = 10


def pack_sampling(sampling) -> list[int]:
    """[on, seed lo, seed hi, top_k, temperature, top_p, min_p as int pairs]: SAMPLING_WORDS ints."""

    import struct

    if sampling is None or sampling.temperature <= 0:
        return [0] * SAMPLING_WORDS
    seed = sampling.seed & 0xFFFFFFFFFFFFFFFF
    out = [1, seed & 0xFFFFFFFF, seed >> 32, int(sampling.top_k)]
    for value in (sampling.temperature, sampling.top_p, sampling.min_p):
        out += list(struct.unpack("<ii", struct.pack("<d", float(value))))
    return out


def unpack_sampling(words: list[int]):
    import struct

    from tensorfold.engine.exact_sampling import Sampling

    if not words[0]:
        return None
    f = [struct.unpack("<d", struct.pack("<ii", words[4 + 2 * i], words[5 + 2 * i]))[0] for i in range(3)]
    return Sampling(seed=(words[2] << 32) | words[1], temperature=f[0], top_k=words[3], top_p=f[1], min_p=f[2])



class RoundSplit:
    """TF_ROUND_PROF=1: where each round's time goes on rank 0. Host marks (ms since the previous mark): fill (a
    prompt step, if any), plan (draft allocation), send (the ROUND message to rank 1: an exchange), draft (the
    drafting pass, its sync included), hash / gather0 / gather1 / wait (``step_multi``: Engram hashes before the
    launch, the two table reads while the window runs, then the wait for its end), post (accept / sample / roll
    back), emit (tokens to the stream); GPU spans by CUDA events: draft_gpu (drafting pass), window_gpu (the verify
    window's graph, launch to end). Rounds are kept by verified rows."""

    HOST = ("fill", "plan", "send", "draft", "hash", "gather0", "gather1", "wait", "post", "emit")

    def __init__(self) -> None:
        self.rounds: list[dict] = []
        self.cur: dict = {}
        self.ev: dict = {}
        self.t = self.t0 = 0.0

    def begin(self, t0: float) -> None:
        self.cur, self.ev, self.t0, self.t = {}, {}, t0, t0

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.cur[name] = self.cur.get(name, 0.0) + 1e3 * (now - self.t)
        self.t = now

    def event(self, name: str) -> None:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.ev[name] = e

    def end(self, rows: int) -> None:
        r = self.cur
        r["round"] = 1e3 * (time.perf_counter() - self.t0)
        r["rows"] = rows
        ev = self.ev
        if "d0" in ev and "d1" in ev:
            r["draft_gpu"] = ev["d0"].elapsed_time(ev["d1"])
        if "w0" in ev and "w1" in ev:
            r["window_gpu"] = ev["w0"].elapsed_time(ev["w1"])
        self.rounds.append(r)

    def take(self) -> list[dict]:
        out, self.rounds = self.rounds, []
        return out

    @staticmethod
    def summary(rounds: list[dict]) -> dict:
        """Mean ms a round of every part (over all rounds), and of the window by verified rows."""

        n = max(len(rounds), 1)
        keys = [*RoundSplit.HOST, "draft_gpu", "window_gpu", "round"]
        out = {k: sum(r.get(k, 0.0) for r in rounds) / n for k in keys}
        out["rounds"] = len(rounds)
        by = {}
        for r in rounds:
            if "window_gpu" in r:
                by.setdefault(r["rows"], []).append(r["window_gpu"])
        out["window_by_rows"] = {k: (sum(v) / len(v), len(v)) for k, v in sorted(by.items())}
        drafted = [r["draft_gpu"] for r in rounds if "draft_gpu" in r]
        out["draft_gpu_when"] = sum(drafted) / len(drafted) if drafted else 0.0
        out["drafting_rounds"] = len(drafted)
        return out


class MultiDecoder:
    """The ``tensorfold.cuda.scheduler.Scheduler``'s decoder over ``SerialEngine`` slots (rank 0 or 1 of two)."""

    def __init__(self, e: SerialEngine, share: Callable[[list[int] | None], list[int]], *, rank: int,
                 drafts: int = 3, step: int = MAX_ROWS, pool=None, gather=None, disk=None, bell=None) -> None:
        self.e, self.share, self.rank = e, share, rank
        # the idle doorbell (``tensorfold.cuda.doorbell``; None: off): armed on both ranks at IDLE, so rank 1 waits for
        # rank 0's next message on the CPU instead of in the GPU collective (``share`` rings / waits on it)
        self.bell = bell
        # the shared cache pool (``pool.Pool``): each admitted stream gets an extent of it, or (None) its slot's
        # fixed extent; a request whose extent does not fit waits (NoRoom) until a stream finishes
        self.pool = pool
        # kept prompts inside the pool (``Kept``): a stream's extent outlives it while it holds kept states, which
        # are evicted (least recently used first) when an admission needs the rows
        banks = len(e.bank[0]) // DRING if pool is not None and getattr(e, "bank", None) else 0
        self.kept_on = banks > 0 and os.environ.get("TF_DSV41_POOL_KEEP", "1") != "0"
        self.kept: list[Kept] = []                 # least recently used first
        self.banks = list(range(banks))            # free bank entries, ascending
        self.next_kid = 0
        self.ext: dict[int, Any] = {}              # sid -> its extent
        self.yielded: list[Stream] = []            # rank 0: background streams giving up their rows (to replay)
        self.kstats = {"kept": 0, "hits": 0, "cached": 0, "takeovers": 0, "copies": 0, "evictions": 0}
        # kept states spilled to NVMe when dropped and restored when a prompt resumes from one (``kvdisk.KeptDisk``;
        # both ranks agree on each restore through ``gather``, the engine's synchronous all-gather of ints)
        self.gather = gather
        self.disk = disk if disk is not None and gather is not None and self.kept_on else None
        if self.disk is not None:
            self.kstats.update({"spills": 0, "spill_mb": 0.0, "spill_ms": 0.0, "disk_hits": 0, "disk_tokens": 0,
                                "disk_bad": 0, "restore_ms": 0.0})
        self.drafts = drafts if e.drafter is not None else 0
        # copy drafts (TF_MULTI_COPY), or None
        self.copy = copy_settings()

        self.check = os.environ.get("TF_MULTI_CHECK") == "1"
        if os.environ.get("TF_MULTI_PROF"):
            e._mprof = {}
        if os.environ.get("TF_MULTI_DRAFTS"):                    # tuning: drafts a stream in concurrent rounds
            self.drafts = min(self.drafts, int(os.environ["TF_MULTI_DRAFTS"]))
        self.prof = {"rounds": 0, "draft": 0.0, "verify": 0.0, "post": 0.0, "fill": 0.0, "round": 0.0, "rows": 0,
                     "streams": 0} \
            if os.environ.get("TF_MULTI_PROF") else None
        if self.prof is not None and self.copy is not None:
            self.prof.update({"copy_rounds": 0, "copy_accepted": 0})
        self.step_rows = step                      # prompt rows a fill step takes while other streams decode
        self.free = list(range(e.slots))
        self.streams: dict[int, Stream] = {}       # decoding, by sid
        self.filling: list[Stream] = []
        self.next_id = 0
        self.eos = (int(e.c.eos_token_id),)
        self.broken: Exception | None = None
        self.model_dir = None                      # rank 1 compiles a request's grammar from it
        self.costs: list[float] | None = None      # verify ms by rows (calibrate)
        # TF_DSV41_COSTS=depth: (key width, ms by rows 1..ROWS) of one stream at depth (None: off), the timed rows,
        # the depth (rank 0's: both ranks time the same rows)
        self.one: tuple[int, list[float]] | None = None
        self.one_rows = self.one_depth = 0
        self.draft_ms = 0.0
        self.draft_curve: list[float] = []
        self.overhead = 2.0                        # a round's host ms besides the forward and drafts
        # acceptance by draft position: a new stream starts at prior0 (DRAFT_RESET) or at prior, the running estimate
        # over every stream so far
        self.reset_prior = DRAFT_RESET
        self.prior0 = [DRAFT_PRIOR] * max(self.drafts, 1)
        self.prior = list(self.prior0)
        # TF_ROUND_PROF=1 (rank 0): each round's host and GPU split (``RoundSplit``), read and cleared by the caller
        self.rsplit = RoundSplit() if os.environ.get("TF_ROUND_PROF") and rank == 0 else None
        import os as _os

        # while streams decode, prompt steps take this share of the time (the rest: decode rounds)
        self.fill_share = float(_os.environ.get("TF_FILL_SHARE") or "0.5")
        self.t_fill = self.t_decode = 0.0
        self.alone_rows = 8 * step                 # a prompt step's rows when nothing decodes (new arrivals join after)

    # -- costs and draft allocation -------------------------------------------------------------------------------
    @torch.no_grad()
    def calibrate(self, gather: Callable[[list[int]], list[list[int]]]) -> None:
        """Verify ms for 1..ROWS rows of distinct tokens spread over the slots, and a draft's ms (the slower rank's,
        both ranks call this together). Caches written here are reset before any stream uses its slot."""

        import random

        e = self.e
        rng = random.Random(0)
        widths = [*e.widths, e.cap]                          # a cost curve for each key width the graphs have
        self._one_layout(gather)
        path = self._calib_path(widths)
        cached = None
        if path is not None:
            try:
                cached = [int(v) for v in json.loads(path.read_text())]
            except (OSError, ValueError):
                cached = None
            if cached is not None and len(cached) != ROWS * len(widths) + self._draft_passes() + self.one_rows:
                cached = None
        if all(f[0] for f in gather([int(cached is not None)])):
            self._curves(gather(cached), widths)             # both ranks timed this image and setting before
            return
        ms = []
        for w in widths:
            for R in range(1, ROWS + 1):
                if R not in e.graphs:
                    ms.append(ms[-1] if ms else 30.0)
                    continue
                g = e.graph_for(R, w)
                g["tok"].copy_(torch.tensor([rng.randrange(1000, 100000) for _ in range(R)]))
                g["pos"].copy_(torch.arange(R) + 200)
                g["sid"].copy_(torch.arange(R) % e.slots)
                best = float("inf")
                for _ in range(CALIB_REPS if R <= CALIB_FEW else 4):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    e._replay_free(g)
                    torch.cuda.synchronize()
                    best = min(best, time.perf_counter() - t)
                ms.append(1e3 * best)
        drafts = []                                          # ms of one drafting pass for 1.. streams at once
        if self.drafts:
            batched = getattr(e.drafter, "multi_graphs", None) or {}
            for M in range(1, max(batched, default=1) + 1):
                items = [(m % e.slots, rng.randrange(1000, 100000), 300) for m in range(M)]
                best = float("inf")
                for _ in range(4):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    if M in batched:
                        e.drafter.propose_multi(items)
                    else:
                        e.drafter.propose(items[0][1], 300)
                    best = min(best, time.perf_counter() - t)
                drafts.append(1e3 * best)
        one = self._time_one(widths) if self.one_rows else []   # (before the resets below: it fills slot 0)
        raw = [int(1e3 * v) for v in ms + drafts + one]
        if path is not None:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(f".{os.getpid()}")
                tmp.write_text(json.dumps(raw))
                os.replace(tmp, path)
            except OSError:
                pass
        self._curves(gather(raw), widths)
        for slot in range(e.slots):                          # the timing rows wrote every slot's caches
            e.select_slot(slot)
            e.reset()
        e.select_slot(0)

    def _draft_passes(self) -> int:
        """How many drafting-pass timings calibration takes (1 .. the batched graph sizes; none without drafts)."""

        return max(getattr(self.e.drafter, "multi_graphs", None) or {}, default=1) if self.drafts else 0

    def _one_layout(self, gather) -> None:
        """TF_DSV41_COSTS=depth: rank 0's depth on both ranks (each timed row is a collective and the timings are
        gathered: both must time the same rows), clamped to slot 0's extent; the rows to time (a stream's most: drafts
        + 1; copy windows beyond them are extrapolated from the random-row curve)."""

        both = gather([COSTS_DEPTH])
        depth = both[0][0]
        if any(row[0] != depth for row in both) and self.rank == 0:
            print(f"[tensorfold] TF_DSV41_COSTS differs between the ranks (depth {[row[0] for row in both]}): rank "
                  f"0's {depth} on both", flush=True)
        e = self.e
        rows = min(ROWS, self.drafts + 1) if self.drafts else 0
        depth = min(depth, min(e.limit, e.extents[0][1]) - ROWS - 1) if depth and rows else 0
        self.one_depth, self.one_rows = (depth, rows) if depth > 0 else (0, 0)

    def _time_one(self, widths: list[int]) -> list[float]:
        """ms of one stream verifying 1 .. one_rows rows of natural consecutive tokens after a natural one_depth-token
        prompt in slot 0: ``step_multi`` as a round calls it (Engram hashes and reads included), median of 5 after a
        warm-up; the rows are dropped after each. The caller resets every slot after."""

        import statistics

        e, depth = self.e, self.one_depth
        ids = natural_ids(getattr(e, "tokenizer_json", None), depth + self.one_rows)
        e.select_slot(0)
        e.reset()
        e.prefill(ids[:depth], final=depth)
        saved = getattr(e, "_mprof", None), getattr(e, "_rsplit", None)
        e._mprof = e._rsplit = None                          # (not a round: kept out of the profiles)
        out: list[float] = []
        try:
            for R in range(1, self.one_rows + 1):
                if R not in e.graphs:
                    out.append(out[-1] if out else 30.0)
                    continue
                rows = [(0, t) for t in ids[depth:depth + R]]
                times = []
                for _ in range(6):
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    e.step_multi(rows)
                    times.append(time.perf_counter() - t)
                    del e.views[0].ids[depth:]
                out.append(1e3 * statistics.median(times[1:]))
        finally:
            e._mprof, e._rsplit = saved
        if depth > 4096:                                     # a long prompt's transient buffers back to the system
            torch.cuda.empty_cache()
        return out

    def cost_report(self) -> list[str]:
        """The startup lines of the policy's curves: the random rows' (the widest key width), and with
        TF_DSV41_COSTS=depth one stream's beside the random rows' at its key width."""

        def ms(c):
            return " ".join(f"{v:.1f}" for v in c)

        curve = " ".join(f"{v:.0f}" for v in self.costs)
        out = [f"verify ms by rows 1..{len(self.costs)}: {curve}; a draft {self.draft_ms:.1f} ms"]
        if self.one is not None:
            w, one = self.one
            k = self.one_rows
            out.append(f"one-stream rounds (TF_DSV41_COSTS=depth:{self.one_depth}, key width {w}) priced by rows "
                       f"1..{k}: {ms(one[:k])} ms (random rows: {ms(dict(self.curves)[w][:k])})")
        return out

    def _calib_path(self, widths: list[int]):
        """Where this rank keeps its timings for this image and setting (TF_REVISION, set in the deploy image: the
        source snapshot's hash), or None (outside an image: always measure). The curves only steer draft counts,
        which never change a token."""

        import hashlib
        from pathlib import Path

        rev = os.environ.get("TF_REVISION", "")
        if rev in ("", "unknown") or os.environ.get("TF_DSV41_CALIB", "cached") != "cached":
            return None
        e = self.e
        from .kvdisk import knob_value

        # (not the copy-draft switches: both arms of an A/B read one curve; nor the warm-up's, which run after it;
        # TF_DSV41_COSTS: by the depth both ranks took, so the calib arm keeps its key and the depth arm, a curve
        # longer by one stream's, has its own)
        knobs = sorted((k, knob_value(k, v)) for k, v in os.environ.items() if k.startswith("TF_") and k not in
                       ("TF_API_KEY", "TF_PORT", "TF_RANK", "TF_DSV41_LAUNCH_T0", "TF_MULTI_COPY", "TF_MULTI_COPY_MIN",
                        "TF_DSV41_COSTS", "TF_DSV41_WARM_RARE", "TF_DSV41_WARM_TRACE"))
        key = [rev, torch.cuda.get_device_name(), torch.__version__, e.cap, e.slots, ROWS, widths, self.drafts,
               sorted(getattr(e.drafter, "multi_graphs", None) or {}), knobs]
        if self.one_rows:
            key.append(["one-v1", self.one_depth, self.one_rows])
        h = hashlib.sha256(json.dumps(key, default=str).encode()).hexdigest()[:16]
        d = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tensorfold" / "dsv41-calib"
        return d / f"r{self.rank}-{h}.json"

    def _curves(self, both: list[list[int]], widths: list[int]) -> None:
        worst = [max(a, b) / 1e3 for a, b in zip(*both)]
        n = ROWS * len(widths)
        self.curves = [(w, worst[i * ROWS:(i + 1) * ROWS]) for i, w in enumerate(widths)]
        self.costs = self.curves[-1][1]
        nd = self._draft_passes()
        self.draft_curve = worst[n:n + nd]
        self.draft_ms = self.draft_curve[0] if self.draft_curve else 0.0
        self.one = None
        if self.one_rows:
            one = worst[n + nd:n + nd + self.one_rows]
            # the key width its rows replayed at; rows past the timed ones: its last plus the random curve's steps
            w = next((w for w in widths if self.one_depth + self.one_rows <= w), widths[-1])
            base = dict(self.curves)[w]
            k = len(one)
            self.one = (w, one + [one[-1] + base[r] - base[k - 1] for r in range(k, ROWS)])

    def _expected(self, acc: list[float], k: int) -> float:
        total, run = 1.0, 1.0
        for j in range(k):
            run *= acc[j]
            total += run
        return total

    def _allocate(self, live: list[Stream]) -> list[int]:
        """Drafts a stream this round: one at a time to the stream whose next draft raises the round's expected
        tokens per ms the most, while any does (a draft is a verify row and, a stream's first, a drafting pass).
        With copy drafts, a stream with a copy proposal may instead take its first j copies (no drafting pass; each
        expected kept at the stream's running share): its count + COPY_FLAG."""

        ks = [0] * len(live)
        caps, rooms = [], []
        for s in live:
            room = min(self.e.limit, self.e.extents[s.slot][1]) - len(self.e.views[s.slot].ids) - 1
            ok = s.draft and (s.constraint is None or DRAFT_GRAMMAR) and self.drafts
            caps.append(max(0, min(self.drafts, room, s.count - len(s.out) - 1)) if ok else 0)
            rooms.append(room)
        props = self._proposals(live, rooms) if self.copy is not None and self.costs is not None else None
        if not any(caps) and not any(props or ()) or self.costs is None:
            return ks
        mode: list[int | None] = [None] * len(live)
        need = max(len(self.e.views[s.slot].ids) for s in live) + ROWS   # the round's graph width (at most)
        width, self.costs = next(((w, c) for w, c in self.curves if need <= w), self.curves[-1])
        if self.one is not None and len(live) == 1 and width == self.one[0]:
            self.costs = self.one[1]                                # TF_DSV41_COSTS=depth: one stream's own curve
        if COST_CURVE:
            n = min(len(COST_CURVE), len(self.costs))
            self.costs = [*COST_CURVE[:n], *(COST_CURVE[n - 1] + c - self.costs[n - 1] for c in self.costs[n:])]
        rows, drafting = len(live), 0
        tokens = float(len(live))
        rate = tokens / (self.costs[rows - 1] + self.overhead)
        while rows < ROWS:
            # a stream's next j drafts at once (j = 1 .. its cap): the cost curve is not convex (some row counts
            # cost much more than the next) and a stream's first draft also pays a drafting pass, so one draft at
            # a time stops early where several would pay (3 streams: almost no drafting)
            best = None
            for i, s in enumerate(live):
                if mode[i] != M_COPY:
                    base = self._expected(s.acc, ks[i])
                    for j in range(1, min(caps[i] - ks[i], ROWS - rows) + 1):
                        gain = self._expected(s.acc, ks[i] + j) - base
                        cost = self.costs[rows + j - 1] + self.overhead + self._draft_cost(drafting + (ks[i] == 0))
                        r = (tokens + gain) / cost
                        if r > rate and (best is None or r > best[0]):
                            best = (r, i, j, gain, M_DSPARK)
                if props is not None and props[i] and mode[i] != M_DSPARK:
                    kept = max(s.cfrac, MULTI_COPY_MIN)            # a copy runs on until an edit: as a share
                    for j in range(1, min(len(props[i]) - ks[i], ROWS - rows) + 1):
                        gain = kept * j
                        r = (tokens + gain) / (self.costs[rows + j - 1] + self.overhead + self._draft_cost(drafting))
                        if r > rate and (best is None or r > best[0]):
                            best = (r, i, j, gain, M_COPY)
            if best is None:
                break
            rate, i, j, gain, how = best
            drafting += how == M_DSPARK and ks[i] == 0
            mode[i] = how
            ks[i] += j
            tokens += gain
            rows += j
        return [k + COPY_FLAG if how == M_COPY else k for k, how in zip(ks, mode)]

    def _proposals(self, live: list[Stream], rooms: list[int]) -> list[list[int]]:
        """Rank 0: each stream's copy proposal this round ([]: none, a grammar's or a serial stream, or backing off
        after copies that missed)."""

        out = []
        for s, room in zip(live, rooms):
            cap = min(self.copy.most, room, s.count - len(s.out) - 1)
            ok = s.draft and s.constraint is None and s.cwait == 0 and cap > 0
            out.append(self._copies(s).propose(cap) if ok else [])
        return out

    def _copies(self, s: Stream) -> CopyDrafts:
        """Stream ``s``'s copy index (both ranks): its prompt's last WINDOW tokens (all a search reads) and its reply,
        the pending token last; built when its prompt ends (``_step``), caught up from the reply's tail here."""

        c = s.copies
        if c is None or len(c) - s.cbase > len(s.out):
            s.cbase = min(len(s.prompt), COPY_WINDOW)
            c = s.copies = CopyDrafts(s.prompt[len(s.prompt) - s.cbase:], self.copy)
        have = len(c) - s.cbase
        if have < len(s.out):
            c.extend(s.out[have:])
        return c

    def _draft_cost(self, streams: int) -> float:
        """ms of drafting for ``streams`` streams: one batched pass where captured, else a pass each."""

        if streams <= 0:
            return 0.0
        curve = self.draft_curve
        if streams <= len(curve) and len(curve) > 1:
            return curve[streams - 1]
        return streams * self.draft_ms

    def _learn(self, s: Stream, k: int, m: int) -> None:
        a = 0.15
        for j in range(min(k, m + 1)):                     # positions after the first rejection are unobserved
            hit = 1.0 if j < m else 0.0
            s.acc[j] = (1 - a) * s.acc[j] + a * hit
            self.prior[j] = (1 - a / 4) * self.prior[j] + a / 4 * hit
        if DRAFT_BORROW and 0 < k == m:
            for j in range(k, len(s.acc)):
                s.acc[j] += a * (s.acc[k - 1] - s.acc[j])

    def _learn_copy(self, s: Stream, k: int, m: int) -> None:
        """A copy round kept m of its k drafts: the running share; below MULTI_COPY_MIN, wait 2, 4 .. 64 rounds (a
        wait for each such round in a row) and start again from MULTI_COPY_MIN."""

        s.copy_rounds += 1
        s.copy_accepted += m
        if self.prof is not None and self.rank == 0:
            self.prof["copy_rounds"] += 1
            self.prof["copy_accepted"] += m
        s.cfrac = 0.5 * s.cfrac + 0.5 * m / k
        if s.cfrac < MULTI_COPY_MIN:
            s.cmiss += 1
            s.cwait = min(2 ** s.cmiss, COPY_BACKOFF)
            s.cfrac = MULTI_COPY_MIN               # (the next try copies again unless it keeps less)
        else:
            s.cmiss = 0

    def _agree_copies(self, plan, spans, rows) -> None:
        """TF_MULTI_CHECK=1: both ranks hash the round's copy windows and compare (a mismatch: out of step)."""

        import zlib

        words = [x for (sid, _, _), (r0, nrows, _, _, copied) in zip(plan, spans) if copied
                 for x in (sid, *(t for _, t in rows[r0:r0 + nrows]))]
        h = zlib.crc32(np.asarray(words, dtype=np.int64).tobytes())
        both = self.gather([h])
        if both[0] != both[1]:
            raise RuntimeError(f"the ranks disagree on a round's copy drafts (hash {both[0][0]} vs {both[1][0]})")

    def reset_policy(self) -> None:
        """The running acceptance estimate back to its start (a measurement's requests then plan as a fresh server's
        first request would, whatever ran before)."""

        self.prior = list(self.prior0)

    # -- bookkeeping --------------------------------------------------------------------------------------------
    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _send(self, values: list[int]) -> None:
        if self.broken is None:
            self.share(values)

    def idle(self) -> None:
        """Rank 0, the scheduler going idle (nothing live, nothing waiting): IDLE sent and the doorbell armed on
        both ranks; the next message, whatever it is (an admission, PERSIST or the end at shutdown), rings it first.
        Again before any message: nothing (rank 1 already waits for the ring)."""

        if self.bell is None or self.bell.armed or self.broken is not None:
            return
        self._send([IDLE])
        self.bell.arm()

    def _check(self) -> None:
        if self.broken is not None:
            raise RuntimeError("the two ranks are out of step after an error; restart both") from self.broken

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    # -- admission and prompts ------------------------------------------------------------------------------------
    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        self._check()
        room = self.e.limit - len(s.prompt) - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.e.limit}-token context")
        s.count = min(s.count, room)
        from tensorfold.engine.grammar import pack

        packed = pack(s.constraint)                # before any pool decision (an eviction is sent at once)
        base = size = eid = -1
        k, m, mode = None, 0, FRESH
        if self.pool is not None:                  # the prompt, the reply and a verify window's rows past it
            if any(x.waiting for x in self.streams.values()):
                from tensorfold.cuda.memory_gate import NoRoom

                raise NoRoom("streams wait for their caches to grow; a new request waits until one finishes")
            size = self._first(s)
            if self.kept_on and s.draft:
                k, m = self._match(s.prompt)
                if self.disk is not None:
                    k, m = self._from_disk(s, size, k, m)
            if k is not None and k.x.owner is None and size - k.x.size <= self.pool.room_after(k.x):
                mode, base, size = TAKEOVER, k.x.base, max(size, k.x.size)
            else:
                try:
                    base = self._room(size, [k.x] if k is not None else [])
                    mode = COPY if k is not None else FRESH
                except Exception:
                    if k is None:
                        raise
                    k, m, mode = None, 0, FRESH    # no room beside the kept rows: without them
                    base = self._room(size, [])
                eid = self.pool.next_eid
        s.sid = self.next_id
        self.next_id += 1
        self._send([ADMIT, s.sid, s.count, int(s.draft), int(s.stop_eos), *pack_sampling(s.sampling), len(packed),
                    base, size, eid, k.kid if k is not None else -1, m, mode])
        self._send(list(s.prompt))
        if packed:
            self._send(packed)
        self._queue(s, base, size, eid, mode, k, m)

    # -- extents: a stream's first rows, growth while it decodes ----------------------------------------------------
    # grow / move / evict / settle: after MiaAI-Lab's GLM recipe patch 0030 multi.py (Apache-2.0); see the module
    # docstring
    def _most(self, s: Stream) -> int:
        """The rows a stream can ever write: its prompt, its reply and a verify window past it."""

        from .pool import align_up

        return min(align_up(len(s.prompt) + s.count + ROWS + 1), self.e.span)

    def _first(self, s: Stream) -> int:
        from .pool import align_up

        if GROW_AHEAD <= 0:
            return self._most(s)
        return min(self._most(s), align_up(len(s.prompt) + GROW_AHEAD + ROWS + 1))

    def _make_room(self, live: list[Stream]) -> list[Stream]:
        """Rank 0, before a round: each decoding stream's extent covers the rows its round can write, oldest first
        (grown in place, else moved to a free run, else after evicting kept states; GROW/MOVE/EVICT sent before the
        ROUND). A stream that cannot grow waits, and so do the newer ones. When even the oldest cannot, one other
        stream gives up its rows (they free when it finishes, before the next round): the newest background one,
        handed back to replay later (``yielded``), else the newest, ended with an error. Returns the streams ended."""

        from .pool import align_up

        if self.pool is None:
            return []
        live = sorted(live, key=lambda x: x.sid)
        blocked = False
        for s in live:
            x = self.ext[s.sid]
            most = self._most(s)
            need = min(len(self.e.views[s.slot].ids) + ROWS + 1, most)   # (never past the limit: _allocate)
            if need <= x.size:
                s.waiting = False
                continue
            size = min(most, max(align_up(need), align_up(need + GROW_AHEAD) if GROW_AHEAD > 0 else 0))
            s.waiting = blocked or not self._grow(s, x, size)
            blocked = blocked or s.waiting
        if not live or not live[0].waiting or self.yielded:
            return []                              # (a stream already giving way frees its rows first)
        others = [x for x in [*live[1:], *self.filling] if not x.done and x not in self.yielded]
        if not others:
            return []
        resumable = [x for x in others if x.background and x.constraint is None and x.vision is None]
        victim = max(resumable or others, key=lambda x: x.sid)
        victim.waiting = False
        if resumable:
            self.yielded.append(victim)            # the scheduler ends it and queues its replay
            return []
        victim.error = RuntimeError(
            f"The shared cache pool ran out of room with {len(live)} streams decoding, so the newest (this request, "
            f"after {len(victim.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt "
            "or max_tokens, or start the server with a smaller --parallel.")
        victim.done, victim.finished = True, time.perf_counter()
        return [victim]

    def _grow(self, s: Stream, x, size: int) -> bool:
        """Rank 0: extent ``x`` of stream ``s`` to ``size`` rows; False when no room opens."""

        if size - x.size <= self.pool.room_after(x):
            self._send([GROW, s.sid, size])
            self._resize(s.sid, size)
            return True
        base = self.pool.place(size, ignore=[x])
        if base is None and self._trim(size, keep_out=[x], free=[x]):
            if size - x.size <= self.pool.room_after(x):
                self._send([GROW, s.sid, size])
                self._resize(s.sid, size)
                return True
            base = self.pool.place(size, ignore=[x])
        if base is None:                           # kept states' extents out of the way, the fewest that do
            order = {id(k): i for i, k in enumerate(self.kept)}
            loose = [y for y in self.pool.extents if y.owner is None and y.kept]
            loose.sort(key=lambda y: max(order[id(k)] for k in y.kept))
            for j in range(1, len(loose) + 1):
                if self.pool.place(size, ignore=[x, *loose[:j]]) is not None:
                    for y in loose[:j]:
                        for k in sorted(y.kept, key=lambda k: order[id(k)]):
                            self._send([EVICT, k.kid])
                            self._drop(k)
                            self.kstats["evictions"] += 1
                    break
            else:
                return False
            if size - x.size <= self.pool.room_after(x):
                self._send([GROW, s.sid, size])
                self._resize(s.sid, size)
                return True
            base = self.pool.place(size, ignore=[x])
        self._send([MOVE, s.sid, base, size])
        self._move(s.sid, base, size)
        return True

    def _resize(self, sid: int, size: int) -> None:
        """Both ranks: stream ``sid``'s extent grown in place (its rows stay)."""

        s, x = self.streams[sid], self.ext[sid]
        self.pool.resize(x, size)
        self.e.bind(s.slot, x.base, x.size, ids=list(self.e.views[s.slot].ids))
        self.kstats["grows"] = self.kstats.get("grows", 0) + 1

    def _move(self, sid: int, base: int, size: int) -> None:
        """Both ranks: stream ``sid``'s extent moved to ``base`` with ``size`` rows (its committed rows, and the kept
        states' in them, copied along)."""

        s, x = self.streams[sid], self.ext[sid]
        ids = list(self.e.views[s.slot].ids)
        old = self.pool.move(x, base, size)
        self.e.copy_rows(old, base, len(ids))
        self.e.bind(s.slot, base, size, ids=ids)
        self.kstats["moves"] = self.kstats.get("moves", 0) + 1

    def yield_for(self, s: Stream) -> list[Stream]:
        """Rank 0, ``s`` (foreground) found no room: the fewest background streams, newest first, whose extents (with
        every kept state's) would open a run for its first rows; [] when even all of them would not. The scheduler
        ends and re-queues them (they replay from their prompts)."""

        if self.pool is None or any(x.waiting for x in self.streams.values()):
            return []                              # (growth comes first: a yield would not admit it)
        cands = [x for x in [*self.streams.values(), *self.filling]
                 if x.background and not x.done and x.constraint is None and x.vision is None
                 and len(x.out) < x.count and x.sid in self.ext]
        cands.sort(key=lambda x: -x.sid)
        loose = [y for y in self.pool.extents if y.owner is None and y.kept]
        size = self._first(s)
        for j in range(len(cands) + 1):           # (j = 0: evicting kept states makes room, no yield needed)
            if self.pool.place(size, ignore=[*loose, *(self.ext[x.sid] for x in cands[:j])]) is not None:
                return cands[:j]
        return []

    # -- kept prompts in the shared pool (both ranks make the same calls in the same order) ------------------------
    def _match(self, prompt: list[int]) -> tuple[Kept | None, int]:
        """The kept state to resume ``prompt`` from and the tokens it covers: the most (at least one left to
        prefill, and the rings holding the window there), then one whose extent is free (taken over without
        copying), then the most recent."""

        p = np.asarray(prompt, dtype=np.int64)
        best, key = None, None
        for i, k in enumerate(self.kept):
            if k.ids[0] != p[0]:
                continue
            m = min(common_prefix(k.ids, p), len(p) - 1, k.n)
            if m < KEEP_MIN or not self._resumable(m, len(p), k.vs, k.vd):
                continue
            kk = (m, k.x.owner is None, i)
            if key is None or kk > key:
                best, key = k, kk
        return (best, key[0]) if best is not None else (None, 0)

    def _resumable(self, m: int, plen: int, vs: int, vd: int) -> bool:
        """Whether a ``plen``-token prompt resumes at m from a state whose rings hold positions from ``vs`` on (every
        layer's exact from ``vd``): the window before m held (and the rows a short tail backs up into)."""

        back = max(0, ROWS + 1 - (plen - m))     # a short tail backs up into rows before m (prefill)
        full = m - back - WINDOW_ROWS >= vd
        # (stale later-layer windows are fine far from the end: the bounded tail recomputes past their reach)
        early = m <= plen - self.e.tail_min and m - WINDOW_ROWS >= vs
        return full or early

    def _room(self, size: int, protect: list) -> int:
        """Rank 0: a base for a new ``size``-row extent, evicting kept states (least recently used first, none in a
        ``protect`` extent or a live stream's) until one fits; NoRoom when even all of them would not do."""

        from tensorfold.cuda.memory_gate import NoRoom

        base = self.pool.place(size)
        if base is None and self._trim(size, keep_out=protect):
            base = self.pool.place(size)
        if base is not None:
            return base
        # whole extents of kept states only (dropping a shorter state beside a longer one frees nothing), least
        # recently used first; the fewest that open a run of ``size`` rows, found before anything is evicted
        order = {id(k): i for i, k in enumerate(self.kept)}
        loose = [x for x in self.pool.extents if x.owner is None and x.kept and all(x is not y for y in protect)]
        loose.sort(key=lambda x: max(order[id(k)] for k in x.kept))
        for j in range(1, len(loose) + 1):
            if self.pool.place(size, ignore=loose[:j]) is not None:
                for x in loose[:j]:
                    for k in sorted(x.kept, key=lambda k: order[id(k)]):
                        self._send([EVICT, k.kid])
                        self._drop(k)
                        self.kstats["evictions"] += 1
                base = self.pool.place(size)
                if base is not None:
                    return base
                break
        raise NoRoom(f"the shared cache pool has no {size}-token extent free ({self.pool.free_rows()} of "
                     f"{self.pool.rows} tokens free, largest run {self.pool.largest_gap()})")

    def _trim(self, size: int, keep_out: list, free: list = ()) -> bool:
        """Rank 0, before whole kept extents are evicted: drop the longest kept states of loose extents (least
        recently used first, each keeping its shortest; none of ``keep_out``), whose rows ``_settle`` then frees,
        the fewest until a ``size``-row run opens (``free``: extents that count as free); True when it did. The
        dropped states go to the NVMe tier as evicted ones do."""

        from .pool import align_up

        if not KEEP_SHRINK:
            return False
        order = {id(k): i for i, k in enumerate(self.kept)}
        out = {id(y) for y in keep_out}
        loose = [y for y in self.pool.extents if y.owner is None and len(y.kept) > 1 and id(y) not in out]
        loose.sort(key=lambda y: max(order[id(k)] for k in y.kept))
        sizes: dict[int, int] = {}
        plan: list[Kept] = []
        for y in loose:
            ks = sorted(y.kept, key=lambda k: -k.n)
            for j in range(1, len(ks)):
                plan.append(ks[j - 1])
                sizes[id(y)] = align_up(ks[j].n)
                if self.pool.place(size, ignore=free, sizes=sizes) is not None:
                    for k in plan:
                        self._send([EVICT, k.kid])
                        self._drop(k)
                        self.kstats["trimmed"] = self.kstats.get("trimmed", 0) + 1
                    return True
        return False

    def _keep(self, s: Stream, x, ids: list[int], vs: int) -> None:
        """Keep stream ``s``'s state of ``ids`` (extent ``x``'s first rows, its slot's rings) as a kept prompt."""

        if not self.kept_on or len(ids) < KEEP_MIN:
            return
        a = np.asarray(ids, dtype=np.int64)
        for k in [k for k in self.kept if k.n == len(a) and np.array_equal(k.ids, a)]:
            self._drop(k, spill=False)             # the same tokens again: the newer state
        if not self.banks:
            self._drop(self.kept[0])
        bank = self.banks.pop(0)
        self.e.save_window(s.slot, bank)
        k = Kept(self.next_kid, x, a, vs, bank, max(vs, self.e.deep_from[s.slot]))
        self.next_kid += 1
        x.kept.append(k)
        self.kept.append(k)
        self.kstats["kept"] += 1

    def _drop(self, k: Kept, spill: bool = True) -> None:
        """Kept state ``k`` out of the pool (its rows and bank entry free); written to the NVMe tier first, if any,
        unless ``spill`` is False (the same state is kept again, or the caches may be broken)."""

        if spill and self.disk is not None:
            self._spill(k)
        self.kept.remove(k)
        k.x.kept.remove(k)
        self.banks.append(k.bank)
        self.banks.sort()
        self._settle(k.x)

    def _settle(self, x) -> None:
        """An extent no stream writes: gone when it keeps nothing, else cut to its longest kept state's rows."""

        from .pool import align_up

        if x.owner is not None:
            return
        if not x.kept:
            self.pool.remove(x)
            return
        need = align_up(max(k.n for k in x.kept))
        if need < x.size:
            self.pool.resize(x, need)

    def _kid(self, kid: int) -> Kept:
        k = next((k for k in self.kept if k.kid == kid), None)
        if k is None:
            raise RuntimeError(f"the ranks disagree on the kept prompts (no kept state {kid})")
        return k

    # -- the NVMe tier of kept prompts (``kvdisk``; both ranks make the same calls in the same order) ----------------
    def _spill(self, k: Kept) -> None:
        """Kept state ``k`` written to disk before its rows and bank entry are reused (only touched when its key is
        there already). The disk's index changes the same way on both ranks; a failed write only costs a miss."""

        d = self.disk
        if k.n < d.min_tokens:
            return
        t0 = time.perf_counter()
        key = k.disk or d.key(k.ids)
        if d.spill(key, k.n, k.vs, k.vd, k.ids, lambda: self.e.kept_views(k.x.base, k.n, k.bank)):
            self.kstats["spills"] += 1
            self.kstats["spill_mb"] += d.index[key].size / 2 ** 20 if key in d.index else 0.0
            self.kstats["spill_ms"] += 1e3 * (time.perf_counter() - t0)
        k.disk = key

    def _from_disk(self, s: Stream, size: int, k: Kept | None, m: int) -> tuple[Kept | None, int]:
        """Rank 0, admitting ``s``: when an entry on disk covers DISK_GAIN more of its prompt than the pool's best
        kept state ``k`` (m tokens), restore it into a free extent and bank entry (RESTORE sent first; evictions for
        the room sent as usual, never of ``k``'s extent), then match again: the kept state to resume from."""

        from tensorfold.cuda.memory_gate import NoRoom

        from .kvdisk import key_ints
        from .pool import align_up

        found = self.disk.find(s.prompt, self._resumable, span=self.e.span, least=KEEP_MIN)
        if found is None or found[0] < m + DISK_GAIN:
            return k, m
        dm, key, ent = found
        protect = [k.x] if k is not None else []
        try:
            base = self._room(max(size, align_up(ent.n)), protect)
        except NoRoom:
            return k, m
        if not self.banks:                         # a bank entry: the least recently used state's outside protect
            v = next((c for c in self.kept if all(c.x is not y for y in protect)), None)
            if v is None:
                return self._match(s.prompt)
            self._send([EVICT, v.kid])
            self._drop(v)
            self.kstats["evictions"] += 1
        args = [ent.n, base, self.pool.next_eid, self.banks[0], self.next_kid, ent.vs, ent.vd]
        self._send([RESTORE, *key_ints(key), *args, dm])
        self._restore(key, *args, prompt=s.prompt, m=dm)
        return self._match(s.prompt)             # (evictions happened whether or not it was restored)

    def _restore(self, key: bytes, n: int, base: int, eid: int, bank: int, kid: int, vs: int, vd: int,
                 prompt: list[int] | None = None, m: int = 0) -> None:
        """Both ranks: disk entry ``key`` (n tokens) into pool rows [base, base + n) and bank entry ``bank`` (both
        free), kept as ``kid`` in a new extent ``eid`` when both ranks read it whole (rank 0 also checks its first m
        tokens against ``prompt``); else deleted on both and nothing changes."""

        from .pool import align_up

        t0 = time.perf_counter()
        ids = None
        try:                                       # (no exception may skip the agreement: the other rank waits)
            ids = self.disk.restore(key, n, vs, vd, lambda: self.e.kept_views(base, n, bank))
            if ids is not None and prompt is not None and not np.array_equal(ids[:m], np.asarray(prompt[:m])):
                ids = None
        except Exception as exc:                   # noqa: BLE001
            print(f"[tensorfold] rank {self.rank}: restoring a kept prompt failed: {exc}", flush=True)
            ids = None
        if not all(row[0] for row in self.gather([int(ids is not None)])):
            self.disk.delete(key)
            self.kstats["disk_bad"] += 1
            return
        try:
            x = self.pool.add(base, align_up(n), owner=None, eid=eid)
            self.banks.remove(bank)
            if kid != self.next_kid:
                raise RuntimeError(f"the ranks disagree on the kept prompts (kept state {kid}, next {self.next_kid})")
            self.next_kid += 1
            k = Kept(kid, x, ids, vs, bank, vd, disk=key)
            x.kept.append(k)
            self.kept.append(k)
            self.disk.touch(key)
        except Exception as exc:
            self.broken = exc
            raise
        self.kstats["disk_hits"] += 1
        self.kstats["disk_tokens"] += n
        self.kstats["restore_ms"] += 1e3 * (time.perf_counter() - t0)

    def _persist(self) -> None:
        """Both ranks, shutting down: every kept state written to disk (none dropped: nothing runs after this)."""

        for k in list(self.kept):
            self._spill(k)
        self.disk.drain()

    def shutdown(self) -> list[Stream]:
        """Rank 0, on the scheduler's thread when the server stops: the live streams ended, every kept state written
        to the NVMe tier on both ranks, and rank 1's ``follow`` released. Returns the streams ended."""

        live = [*self.streams.values(), *self.filling]
        if self.broken is None:
            self.finish(live)
            if self.disk is not None:
                self._send([PERSIST])
                self._persist()
                print(f"[tensorfold] kept prompts written to NVMe: {self.disk.describe()}", flush=True)
            self._send([])
        return live

    def _queue(self, s: Stream, base: int = -1, size: int = -1, eid: int = -1, mode: int = FRESH,
               k: Kept | None = None, m: int = 0) -> None:
        s.acc = list(self.prior0 if self.reset_prior else self.prior)
        if self.copy is not None:                  # copy drafts: the index (at the prompt's end), the commit rule
            s.copies, s.cbase = None, 0
            s.cfrac, s.cmiss, s.cwait = 1.0, 0, 0  # running kept share, copy rounds missed in a row, rounds to wait
            s.copy_rounds = s.copy_accepted = 0
        s.slot = self.free.pop(0)
        s.pos = -1                                 # prompt tokens prefilled so far (-1: not started)
        if base >= 0:                              # the stream's extent of the shared pool (rank 1: rank 0's place)
            e = self.e
            if mode == TAKEOVER:                   # a kept state's free extent: its rows are the stream's prefix
                x = k.x
                if x.owner is not None or x.base != base:
                    raise RuntimeError("the ranks disagree on a kept prompt's extent")
                if size > x.size:
                    self.pool.resize(x, size)
                x.owner = s.sid
            else:
                x = self.pool.add(base, size, owner=s.sid, eid=eid)
            self.ext[s.sid] = x
            if k is not None:
                if mode == COPY:
                    e.copy_rows(k.x.base, x.base, m)
                    self.kstats["copies"] += 1
                else:
                    self.kstats["takeovers"] += 1
                e.bind(s.slot, x.base, x.size, ids=list(s.prompt[:m]))
                e.select_slot(s.slot)
                e.load_window(k.bank, s.slot)
                e.ring_from[s.slot] = k.vs
                e.deep_from[s.slot] = k.vd
                self.kept.remove(k)                # most recently used
                self.kept.append(k)
                for c in [c for c in x.kept if c.n > m]:
                    # rows the stream overwrites (the hit too, its window loaded); not spilled when this prompt is
                    # that state again (a regenerated reply: kept again, the same bytes, at the prompt's end)
                    self._drop(c, spill=not (c.n == len(s.prompt) and np.array_equal(c.ids, s.prompt)))
                s.pos = s.cached = m
                self.kstats["hits"] += 1
                self.kstats["cached"] += m
            else:
                e.bind(s.slot, x.base, x.size)
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        s = next_fill(self.filling)
        busy = any(not x.done for x in self.streams.values())
        rows = self.step_rows if busy else self.alone_rows
        self._send([FILL, s.sid, rows])
        first = self._step(s, rows)
        if first is None:
            return []
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _step(self, s: Stream, rows: int) -> int | None:
        """Prefill the next ``rows`` prompt tokens in the stream's slot (resuming a kept or live state first, so the
        step ends past what was resumed); at the prompt's end, keep its state and sample the first token."""

        e = self.e
        t0 = time.perf_counter()
        try:
            e.select_slot(s.slot)
            n = len(s.prompt)
            if s.pos < 0 and self.pool is not None:   # (a shared pool resumes kept states at admission)
                e.reset()
                s.pos = s.cached = 0
            if s.pos < 0:                          # the first step: resume what the slot or the pool holds
                cached = e.reusable(s.prompt) if s.draft else 0
                kept = e.pool.match(s.prompt) if s.draft and e.pool is not None else None
                if kept is not None and len(kept.ids) > cached:
                    e.load_prefix(kept.snapshot, kept.ids)
                    cached = len(kept.ids)
                elif cached:
                    del e.state.ids[cached:]
                else:
                    e.reset()
                s.pos, s.cached = cached, cached
            stop = min(n, s.pos + rows)
            # kept at the prompt's end and, for prompts that will share less, at its last chunk start (a long
            # document asked a long new question resumes there); long prompts also at 3/4 and 7/8 (``_trim``)
            keep = self.kept_on and s.draft
            points = [(n - 1) // MAX_ROWS * MAX_ROWS]
            if KEEP_SHRINK and n >= KEEP_SHRINK_MIN:
                points += [n * a // b // MAX_ROWS * MAX_ROWS for a, b in ((3, 4), (7, 8))]
            points = sorted({p for p in points if keep and p >= KEEP_MIN and s.cached < p < n - (DRING - WINDOW_ROWS)})
            # (no <= ROWS-row call before a split: prompt arithmetic)
            split = next((p for p in points if s.pos + ROWS < p < stop), None)
            if split is not None:
                stop = split
            # a call bounds its chunks by the next kept point at or after its end, so the state kept there is exact in
            # every layer (rows' values do not depend on it: the deep layers keep only their windows)
            if PREFILL_AHEAD and stop < n:                  # the next step's first chunk, read during this one's last
                nxt = min(n, stop + rows, next((p for p in points if stop + ROWS < p), n))
                p1 = len(e.state.ids) + stop - s.pos
                e.ahead_next = (list(e.state.ids) + list(s.prompt[s.pos:nxt]), p1, min(MAX_ROWS, nxt - stop))
            logits = e.prefill(s.prompt[s.pos:stop], final=next((p for p in points if p >= stop), n))
            s.pos = stop
            if split is not None:
                self._keep(s, self.ext[s.sid], e.state.ids, e.ring_from[s.slot])
            if stop < n:
                return None
            if keep and n >= KEEP_MIN and s.cached < n:
                self._keep(s, self.ext[s.sid], e.state.ids, e.ring_from[s.slot])
            if s.draft and self.pool is None:
                e.keep_prompt(s.prompt)
            if n - s.cached > RELEASE_AFTER:       # a long prompt's transient buffers back to the system (both
                self._release()                    # ranks: the unified memory the OS and the next prompt share)
            last = logits[-1:]
            if s.constraint is not None:
                last = s.constraint.mask(last.float().clone())
            first = sample_rows(last, [n], s.sampling)[0]
        except Exception as exc:
            self.broken = exc
            raise
        finally:
            s.prefill_s += time.perf_counter() - t0
        s.context = list(s.prompt)
        if self.copy is not None and s.draft and s.constraint is None:
            self._copies(s)                        # (off the rounds: a long prompt's window to int64 once)
        s.started = time.perf_counter()
        self.filling = [x for x in self.filling if x is not s]
        self.streams[s.sid] = s
        return first

    def _release(self) -> None:
        """The allocator's cached free blocks back to the system (graph pools and live tensors stay), and the host
        heap's free pages (rank 0 tokenized the request's text: a long prompt leaves ~1 GiB in glibc's arenas)."""

        from .engine import available_bytes

        log = os.environ.get("TF_DSV41_MEMLOG") == "1"
        before = torch.cuda.memory_reserved() if log else 0
        t0 = time.perf_counter()
        # the CUDA allocator's blocks go back only when memory runs low: handed back, they return as scattered
        # pages, and the next prompt's ~1 GiB of buffers then waits on reclaim and compaction for large blocks
        # (8K prompts ran 650-1000 tok/s instead of ~1750 with a release after every long prompt)
        if available_bytes() < RELEASE_BELOW:
            torch.cuda.empty_cache()
        t1 = time.perf_counter()
        try:
            import ctypes

            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
        t2 = time.perf_counter()
        if log:
            print(f"[memlog r{self.rank}] prompt released {(before - torch.cuda.memory_reserved()) / 2 ** 30:.2f} GiB; "
                  f"available {available_bytes() / 2 ** 30:.2f} GiB; empty_cache {1e3 * (t1 - t0):.0f} ms, "
                  f"malloc_trim {1e3 * (t2 - t1):.0f} ms", flush=True)

    # -- rounds ---------------------------------------------------------------------------------------------------
    def _plan(self, live: list[Stream]) -> list[tuple[int, int]]:
        """(sid, drafts) a stream: drafts while the round's rows fit (a grammar's stream: only with DRAFT_GRAMMAR); a
        copy round's drafts + COPY_FLAG."""

        return [(s.sid, k) for s, k in zip(live, self._allocate(live))]

    @torch.no_grad()
    def round(self) -> list[Stream]:
        self._check()
        tr = time.perf_counter()
        busy = any(not x.done and not x.waiting for x in self.streams.values())   # (waiting ones: fill goes on)
        share = self.fill_share / max(1e-6, 1.0 - self.fill_share)
        done = []
        if self.filling and (not busy or self.t_fill <= share * self.t_decode):   # time-sliced while others decode
            done = self._fill()
            if busy:
                self.t_fill += time.perf_counter() - tr
        if self.prof is not None and self.rank == 0:
            self.prof["fill"] += time.perf_counter() - tr
        live = [s for s in self.streams.values() if not s.done]
        if self.rank == 0:
            done += self._make_room(live)
        live = [s for s in live if not s.done and not s.waiting and s not in self.yielded]
        if not live:
            return done
        rs = self.rsplit
        if rs is not None:
            rs.begin(tr)
            rs.mark("fill")
        plan = self._plan(live)
        rs is not None and rs.mark("plan")
        self._send([ROUND, len(plan), *[x for item in plan for x in item]])
        rs is not None and rs.mark("send")
        td = time.perf_counter()
        news = self._verify(plan)
        if self.filling:
            self.t_decode += time.perf_counter() - td
        else:
            self.t_fill = self.t_decode = 0.0                  # nothing waits to fill: the shares start over
        for s, new in zip(live, news):
            if s.error is None:
                s.take(new, self._ends(s))
            else:
                s.done, s.finished = True, time.perf_counter()
        if rs is not None:
            rs.mark("emit")
            rs.end(sum(k & (COPY_FLAG - 1) for _, k in plan) + len(plan))
        if self.prof is not None and self.rank == 0:
            self.prof["round"] += time.perf_counter() - tr
        return done + [s for s in live if s.done]

    def _verify(self, plan: list[tuple[int, int]]) -> list[list[int]]:
        """One forward over every planned stream's pending token and drafts; returns each stream's new tokens (kept
        drafts, then its next token) and rolls rejected rows back in its slot."""

        e = self.e
        try:
            rows, spans = [], []
            t0 = time.perf_counter()
            # (sid, drafts, copied): a copy round's k + COPY_FLAG decoded once, here; spans carry it on
            plan = [(sid, k & (COPY_FLAG - 1), k >= COPY_FLAG) for sid, k in plan]
            want = [(sid, k) for sid, k, copied in plan if k and not copied]
            rs = self.rsplit
            rs is not None and want and rs.event("d0")
            batched = getattr(e.drafter, "multi_graphs", None) if e.drafter is not None else None
            proposals: dict[int, list[int]] = {}
            if want and batched and len(want) in batched:     # one drafting pass for every drafting stream
                items = [(self.streams[sid].slot, self.streams[sid].out[-1], len(e.views[self.streams[sid].slot].ids))
                         for sid, _ in want]
                proposals = {sid: d for (sid, _), d in zip(want, e.drafter.propose_multi(items))}
            for sid, k, copied in plan:
                s = self.streams[sid]
                pending = s.out[-1]
                drafts: list[int] = []
                if copied:                                 # both ranks: the same search of the same context
                    if self.copy is None:
                        raise RuntimeError("the ranks disagree on copy drafts (TF_MULTI_COPY)")
                    drafts = self._copies(s).propose(k)[:k]
                    if len(drafts) != k:
                        raise RuntimeError(f"stream {sid}: a copy proposal of {len(drafts)} tokens, planned {k}")
                elif k and sid in proposals:
                    drafts = proposals[sid][:k]
                elif k:
                    e.select_slot(s.slot)
                    drafts = e.drafter.propose(pending, len(e.views[s.slot].ids))[:k]
                if s.constraint is not None:
                    s.constraint.advance([pending])
                spans.append((len(rows), len(drafts) + 1, len(e.views[s.slot].ids), k, copied))
                rows += [(s.slot, t) for t in [pending, *drafts]]
            if self.check and self.gather is not None and any(c for *_, c in spans):
                self._agree_copies(plan, spans, rows)
            if rs is not None:
                want and rs.event("d1")
                rs.mark("draft")
                e._rsplit = rs                             # step_multi marks its parts and the window's GPU span
            t1 = time.perf_counter()
            logits, greedy = e.step_multi(rows)
            if rs is not None:
                e._rsplit = None
            if self.check and len(rows) > 1:               # debug: row 0 of each stream alone, at the same position
                main = logits[:len(rows)].float().clone()
                for (sid, _, _), (r0, nrows, p0, _, _) in zip(plan, spans):
                    s = self.streams[sid]
                    ids = e.views[s.slot].ids
                    keep = ids[p0:]
                    del ids[p0:]
                    alone, g1 = e.step_multi([(s.slot, rows[r0][1])])
                    a = alone[0].float()
                    if self.rank == 0:
                        d = (a - main[r0]).abs().max().item()
                        top = torch.topk(main[r0], 2).values.tolist()
                        print(f"[check] sid {sid} pos {p0} rows {len(rows)}: max|diff| {d:.3g}, argmax multi "
                              f"{int(main[r0].argmax())} alone {g1[0]}, top2 gap {top[0] - top[1]:.3g}", flush=True)
                    del ids[p0:]
                    ids.extend(keep)
            if self.prof is not None and self.rank == 0:
                p = self.prof
                p["rounds"] += 1
                p["draft"] += t1 - t0
                p["verify"] += time.perf_counter() - t1
                p["rows"] += len(rows)
                p["streams"] += len(plan)
                if p["rounds"] % 50 == 0:
                    n = p["rounds"]
                    mp = getattr(e, "_mprof", None) or {}
                    if mp.get("n"):
                        print("[multi] step_multi: " + ", ".join(f"{k} {1e3 * mp[k] / mp['n']:.1f} ms" for k in
                                                                 ("hash", "gather0", "gather1", "wait")), flush=True)
                    print(f"[multi] {n} rounds: draft {1e3 * p['draft'] / n:.1f} ms, verify {1e3 * p['verify'] / n:.1f} ms, "
                          f"post {1e3 * p['post'] / n:.1f} ms, fill {1e3 * p['fill'] / n:.1f} ms, round "
                          f"{1e3 * p['round'] / n:.1f} ms, {p['rows'] / n:.1f} rows, {p['streams'] / n:.1f} streams"
                          + (f"; copy rounds {p['copy_rounds']} kept {p['copy_accepted']} drafts"
                             if "copy_rounds" in p else "")
                          + (f"; kept {len(self.kept)} {self.kstats}" if self.kept_on else ""), flush=True)
            t2 = time.perf_counter()
            news = []
            for (sid, _, _), (r0, nrows, p0, k, copied) in zip(plan, spans):
                s = self.streams[sid]
                if s.sampling is not None and s.sampling.temperature > 0 or s.constraint is not None:
                    block = logits[r0:r0 + nrows].float()
                    if s.constraint is not None:            # each verify row masked by the grammar after its path
                        chain = [t for _, t in rows[r0:r0 + nrows]]
                        block = s.constraint.mask(block.clone(), s.constraint.window(chain, [-1, *range(nrows - 1)]))
                    target = sample_rows(block, [p0 + 1 + j for j in range(nrows)], s.sampling)
                else:
                    target = greedy[r0:r0 + nrows]
                drafts = [t for _, t in rows[r0 + 1:r0 + nrows]]
                m = 0
                while m < len(drafts) and drafts[m] == target[m]:
                    m += 1
                del e.views[s.slot].ids[p0 + 1 + m:]           # rejected rows: overwritten later
                if copied:                                     # only the copy rule learns from copy rounds
                    self._learn_copy(s, k, m)
                elif k:
                    self._learn(s, k, m)
                elif DRAFT_RELAX > 0:
                    s.acc = [a + DRAFT_RELAX * (p - a) for a, p in zip(s.acc, self.prior0)]
                if self.copy is not None and not copied and s.cwait > 0:
                    s.cwait -= 1
                if s.constraint is not None and m:
                    s.constraint.advance(drafts[:m])
                s.counted(nrows)
                new = drafts[:m] + [target[m]]
                ends = self._ends(s)                           # a kept draft can be the end token: stop at it
                cut = next((j + 1 for j, t in enumerate(new) if t in ends), len(new))
                news.append(new[:min(cut, max(1, s.count - len(s.out)))])
            rs is not None and rs.mark("post")
            if self.prof is not None and self.rank == 0:
                self.prof["post"] += time.perf_counter() - t2
            return news
        except Exception as exc:
            self.broken = exc
            raise

    # -- completion -----------------------------------------------------------------------------------------------
    def finish(self, done: list[Stream]) -> None:
        if done:
            self._send([DONE, len(done), *[s.sid for s in done]])
            for s in done:
                self._finish(s.sid)

    def _finish(self, sid: int) -> None:
        self.yielded = [x for x in self.yielded if x.sid != sid]
        s = self.streams.pop(sid, None)
        if s is None:
            s = next((x for x in self.filling if x.sid == sid), None)
            if s is not None:
                self.filling.remove(s)
        x = self.ext.pop(sid, None)
        if x is not None:                          # (a reply's decoded rows are not kept: a decode graph's rows
            x.owner = None                         # can differ from the prompt chunk's that would recompute them)
            self._settle(x)
        if s is not None and s.slot not in self.free:
            self.free.append(s.slot)
            self.free.sort()

    def drop(self) -> list[Stream]:
        live = list(self.streams.values()) + self.filling     # done ones too: their slots and extents go back
        self.kept_on = False                                  # (and nothing kept from a failed round)
        for s in live:
            self._finish(s.sid)
        for k in list(self.kept):
            self._drop(k, spill=False)
        if self.broken is None:
            self.broken = RuntimeError("a round failed")
        return live

    @torch.no_grad()
    def follow(self) -> None:
        """Rank 1: mirror rank 0's admissions, prefill steps, rounds and completions, forever."""

        while True:
            msg = self.share(None)
            if not msg:                                # rank 0 has stopped (tests)
                return
            if msg[0] == ADMIT:
                sid, count, draft, stop_eos = msg[1:5]
                words, npacked = msg[5:5 + SAMPLING_WORDS], msg[5 + SAMPLING_WORDS]
                base, size, eid, kid, m, mode = msg[6 + SAMPLING_WORDS:12 + SAMPLING_WORDS]
                if (base >= 0) != (self.pool is not None):
                    raise RuntimeError("the ranks disagree on the shared cache pool")
                k = self._kid(kid) if kid >= 0 else None
                s = Stream(self.share(None), count, unpack_sampling(words), draft=bool(draft),
                           stop_eos=bool(stop_eos), sid=sid)
                if npacked:
                    from tensorfold.engine import grammar

                    s.constraint = grammar.compiler(self, self.model_dir, self.eos).follow(self.share(None))
                self._queue(s, base, size, eid, mode, k, m)    # rank 0's placement, replayed
            elif msg[0] == EVICT:
                self._drop(self._kid(msg[1]))
            elif msg[0] == GROW:
                self._resize(msg[1], msg[2])
            elif msg[0] == MOVE:
                self._move(msg[1], msg[2], msg[3])
            elif msg[0] == FILL:
                s = next(x for x in self.filling if x.sid == msg[1])
                first = self._step(s, msg[2])
                if first is not None:
                    s.out.append(first)
            elif msg[0] == ROUND:
                plan = [(msg[2 + 2 * i], msg[3 + 2 * i]) for i in range(msg[1])]
                for (sid, _), new in zip(plan, self._verify(plan)):
                    self.streams[sid].out.extend(new)
            elif msg[0] == DONE:
                for sid in msg[2:2 + msg[1]]:
                    self._finish(sid)
            elif msg[0] == RESTORE:
                from .kvdisk import ints_key

                if self.disk is None:
                    raise RuntimeError("the ranks disagree on the NVMe tier")
                self._restore(ints_key(msg[1], msg[2]), *msg[3:10])
            elif msg[0] == PERSIST:
                self._persist()
            elif msg[0] == IDLE:                       # wait for the next message on the CPU (share: the doorbell)
                if self.bell is None:
                    raise RuntimeError("the ranks disagree on the idle doorbell")
                self.bell.arm()


def stream_stats(s: Stream) -> dict[str, Any]:
    out = s.stats()
    if hasattr(s, "copy_rounds"):                  # copy drafts on (TF_MULTI_COPY)
        out.update({"copy_rounds": s.copy_rounds, "copy_accepted": s.copy_accepted})
    return out
