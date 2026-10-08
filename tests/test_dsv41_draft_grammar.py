"""Drafted grammar streams in DeepSeek-V4.1's concurrent rounds (TF_DSV41_DRAFT_GRAMMAR), host side: with the switch
on, a constrained stream's verify rows are each masked by the grammar after the row's path (``Constraint.window``), so
its reply equals the serial (one row a round) reply token for token, greedy and seeded, through the </think>
activation, with grammar-illegal, early-</think> and stop-token drafts in the windows; both ranks stay in step; with
the switch off, constrained streams never draft. A fake engine whose logits are a pure function of a row's path, a real
xgrammar matcher over a toy character vocabulary; sampling on the host (``choose_rows``, the CUDA rule's CPU form)."""

from __future__ import annotations

import random
import zlib

import numpy as np
import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")
xgr = pytest.importorskip("xgrammar")

from tensorfold.cuda.streams import Stream
from tensorfold.engine import grammar
from tensorfold.engine.exact_sampling import Sampling, choose_rows
from tensorfold.families.deepseek_v41.cuda import multi as M
from tests.test_dsv41_kept import FakeEngine, pair, run, same
from tests.test_dsv41_multi_copy import curve

STOP = 0
VOCAB = ["", *[chr(c) for c in range(0x21, 0x7f)], "</think>", "<think>"]
V = len(VOCAB)
THINK_END, THINK_OPEN = V - 2, V - 1
TOK = {s: i for i, s in enumerate(VOCAB)}
SPEC = grammar.Spec("regex", r"\{[a-f]:[0-9](,[a-f]:[0-9]){0,3}\}")
GRAMMARS = grammar.Grammars(xgr.TokenizerInfo(VOCAB, xgr.VocabType.RAW, vocab_size=V, stop_token_ids=[STOP]))
COMPILED = GRAMMARS.compile(SPEC)
ILLEGAL = [TOK[c] for c in "#xyz~Q"] + [THINK_OPEN]        # never in the regex
DRAFTS = 5


def cpu_sample_rows(logits, positions, sampling):
    """``sample_rows`` on the host: argmax, or the exact position-keyed rule over the whole row (-inf: masked)."""

    if logits.ndim != 2 or len(positions) != logits.shape[0]:
        raise ValueError("expected [rows, vocab] logits and one position per row")
    if sampling is None or sampling.temperature <= 0:
        return [int(x) for x in logits.argmax(dim=-1).tolist()]
    values = logits.float().cpu().numpy()
    ids = np.broadcast_to(np.arange(values.shape[1], dtype=np.int64), values.shape)
    return choose_rows(values, ids, positions, sampling)


@pytest.fixture(autouse=True)
def host_sampling(monkeypatch):
    monkeypatch.setattr(M, "sample_rows", cpu_sample_rows)
    monkeypatch.setattr(M, "MULTI_COPY", False)


@pytest.fixture
def draft_grammar(monkeypatch):
    def set_(on: bool):
        monkeypatch.setattr(M, "DRAFT_GRAMMAR", on)
    return set_


def model_logits(ids: list[int]) -> torch.Tensor:
    """The next token's logits after ``ids`` (a pure function of the path): random over the vocabulary; while a
    reply thinks (``<think>`` in the prompt, no ``</think>`` since), </think> is held back for a few tokens and then
    all but forced, and the stop token never comes."""

    seed = zlib.crc32(np.asarray(ids[-6:] + [len(ids)], dtype=np.int64).tobytes())
    x = np.random.default_rng(seed).normal(0.0, 2.5, V).astype(np.float32)
    if THINK_OPEN in ids:
        j = len(ids) - 1 - ids[::-1].index(THINK_OPEN)
        if THINK_END not in ids[j:]:
            since = len(ids) - j - 1
            x[STOP] = -40.0
            x[THINK_END] = 40.0 if since >= 5 + zlib.crc32(bytes(ids[:4])) % 5 else -40.0
    return torch.from_numpy(x)


class Drafter:
    """Proposes from the stream's reference reply (``refs``, by the prompt's first 8 tokens), mixing in tokens the
    grammar rejects, an early </think> and the stop token; a function of the context only (both ranks agree)."""

    def __init__(self, e, refs: dict, mix: float = 0.5) -> None:
        self.e, self.refs, self.mix = e, refs, mix
        self.calls = self.illegal = 0

    def propose(self, pending: int, n: int) -> list[int]:
        ids = self.e.views[self.e.slot].ids
        assert len(ids) == n
        self.calls += 1
        plen, reply = self.refs.get(tuple(ids[:8]), (n, []))
        at = n - plen + 1                             # the reply index of the token after ``pending``
        rng = random.Random(zlib.crc32(np.asarray(ids[-8:] + [pending, n], dtype=np.int64).tobytes()))
        out = []
        for j in range(DRAFTS):
            r = rng.random()
            good = reply[at + j] if at + j < len(reply) else STOP
            if r < self.mix:
                out.append(good)
            elif r < self.mix + 0.2:
                out.append(rng.choice(ILLEGAL))
                self.illegal += 1
            elif r < self.mix + 0.3:
                out.append(THINK_END)
            elif r < self.mix + 0.4:
                out.append(STOP)
            else:
                out.append(rng.randrange(V))
        return out


def engine_class(refs: dict, mix: float = 0.5):
    class GrammarEngine(FakeEngine):
        """FakeEngine with rounds and real logits: ``model_logits`` of each row's path (row-invariant)."""

        def __init__(self, *a, **k) -> None:
            super().__init__(*a, **k)
            self.c = type("C", (), {"eos_token_id": STOP})()
            self.drafter = Drafter(self, refs, mix)

        def prefill(self, tokens, final=None):
            super().prefill(tokens, final)
            out = torch.zeros((len(tokens), V))
            out[-1] = model_logits(self.state.ids)
            return out

        def step_multi(self, rows):
            logits = []
            for slot, token in rows:
                ids = self.views[slot].ids
                base, _ = self.extents[slot]
                self.arena[base + len(ids)] = token
                ids.append(token)
                logits.append(model_logits(ids))
            logits = torch.stack(logits)
            return logits, [int(t) for t in logits.argmax(dim=-1).tolist()]

    return GrammarEngine


# (prompt, thinks, constrained, count, stop_eos)
def jobs(seed: int):
    rng = random.Random(seed)

    def prompt(n, think):
        p = [rng.randrange(1, THINK_END) for _ in range(n)]
        return p + [THINK_OPEN] if think else p
    return [(prompt(20, True), True, True, 60, True),           # thinks, then the grammar from </think> on
            (prompt(14, False), False, True, 40, True),         # the grammar from the first token
            (prompt(17, True), True, False, 30, True),          # plain: drafts as before beside them
            (prompt(11, True), True, True, 45, False)]          # ignore_eos: past the grammar's stop, unmasked


def make_stream(job, sampling, draft):
    p, think, constrained, count, stop_eos = job
    s = Stream(list(p), count, sampling, draft=draft, stop_eos=stop_eos)
    if constrained:
        s.constraint = GRAMMARS.constraint(COMPILED, think_end=THINK_END if think else None, spec=SPEC)
    return s


def decoders(refs, mix=0.5, slots=4):
    d0, d1 = pair(slots=slots, engine=engine_class(refs, mix))
    for d in (d0, d1):
        d.drafts = DRAFTS
        d.prior0 = [M.DRAFT_PRIOR] * DRAFTS
        d.prior = list(d.prior0)
        d.grammars = GRAMMARS                         # rank 1 compiles the packed grammar with it
    curve(d0, drafts=DRAFTS)
    return d0, d1


def serve(d0, d1, streams, forced=None):
    """Decode ``streams`` together on both ranks; rank 1 compared with rank 0 after every round. ``forced``: a
    random draft count per stream and round (up to what the stream can take) in place of the allocator's."""

    plans = []
    if forced is not None:
        rng = random.Random(forced)
        real = d0._allocate

        def allocate(live):
            ks = []
            for s in live:
                room = min(d0.e.limit, d0.e.extents[s.slot][1]) - len(d0.e.views[s.slot].ids) - 1
                cap = max(0, min(DRAFTS, room, s.count - len(s.out) - 1))
                ks.append(rng.randint(0, cap) if s.draft and cap else 0)
            ks = [k if s.constraint is None or M.DRAFT_GRAMMAR else 0 for s, k in zip(live, ks)]
            real(live)                                # (its bookkeeping, as in a real round)
            plans.append([(s.sid, k) for s, k in zip(live, ks)])
            return ks
        d0._allocate = allocate
    else:
        real = d0._allocate

        def allocate(live):
            ks = real(live)
            plans.append([(s.sid, k) for s, k in zip(live, ks)])
            return ks
        d0._allocate = allocate
    for s in streams:
        run(d0, s)
    for _ in range(500):
        if all(s.done for s in streams):
            break
        d0.round()
        d1.sync()
        for s in streams:
            r = d1.streams[s.sid]
            assert r.out == s.out
            assert d1.e.views[r.slot].ids == d0.e.views[s.slot].ids
            if s.constraint is not None:
                assert r.constraint.active == s.constraint.active
                assert r.constraint.finished == s.constraint.finished
    assert all(s.done for s in streams)
    same(d0, d1)
    finished = [s.constraint.finished if s.constraint is not None else None for s in streams]
    rank1 = [d1.streams[s.sid].constraint.finished if s.constraint is not None else None for s in streams]
    assert finished == rank1
    d0.finish(streams)
    same(d0, d1)
    return plans


def ends_finished(c, out: list[int]) -> bool:
    """The reply's grammar is complete at its last token: the stream ends on the stop token before a round follows it
    (serial: never advanced past it) or after the round that kept it as a draft advanced through it."""

    if not c.finished and out and out[-1] == STOP:
        c.advance([STOP])
    return c.finished


def finish_at(out: list[int], think: bool) -> int:
    """Where a fresh matcher following the reply (from after </think>) takes the stop token, or -1."""

    m = xgr.GrammarMatcher(COMPILED)
    start = out.index(THINK_END) + 1 if think else 0
    for i in range(start, len(out)):
        assert m.accept_token(out[i]), (i, VOCAB[out[i]])
        if m.is_terminated():
            return i
    return -1


def reference(seed, sampling):
    """The serial replies: every stream one row a round (drafts off)."""

    d0, d1 = decoders({})
    streams = [make_stream(j, sampling, draft=False) for j in jobs(seed)]
    plans = serve(d0, d1, streams)
    assert all(k == 0 for plan in plans for _, k in plan)
    for s, (p, think, constrained, count, stop_eos) in zip(streams, jobs(seed)):
        if constrained:
            assert ends_finished(s.constraint, s.out)
            at = finish_at(s.out, think)
            assert at >= 0 and s.out[at] == STOP
            if stop_eos:
                assert at == len(s.out) - 1
            else:
                assert len(s.out) == count and at < count - 1     # it went on past the stop, unmasked
            if think:
                assert s.out.index(THINK_END) >= 5
    return [list(s.out) for s in streams]


SAMPLINGS = {"greedy": None, "seeded": Sampling(seed=1234, temperature=0.9, top_k=20, top_p=0.95),
             "seeded_wide": Sampling(seed=99, temperature=1.3, top_k=0, top_p=1.0)}


@pytest.mark.parametrize("sampling", list(SAMPLINGS))
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("mix", [0.9, 0.5, 0.0])
def test_drafted_constrained_replies_equal_the_serial_ones(draft_grammar, sampling, seed, mix):
    draft_grammar(True)
    smp = SAMPLINGS[sampling]
    want = reference(seed, smp)
    js = jobs(seed)
    refs = {tuple(p[:8]): (len(p), w) for (p, *_), w in zip(js, want)}
    d0, d1 = decoders(refs, mix)
    streams = [make_stream(j, smp, draft=True) for j in js]
    plans = serve(d0, d1, streams, forced=seed + 100)
    assert [s.out for s in streams] == want
    for s, w, (p, think, constrained, *_) in zip(streams, want, js):
        if constrained:
            assert ends_finished(s.constraint, s.out)
            assert finish_at(s.out, think) == finish_at(w, think) >= 0
            assert s.drafted > 0                      # it drafted under its grammar
    sids = {s.sid for s in streams if s.constraint is not None}
    assert any(k for plan in plans for sid, k in plan if sid in sids)
    if mix >= 0.5:                                    # kept drafts under the grammar
        assert sum(s.accepted for s in streams if s.constraint is not None) > 0
    assert d0.e.drafter.illegal > 0                   # grammar-illegal drafts were in the windows


@pytest.mark.parametrize("sampling", list(SAMPLINGS))
def test_allocator_driven_rounds_equal_the_serial_ones(draft_grammar, sampling):
    draft_grammar(True)
    smp = SAMPLINGS[sampling]
    want = reference(5, smp)
    js = jobs(5)
    refs = {tuple(p[:8]): (len(p), w) for (p, *_), w in zip(js, want)}
    d0, d1 = decoders(refs, 0.9)
    streams = [make_stream(j, smp, draft=True) for j in js]
    plans = serve(d0, d1, streams)
    assert [s.out for s in streams] == want
    sids = {s.sid for s in streams if s.constraint is not None}
    assert any(k for plan in plans for sid, k in plan if sid in sids)


@pytest.mark.parametrize("sampling", ["greedy", "seeded"])
def test_switch_off_constrained_streams_never_draft(draft_grammar, sampling):
    draft_grammar(False)
    smp = SAMPLINGS[sampling]
    want = reference(3, smp)
    js = jobs(3)
    refs = {tuple(p[:8]): (len(p), w) for (p, *_), w in zip(js, want)}
    d0, d1 = decoders(refs, 0.9)
    streams = [make_stream(j, smp, draft=True) for j in js]
    plans = serve(d0, d1, streams)
    assert [s.out for s in streams] == want
    sids = {s.sid for s in streams if s.constraint is not None}
    assert all(k == 0 for plan in plans for sid, k in plan if sid in sids)
    for s in streams:
        if s.constraint is not None:
            assert s.drafted == 0 and s.min_rows == 1
    plain = next(s for s in streams if s.constraint is None)
    assert plain.drafted > 0                          # the plain stream beside them still drafts


def test_allocate_gives_constrained_streams_drafts_only_with_the_switch(draft_grammar):
    d0, _ = decoders({})
    js = jobs(4)
    streams = [make_stream(j, None, draft=True) for j in js]
    for s in streams:
        run(d0, s)
    live = [s for s in streams]
    draft_grammar(False)
    off = d0._allocate(live)
    draft_grammar(True)
    on = d0._allocate(live)
    for s, a, b in zip(live, off, on):
        if s.constraint is not None:
            assert a == 0 and b > 0
        else:
            assert a > 0                          # (with rows shared, on may give it fewer)


def test_one_window_by_hand(draft_grammar):
    """One verify window: a legal draft, an illegal one, a draft after it; the rows from the illegal draft on are not
    masked (the parent row's mask never chooses it, so no kept path reaches them)."""

    draft_grammar(True)
    c = GRAMMARS.constraint(COMPILED, spec=SPEC)
    c.advance([TOK["{"], TOK["a"]])                          # _verify advances the pending token first
    chain = [TOK["a"], TOK[":"], TOK["#"], TOK["1"]]          # pending 'a', ':' legal, '#' not, '1'
    w = c.window(chain, [-1, 0, 1, 2])
    assert w.tokens == chain[:2] and w.rows == [0, 1]        # the path ends at the draft row 1's mask rejects
    block = c.mask(torch.zeros((4, V)), w)
    allowed = [[t for t in range(V) if block[r, t] == 0] for r in range(4)]
    assert allowed[0] == [TOK[":"]]
    assert allowed[1] == [TOK[x] for x in "0123456789"]
    assert TOK["#"] not in allowed[1]                        # so the kept drafts stop before '#'
    assert len(allowed[2]) == len(allowed[3]) == V           # rows on the rejected path: unmasked, never used
    # thinking: the grammar starts at </think> inside the window
    t = GRAMMARS.constraint(COMPILED, think_end=THINK_END, spec=SPEC)
    t.advance([TOK["q"]])
    w = t.window([TOK["q"], TOK["r"], THINK_END, TOK["{"], TOK["Q"]], [-1, 0, 1, 2, 3])
    assert w.tokens == [TOK["q"], TOK["r"], THINK_END, TOK["{"]] and w.rows == [2, 3]
    block = t.mask(torch.zeros((5, V)), w)
    assert (block[0] == 0).all() and (block[1] == 0).all()
    assert [x for x in range(V) if block[2, x] == 0] == [TOK["{"]]
    assert TOK["Q"] not in [x for x in range(V) if block[3, x] == 0]
    assert not t.active                                       # the window leaves the matcher where it was
