"""DSpark's Markov steps (the drafter's last part) as fixed-order kernels inside the drafter's graph: each rank scores
its vocabulary half, the bias rows of frequent tokens are cached, and one small gather a step picks the draft.

Adapted from bertholomus/TensorFold (branch deepseek-v41-tp2, commit bd0024d, ``deepseek_v41/cuda/markov.py``;
Apache License 2.0, Copyright 2026 Albert Lee / BertholomusAI). Changed for this engine: the bias stays fp32 (our
loop's ``torch.mm(e.half(), head.T, out_dtype=F32)``, not an fp16 GEMM output widened), no PDL, our token list and
comm. See NOTICE.

Step i picks d_{i+1} = argmax_v (logits_i[v] + bias(d_i)[v]), bias(t) = markov_head @ fp16(markov_embed[t]):

- ``_bias``: for each row whose token has no cached row, every bias element as one tensor-core dot of the head row and
  e(t) (``tl.dot`` [64, rank] x [rank, 16], fp32 accumulation) into a staging row. The arithmetic of an element does
  not depend on the vocabulary range, the tile, the row count or the other rows. A launch whose rows all have cached
  rows only reads the tokens and slots.
- ``_score``: logit + bias (the token's cached row, else its staging row); each program's best (value, index) a row in
  torch.argmax's order (NaN above everything, then the larger value, ties to the lower index: a total order, so any
  reduction tree gives the same best). ``_finish``: the row's best of the programs'.
- Cached rows (``TF_DSV41_MARKOV_CACHE`` = K, default 256; 0: none): bias(t) of the first K tokens of
  ``markov_tokens.TOKENS``, made at load time by ``_bias`` itself (a cached row has the computed bits). Which tokens
  are cached changes the time, never a draft. K x V / world fp32 a rank (66 MB at 256).
- Vocabulary split (TP > 1): each rank scores its half of the drafter head's logits (no logits gather) with its half of
  the head's rows; the ranks' bests, (value, index) a row, go through one 16-byte-a-row gather a step, and ``_pick``
  takes the best in the same order (rank 0's, the lower indices, wins a tie). The best of the union under a total
  order is the best of the halves' bests, and every rank picks from the same gathered bytes.

``TF_DSV41_MARKOV=0``: the torch loop (``dspark.py``).
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

ON = os.environ.get("TF_DSV41_MARKOV", "1") != "0"
CACHE_ROWS = int(os.environ.get("TF_DSV41_MARKOV_CACHE") or 256)
BN = 64                 # vocabulary entries a dot (V / world is a multiple of it)
SUB = 4                 # dots a bias program, one after another
BC = 1024               # vocabulary entries a scoring program
NP = 16                 # rows a launch at most (tl.dot's smallest N)
BIG = tl.constexpr(1 << 30)


@triton.jit
def _better(v1, i1, v2, i2):
    """(v1, i1) before (v2, i2) in torch.argmax's order: NaN above everything, then the larger value, ties (and NaN
    against NaN) to the lower index."""

    n1 = v1 != v1                                          # noqa: PLR0124 - NaN test
    n2 = v2 != v2                                          # noqa: PLR0124
    gt = (v1 > v2) | ((v1 == v2) & (i1 < i2))
    return (n1 & (~n2 | (i1 < i2))) | (~n1 & ~n2 & gt)


@triton.jit
def _best(v1, i1, v2, i2):
    t = _better(v1, i1, v2, i2)
    return tl.where(t, v1, v2), tl.where(t, i1, i2)


@triton.jit(do_not_specialize=["t_stride", "o_row", "n_rows", "n_cols", "seg0"])
def _bias(TOK, t_stride, SLOT, EMB, HEAD, OUT, o_row, n_rows, n_cols, seg0, BN: tl.constexpr, SUB: tl.constexpr,
          RANK: tl.constexpr, NP: tl.constexpr):
    """Program (t, s): for each row whose token has no slot, OUT[r, s * n_cols + c] = head[v] . fp16(emb[tok]) (fp32)
    for SUB tiles of BN columns from (t * SUB) * BN (vocabulary entries v = (seg0 + s) * n_cols + c)."""

    t = tl.program_id(0)
    s = tl.program_id(1)
    r = tl.arange(0, NP)
    rm = r < n_rows
    tok = tl.load(TOK + r * t_stride, mask=rm, other=0)
    slot = tl.load(SLOT + tok, mask=rm, other=0)
    miss = rm & (slot < 0)
    if tl.max(miss.to(tl.int32), axis=0) > 0:
        k = tl.arange(0, RANK)
        e = tl.load(EMB + tok[:, None].to(tl.int64) * RANK + k[None, :], mask=miss[:, None], other=0.0)
        e = e.to(tl.float16)
        for u in tl.static_range(SUB):
            c = (t * SUB + u) * BN + tl.arange(0, BN)
            cm = c < n_cols
            v = (seg0 + s) * n_cols + c
            w = tl.load(HEAD + v[:, None].to(tl.int64) * RANK + k[None, :], mask=cm[:, None], other=0.0)
            acc = tl.dot(w, tl.trans(e))                  # [BN, NP] fp32
            tl.store(OUT + r[None, :].to(tl.int64) * o_row + (s * n_cols + c)[:, None], acc,
                     mask=miss[None, :] & cm[:, None])


@triton.jit(do_not_specialize=["l_seg", "l_row", "t_stride", "c_row", "s_row", "n_rows", "n_cols", "seg0",
                               "n_part"])
def _score(LG, l_seg, l_row, TOK, t_stride, SLOT, CACHE, c_row, STAGE, s_row, PV, PI, n_rows, n_cols, seg0, n_part,
           BC: tl.constexpr):
    """Program (t, s): columns t * BC .. of segment s for each row: score = logit + bias (the token's cached row, else
    its staging row); the best (value, index) a row into PV / PI [rows, n_part]."""

    t = tl.program_id(0)
    s = tl.program_id(1)
    c = t * BC + tl.arange(0, BC)
    cm = c < n_cols
    cc = s * n_cols + c
    v = (seg0 + s) * n_cols + c
    idx = tl.where(cm, v, BIG)
    p = s * tl.num_programs(0) + t
    for r in range(n_rows):
        tok = tl.load(TOK + r * t_stride)
        slot = tl.load(SLOT + tok)
        hit = slot >= 0
        bc = tl.load(CACHE + slot.to(tl.int64) * c_row + cc, mask=cm & hit, other=0.0)
        bs = tl.load(STAGE + r * s_row + cc, mask=cm & (slot < 0), other=0.0)
        lg = tl.load(LG + s * l_seg + r * l_row + c, mask=cm, other=0.0)
        score = tl.where(cm, lg + tl.where(hit, bc, bs), float("-inf"))
        bv, bi = tl.reduce((score, idx), 0, _best)
        tl.store(PV + r * n_part + p, bv)
        tl.store(PI + r * n_part + p, bi)


@triton.jit(do_not_specialize=["n_part", "o_stride", "send"])
def _finish(PV, PI, n_part, OUT, o_stride, SEND, send, BP: tl.constexpr):
    """Row r's best of its n_part program bests: the token into OUT[r * o_stride] (int64), or (``send``) (value,
    index, 0, 0) fp32 into SEND[r] for the ranks' gather."""

    r = tl.program_id(0)
    p = tl.arange(0, BP)
    pm = p < n_part
    v = tl.load(PV + r * n_part + p, mask=pm, other=float("-inf"))
    i = tl.load(PI + r * n_part + p, mask=pm, other=BIG)
    bv, bi = tl.reduce((v, i), 0, _best)
    if send != 0:
        q = tl.arange(0, 4)
        val = tl.where(q == 0, bv, tl.where(q == 1, bi.to(tl.float32), 0.0))
        tl.store(SEND + r * 4 + q, val)
    else:
        tl.store(OUT + r * o_stride, bi.to(tl.int64))


@triton.jit(do_not_specialize=["n_rows", "o_stride"])
def _pick(G, n_rows, OUT, o_stride, WORLD: tl.constexpr):
    """Row r's token from the ranks' (value, index) bests [WORLD, rows, 4], in the same order."""

    r = tl.program_id(0)
    bv = tl.load(G + r * 4)
    bi = tl.load(G + r * 4 + 1).to(tl.int32)
    for w in tl.static_range(1, WORLD):
        v = tl.load(G + (w * n_rows + r) * 4)
        i = tl.load(G + (w * n_rows + r) * 4 + 1).to(tl.int32)
        bv, bi = _best(v, i, bv, bi)
    tl.store(OUT + r * o_stride, bi.to(tl.int64))


class Markov:
    """The Markov loop of the drafter on one rank: its vocabulary part of the head's rows (all of them at world 1),
    the cached bias rows and every token's slot (-1: none)."""

    def __init__(self, dw, comm, vocab: int, rows: int | None = None, tokens: list[int] | None = None) -> None:
        self.comm, self.world, self.rank = comm, comm.world, comm.rank
        self.V = vocab
        self.rank_dim = dw.markov_head.shape[1]
        self.n_cols = vocab // self.world                 # this rank's columns: the drafter head's vocabulary part
        assert self.n_cols * self.world == vocab and self.n_cols % BN == 0, (vocab, self.world)
        self.seg0 = self.rank
        self.head = dw.markov_head.contiguous()
        self.emb = dw.markov_embed.contiguous()
        self.b_tiles = triton.cdiv(self.n_cols, BN * SUB)
        self.s_tiles = triton.cdiv(self.n_cols, BC)
        self.n_part = self.s_tiles
        self.bp = triton.next_power_of_2(self.n_part)
        dev = self.head.device
        self.none = torch.full((vocab,), -1, dtype=torch.int32, device=dev)
        self.slot, self.cache, self.tokens = self.none, torch.zeros((1,), dtype=torch.float32, device=dev), []
        self._sc: dict[int, tuple] = {}
        rows = CACHE_ROWS if rows is None else int(rows)
        if rows > 0:
            if tokens is None:
                from .markov_tokens import TOKENS as tokens
            self.fill([t for t in dict.fromkeys(int(t) for t in tokens) if 0 <= t < vocab][:rows])

    def bias(self, tok: torch.Tensor, t_stride: int, slot: torch.Tensor, out: torch.Tensor, o_row: int, n: int) -> None:
        """``_bias`` for n rows (tokens tok[r * t_stride]) into out [n, n_cols] fp32 (the rows without a slot)."""

        _bias[(self.b_tiles, 1)](tok, t_stride, slot, self.emb, self.head, out, o_row, n, self.n_cols, self.seg0,
                                 BN=BN, SUB=SUB, RANK=self.rank_dim, NP=NP, num_warps=4)

    def fill(self, tokens: list[int]) -> None:
        """The cached rows of ``tokens``: ``_bias``'s own rows (computed with the slot table of none). At load time,
        never under graph capture (it synchronizes)."""

        dev = self.head.device
        k = len(tokens)
        cache = torch.empty((max(k, 1), self.n_cols), dtype=torch.float32, device=dev)
        tok = torch.tensor(tokens, dtype=torch.int64, device=dev)
        for r0 in range(0, k, NP):
            self.bias(tok[r0:], 1, self.none, cache[r0], self.n_cols, min(NP, k - r0))
        slot = torch.full((self.V,), -1, dtype=torch.int32, device=dev)
        if k:
            slot[tok] = torch.arange(k, dtype=torch.int32, device=dev)
        torch.cuda.synchronize()
        self.cache, self.slot, self.tokens = cache, slot, tokens

    def _scratch(self, n: int, dev) -> tuple:
        sc = self._sc.get(n)
        if sc is None:                                     # each row count its own: graphs never share them
            sc = self._sc[n] = (torch.empty((n, self.n_part), dtype=torch.float32, device=dev),
                                torch.empty((n, self.n_part), dtype=torch.int32, device=dev),
                                torch.zeros((n, 4), dtype=torch.float32, device=dev),
                                torch.empty((n, self.n_cols), dtype=torch.float32, device=dev))
        return sc

    def local_best(self, lg: torch.Tensor, out: torch.Tensor, block: int, i: int) -> torch.Tensor | None:
        """Step i on this rank's columns: the token into out[:, i + 1] (world 1), else this rank's (value, index)
        [M, 4] fp32 for the gather (returned)."""

        M = out.shape[0]
        assert M <= NP and out.stride(1) == 1 and lg.is_contiguous()
        nc = self.n_cols
        pv, pi, send, st = self._scratch(M, out.device)
        ts = out.stride(0)
        tok = out[:, i]
        self.bias(tok, ts, self.slot, st, nc, M)
        _score[(self.s_tiles, 1)](lg.view(-1)[i * nc:], 0, block * nc, tok, ts, self.slot, self.cache, nc, st, nc,
                                  pv, pi, M, nc, self.seg0, self.n_part, BC=BC, num_warps=4)
        split = self.world > 1
        _finish[(M,)](pv, pi, self.n_part, out[:, i + 1], ts, send, int(split), BP=self.bp, num_warps=4)
        return send if split else None

    def pick(self, g: torch.Tensor, out: torch.Tensor, i: int) -> None:
        """out[:, i + 1] from every rank's (value, index) bests [world, M, 4]."""

        M = out.shape[0]
        _pick[(M,)](g, M, out[:, i + 1], out.stride(0), WORLD=g.shape[0], num_warps=1)

    def steps(self, lg: torch.Tensor, out: torch.Tensor, block: int, steps: int) -> None:
        """out [M, steps + 1] int64 (out[:, 0] each stream's anchor) -> out[:, 1:] the drafts. lg: this rank's
        drafter-head logits [M * block, n_cols] fp32, stream m's step i at row m * block + i."""

        for i in range(steps):
            send = self.local_best(lg, out, block, i)
            if send is not None:
                self.pick(self.comm.partials(send), out, i)          # [world, M, 4] fp32 in rank order
