# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jay Leaton. The DeepSeek-V4.1-Flash family of TensorFold (Apache-2.0): see THIRD_PARTY_NOTICES.md.
# Modified for TensorFold dsv41-cuda: from tests/dsv41_dense3_emu.py (patches/0002), checked against our decode.cuh.
"""TF_EXL3_LANES without a GPU: a numpy emulation of ``cuda/exl3/lanes.cu`` against ``decode.cuh``.

- the lanes layout as array operations round-trips (a bit permutation of each strip), and the relayout kernels'
  statements (transcribed, one destination word) equal it;
- pair_frags' 16-bit states (take_bits, the tails shuffled from lane - 1, the 8 windows), for every lane, tile and
  step of a load group, equal decode.cuh lane_states' (lane_start, the lane's words, hi / mid / lo, window16) and
  ``format.states`` — so decode2 sees the same states and the B fragments are linear.cu's;
- a warp's walk reads each word of its K range once, in coalesced 512-byte chunks.

The layout and the kernel statements are adapted from jayleaton/deepseek-v41-tensorfold-spark tests/dsv41_dense3_emu.py
(MIT, Copyright (c) 2026 Jay Leaton; THIRD_PARTY_NOTICES.md)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.cuda.exl3 import format as fmt

U64 = np.uint64
K2S = (4, 6, 8, 10, 12)
G = 2


# -- bits <-> words (bit b of a word stream = bit 31 - b % 32 of word b // 32, decode.cuh's reading) ----------------
def bits_of(w: np.ndarray) -> np.ndarray:
    return np.unpackbits(np.ascontiguousarray(w.astype(np.uint32).astype(">u4")).view(np.uint8), axis=-1)


def words_of(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.packbits(b, axis=-1)).view(">u4").astype(np.uint32)


def strips_of(trellis: np.ndarray) -> np.ndarray:
    """int16 trellis [K/16, N/16, 16 bits] -> strips words uint32 [N/128, K/16, 8, 8 bits] (linear.strips)."""

    kt, nt, w16 = trellis.shape
    words = np.ascontiguousarray(trellis).view(np.uint32).reshape(kt, nt, w16 // 2)
    return np.ascontiguousarray(words.reshape(kt, nt // 8, 8, w16 // 2).transpose(1, 0, 2, 3))


def to_lanes(strips: np.ndarray, k2: int) -> np.ndarray:
    nb, kt = strips.shape[:2]
    n, GW = 4 * k2, G * k2
    own = bits_of(strips).reshape(nb, kt // G, G, 8, 32, n).transpose(0, 1, 4, 2, 3, 5)
    lw = words_of(own.reshape(nb, kt // G, 32, GW * 32))
    return np.ascontiguousarray(lw.reshape(nb, kt // G, 32, GW // 4, 4).transpose(0, 1, 3, 2, 4)).reshape(strips.shape)


def to_strips(lanes: np.ndarray, k2: int) -> np.ndarray:
    nb, kt = lanes.shape[:2]
    n, GW = 4 * k2, G * k2
    lw = lanes.reshape(nb, kt // G, GW // 4, 32, 4).transpose(0, 1, 3, 2, 4).reshape(nb, kt // G, 32, GW)
    own = bits_of(lw).reshape(nb, kt // G, 32, G, 8, n).transpose(0, 1, 3, 4, 2, 5)
    return words_of(own.reshape(nb, kt, 8, 32 * n)).reshape(lanes.shape)


def _bit(flat: np.ndarray, at: int, sb: int) -> int:
    return (int(flat[at + (sb >> 5)]) >> (31 - (sb & 31))) & 1


def to_lanes_word(src: np.ndarray, i: int, k2: int, KT: int) -> int:
    """lanes.cu to_lanes_kernel's statements for destination word i."""

    n = TW = 4 * k2
    strip_words, gw = KT * 32 * k2, G * 32 * k2
    strip, r = divmod(i, strip_words)
    gi, qd = divmod(r, gw)
    c, L = qd // 128, (qd % 128) // 4
    w = 4 * c + qd % 4
    out = 0
    for b in range(32):
        lb = 32 * w + b
        tt, ob = divmod(lb, n)
        kt = gi * G + tt // 8
        out |= _bit(src, strip * strip_words + kt * 32 * k2 + (tt % 8) * TW, n * L + ob) << (31 - b)
    return out


def to_strips_word(src: np.ndarray, i: int, k2: int, KT: int) -> int:
    """lanes.cu to_strips_kernel's statements for destination word i."""

    n = TW = 4 * k2
    strip_words = KT * 32 * k2
    strip, r = divmod(i, strip_words)
    kt, tw = divmod(r, 32 * k2)
    j, wd = divmod(tw, TW)
    gi = kt // G
    tt = (kt % G) * 8 + j
    s0 = strip * strip_words + gi * G * 32 * k2
    out = 0
    for b in range(32):
        sb = 32 * wd + b
        L = sb // n
        lb = tt * n + sb % n
        w = lb >> 5
        out |= ((int(src[s0 + ((w >> 2) * 32 + L) * 4 + (w & 3)]) >> (31 - (lb & 31))) & 1) << (31 - b)
    return out


# -- lanes.cu's k loop: a lane's group words and pair_frags' states -------------------------------------------------
def group_words(flat: np.ndarray, k2: int, K: int, nb: int, kt0: int, gi: int) -> np.ndarray:
    """[32, 2 K2]: R of every lane after load_group(base + gi * gstride, R), base = T + nb stride_nb + kt0 stride_k
    + 4 lane, chunk c at + 128 c."""

    stride_k, stride_nb = 32 * k2, (K // 16) * 32 * k2
    GW = G * k2
    lane = np.arange(32)
    base = nb * stride_nb + kt0 * stride_k + 4 * lane + gi * G * stride_k
    idx = base[:, None] + (np.arange(GW // 4)[None, :, None] * 128 + np.arange(4)[None, None, :]).reshape(1, GW)
    return flat[idx]


def take_bits(R: np.ndarray, p: int, n: int) -> np.ndarray:
    GW = R.shape[-1]
    wi, bo = p >> 5, p & 31
    assert bo + n <= 64
    x = (R[..., wi].astype(U64) << U64(32)) | (R[..., wi + 1].astype(U64) if wi + 1 < GW else U64(0))
    return x if n == 64 else (x << U64(bo)) >> U64(64 - n)


def pair_states(R: np.ndarray, k2: int, g: int, j: int) -> np.ndarray:
    """pair_frags' states [2, 32 lanes, 8] of tiles j, j + 1 of step g."""

    n = 4 * k2
    o0, o1 = take_bits(R, (g * 8 + j) * n, n), take_bits(R, (g * 8 + j + 1) * n, n)
    tails = (o0 & U64(0xFFFF)) | ((o1 & U64(0xFFFF)) << U64(16))
    pre = np.roll(tails, 1)                                   # __shfl_sync(tails, (lane + 31) & 31)
    out = np.empty((2, 32, 8), dtype=np.uint32)
    for h in range(2):
        o = o1 if h else o0
        p = (pre >> U64(16)) if h else (pre & U64(0xFFFF))
        for v in range(8):
            sh = (7 - v) * k2 // 2
            w = o >> U64(sh)
            if n - sh < 16:
                w = w | (p << U64(n - sh))
            out[h, :, v] = (w & U64(0xFFFF)).astype(np.uint32)
    return out


# -- decode.cuh's lane_states, transcribed ------------------------------------------------------------------------
def stream_end(k2: int, p: int) -> int:
    return ((p + 1) * k2 - ((p + 1) & 1)) // 2 if k2 & 1 else (p + 1) * (k2 // 2)


def lane_words_n(k2: int) -> int:
    most = 0
    for lane in range(32):
        first = (stream_end(k2, 8 * lane) - 16 + 128 * k2) % 32
        need = first + stream_end(k2, 8 * lane + 7) - stream_end(k2, 8 * lane) + 16
        most = max(most, need)
    return (most + 31) // 32


def lane_start(k2: int, lane: int) -> tuple[int, int]:
    first = 4 * lane * k2 + k2 // 2 - 16 + 128 * k2
    return (first >> 5) % (4 * k2), first & 31


def funnel_l(lo: int, hi: int, s: int) -> int:
    """__funnelshift_l(lo, hi, s): the high 32 bits of (hi:lo) << (s & 31)."""

    s &= 31
    return ((((hi << 32) | lo) << s) >> 32) & 0xFFFFFFFF


def window16(hi: int, mid: int, lo: int, d: int) -> int:
    if d <= 16:
        return (hi >> (16 - d)) & 0xFFFF
    if d < 32:
        return funnel_l(mid, hi, d) >> 16
    if d <= 48:
        return (mid >> (48 - d)) & 0xFFFF
    return funnel_l(lo, mid, d - 32) >> 16


def lane_states(tile: np.ndarray, k2: int, lane: int) -> list[int]:
    """decode.cuh load_lane_words + lane_states for one lane of one tile (tile: its 4 K2 words)."""

    tw, lw = 4 * k2, lane_words_n(k2)
    word, offset = lane_start(k2, lane)
    w = [int(tile[(word + i) % tw]) for i in range(lw)]
    c = w[2] if lw > 2 else 0
    hi = funnel_l(w[1], w[0], offset)
    mid = funnel_l(c, w[1], offset)
    lo = (c << offset) & 0xFFFFFFFF
    return [window16(hi, mid, lo, stream_end(k2, j) - stream_end(k2, 0)) for j in range(8)]


def _strips(k2: int, kt: int = 8, nt: int = 16, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + k2)
    trellis = rng.integers(-2**15, 2**15, size=(kt, nt, 8 * k2)).astype(np.int16)
    return trellis, strips_of(trellis)


@pytest.mark.parametrize("k2", K2S)
@pytest.mark.parametrize("kt", (8, 72, 80))                       # K = 128, 1152, 1280
def test_layout_round_trips_and_permutes_bits(k2: int, kt: int):
    _, s = _strips(k2, kt=kt, nt=16)
    lanes = to_lanes(s, k2)
    assert lanes.shape == s.shape
    assert np.array_equal(to_strips(lanes, k2), s)
    for strip in range(s.shape[0]):                                # a permutation inside each strip: same bit count
        assert bits_of(lanes[strip]).sum() == bits_of(s[strip]).sum()


@pytest.mark.parametrize("k2", K2S)
def test_relayout_kernel_statements_equal_the_layout(k2: int):
    _, s = _strips(k2, kt=8, nt=16)
    kt = s.shape[1]
    lanes = to_lanes(s, k2)
    src, ref = s.reshape(-1), lanes.reshape(-1)
    rng = np.random.default_rng(k2)
    for i in sorted(set(rng.integers(0, src.size, 300).tolist()) | {0, src.size - 1}):
        assert to_lanes_word(src, i, k2, kt) == int(ref[i])
        assert to_strips_word(ref, i, k2, kt) == int(src[i])


@pytest.mark.parametrize("k2", K2S)
def test_pair_frags_states_equal_decode_cuh_lane_states(k2: int):
    """Every lane, every tile pair, both steps of every group: pair_frags' states == lane_states' == format.states."""

    trellis, s = _strips(k2, kt=8, nt=16)
    nb, KT = s.shape[:2]
    K = 16 * KT
    flat = to_lanes(s, k2).reshape(-1)
    ref = fmt.states(trellis, k2 / 2)                              # [KT, NT, 256]
    for strip in range(nb):
        for gi in range(KT // G):
            R = group_words(flat, k2, K, strip, 0, gi)
            for g in range(G):
                kt = gi * G + g
                for j in range(0, 8, 2):
                    st = pair_states(R, k2, g, j)
                    for h in range(2):
                        tile = s[strip, kt, j + h]
                        want = np.array([lane_states(tile, k2, lane) for lane in range(32)], dtype=np.uint32)
                        assert np.array_equal(st[h], want), (strip, kt, j + h)
                        assert np.array_equal(st[h].reshape(256), ref[kt, strip * 8 + j + h])


@pytest.mark.parametrize("k2", K2S)
@pytest.mark.parametrize("split", [(1, 4), (2, 4), (1, 8), (4, 4)])
def test_a_warp_reads_its_range_once_in_coalesced_chunks(k2: int, split: tuple[int, int]):
    K, sk, wk = 1280, *split
    kt_all = K // 16
    per_warp = kt_all // sk // wk
    if kt_all % (sk * wk) or per_warp % 2:
        pytest.skip("not a lanes plan")
    stride_k, stride_nb = 32 * k2, kt_all * 32 * k2
    seen = np.zeros(2 * stride_nb, dtype=np.int32)                 # two strips' words
    for nb in range(2):
        for sp in range(sk):
            for warp in range(wk):
                kt0 = sp * per_warp * wk + warp * per_warp
                for gi in range(per_warp // G):
                    for c in range(G * k2 // 4):
                        first = nb * stride_nb + kt0 * stride_k + gi * G * stride_k + 128 * c
                        assert first % 4 == 0                      # 16-byte aligned lanes, 512 contiguous bytes
                        seen[first:first + 128] += 1
    assert (seen == 1).all()


def test_kernel_widths_fit_take_bits():
    """take_bits needs bo + 4 K2 <= 64 for every tile's own bits: true for K2 4..12, false for 14."""

    def fits(k2: int) -> bool:
        n = 4 * k2
        return all(((g * 8 + j) * n) % 32 + n <= 64 for g in range(G) for j in range(8))

    assert all(fits(k2) for k2 in K2S)
    assert not fits(14)
