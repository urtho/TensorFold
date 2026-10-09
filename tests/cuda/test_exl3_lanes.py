"""TF_EXL3_LANES on a GPU (``cuda/exl3/lanes.py``, ``lanes.cu``): the lanes layout's linear and unpack against
linear.cu's on the same trellis, bit for bit (torch.equal) — every codebook and even width the kernel builds, 1 to 32
rows, every lanes split plan, split-K counters reused, bias, output dtypes, grouped linears with their slices, the
module paths (linear_rotated, grouped_rotated with the rotated-output epilogue, the prompt GEMM's unpack), and the
in-place conversion rules (data pointer kept, once per storage, exclusions, idempotent)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import lanes, linear

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

COMBOS = [("3inst", b) for b in (2, 3, 4, 5, 6)] + [("mcg", b) for b in (2, 3, 4, 6)] \
    + [("mul1", b) for b in (2, 3, 4, 5, 6)]
ROWS = (1, 2, 6, 16, 17, 32)
SPLITS = [(1, 4), (2, 4), (4, 4), (8, 4), (1, 8), (2, 8), (4, 8)]      # K = 1024: per_warp 16 .. 2, all even


def _tensors(bits: float, kt: int, nt: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(kt, nt, fmt.tile_words(bits))).astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(16 * kt) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(16 * nt) * 0.05).astype(np.float16))
    bias = torch.from_numpy((rng.standard_normal(16 * nt) * 0.05).astype(np.float16))
    return trellis, suh, svh, bias


def _pair(codebook: str, bits: float, kt: int = 64, nt: int = 16, split=None, bias: bool = False, seed: int = 0):
    """(strips layer, lanes layer) on the same trellis: two copies of the words, the second converted."""

    trellis, suh, svh, b = _tensors(bits, kt, nt, seed)
    a = linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook, bias=b if bias else None)
    c = linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook, bias=b if bias else None)
    if split is not None:
        a.split = c.split = split
    st = lanes.convert(c, log=None)
    assert st["matrices"] == 1 and c.layout == "lanes" and isinstance(c, lanes.LanesExl3Linear)
    return a, c


@pytest.mark.parametrize("codebook,bits", COMBOS)
@pytest.mark.parametrize("split", SPLITS)
def test_lanes_linear_equals_strips(codebook: str, bits: float, split: tuple[int, int]):
    a, c = _pair(codebook, bits, split=split)
    assert not torch.equal(a.words, c.words)                      # really repacked
    x = torch.randn((max(ROWS), a.k), device="cuda").half()
    for m in ROWS:
        for _ in range(2):                                        # split-K counters left zero and reused
            assert torch.equal(c(x[:m]), a(x[:m])), f"{m} rows"


@pytest.mark.parametrize("codebook,bits", COMBOS)
def test_lanes_unpack_equals_strips(codebook: str, bits: float):
    a, c = _pair(codebook, bits, kt=16, nt=16)
    assert torch.equal(c.unpack().view(torch.int16), a.unpack().view(torch.int16))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("bias", [False, True])
def test_dtypes_and_bias(dtype, bias: bool):
    a, c = _pair("mul1", 5, kt=80, nt=32, bias=bias)              # K = 1280 (wq_b's), plan's own split
    x = torch.randn((17, a.k), device="cuda").to(dtype)
    for od in (None, torch.float32, torch.bfloat16):
        assert torch.equal(c(x, out_dtype=od), a(x, out_dtype=od))


def test_model_like_shapes():
    """A few of DSV4.1's per-rank shapes at their own plans (5-bit attention, 4-bit and 6-bit), rows 1..32."""

    for k, n, bits in ((5120, 1024, 5), (1280, 2048, 5), (4096, 1024, 5), (5120, 1152, 4), (5120, 512, 6)):
        a, c = _pair("mul1", bits, kt=k // 16, nt=n // 16, seed=k + n)
        if not lanes.lanes_ok(a.k2, a.k, a.split):
            continue
        x = torch.randn((32, k), device="cuda").half()
        for m in ROWS:
            assert torch.equal(c(x[:m]), a(x[:m])), (k, n, bits, m)


def test_rejected_layers_stay_on_strips():
    assert lanes.why_not(5, 1024, (1, 4)) is not None              # odd K2 (2.5 bits)
    assert lanes.why_not(16, 1024, (1, 4)) is not None             # 8 bits
    assert lanes.why_not(14, 1024, (1, 4)) is not None
    assert lanes.why_not(2, 1024, (1, 4)) is not None              # 1 bit: windows reach two lanes back
    assert lanes.why_not(10, 1152, (2, 4)) is not None             # per_warp 9 (shared w2)
    assert lanes.why_not(10, 1024, (1, 2)) is not None             # 2 warps: not built
    assert lanes.why_not(10, 1024, (2, 4)) is None
    trellis, suh, svh, _ = _tensors(2.5, 64, 16)
    odd = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    w9 = linear.Exl3Linear.from_tensors(*_tensors(5, 72, 16)[:3], "mul1")
    w9.split = (2, 4)
    before = [odd.words.clone(), w9.words.clone()]
    st = lanes.convert([odd, w9], log=None)
    assert st["matrices"] == 0 and st["kept"] == 2
    assert odd.layout == w9.layout == "strips" and type(odd) is linear.Exl3Linear
    assert torch.equal(odd.words, before[0]) and torch.equal(w9.words, before[1])


def test_conversion_keeps_storage_and_is_idempotent():
    trellis, suh, svh, _ = _tensors(5, 64, 16)
    a = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    c = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    ex = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    ptr = c.words.data_ptr()
    lanes.convert({"c": c, "again": [c], "ex": ex}, exclude=(ex,), log=None)
    assert c.words.data_ptr() == ptr and c.layout == "lanes"
    assert ex.layout == "strips" and torch.equal(ex.words, a.words)
    once = c.words.clone()
    assert lanes.convert(c, log=None)["matrices"] == 0            # a second call: no second permutation
    assert torch.equal(c.words, once)
    back = torch.empty_like(c.words)                               # and the layout inverts to the strips words
    lanes._ext().relayout(c.words, back, c.k2, c.k, False)
    assert torch.equal(back, a.words)


def _group(bits: float = 5, groups: int = 4, kt: int = 32, nt: int = 16):
    layers = []
    for gi in range(groups):
        trellis, suh, svh, _ = _tensors(bits, kt, nt, seed=100 + gi)
        layers.append(linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1"))
    return layers, linear.GroupedLinear(layers)


def test_grouped_linear_and_its_slices():
    """wo_a: the group's shared words repacked once; the group and every slice read lanes, slices included in
    the prompt path; the same bits as strips."""

    la, ga = _group()
    lc, gc = _group()
    ptrs = [m.words.data_ptr() for m in lc]
    st = lanes.convert({"slices": lc, "group": gc}, log=None)
    assert st["matrices"] == 1                                     # one storage
    assert gc.layout == "lanes" and isinstance(gc, lanes.LanesGroupedLinear)
    assert all(m.layout == "lanes" for m in lc) and [m.words.data_ptr() for m in lc] == ptrs
    x = torch.randn((32, ga.groups * ga.k), device="cuda").half()
    for m in ROWS:
        assert torch.equal(gc(x[:m]), ga(x[:m]))
        assert torch.equal(linear.grouped_rotated(gc, None, torch.float32, x=x[:m]),
                           linear.grouped_rotated(ga, None, torch.float32, x=x[:m]))
    for sa, sc in zip(la, lc):
        xs = torch.randn((6, sa.k), device="cuda").half()
        assert torch.equal(sc(xs), sa(xs))
        assert torch.equal(sc.unpack().view(torch.int16), sa.unpack().view(torch.int16))


def test_grouped_rotated_output_epilogue():
    mode = linear.rot_mode()
    if mode is None:
        pytest.skip("no rot128 form equals this build's rot_in")
    _, ga = _group()
    _, gc = _group()
    lanes.convert(gc, log=None)
    suh = torch.from_numpy((np.random.default_rng(5).standard_normal(ga.groups * ga.n) * 0.05).astype(np.float16)).cuda()
    x = torch.randn((6, ga.groups * ga.k), device="cuda").half()
    xo_a = torch.empty((6, ga.groups * ga.n), dtype=torch.float16, device="cuda")
    xo_c = torch.empty_like(xo_a)
    ya = linear.grouped_rotated(ga, None, torch.float32, x=x, rot_out=(suh, xo_a, mode))
    yc = linear.grouped_rotated(gc, None, torch.float32, x=x, rot_out=(suh, xo_c, mode))
    assert torch.equal(ya, yc) and torch.equal(xo_a, xo_c)


def test_linear_rotated_dispatches_on_layout():
    a, c = _pair("mul1", 5, kt=64, nt=32)
    x = torch.randn((6, a.k), device="cuda").half()
    xh = torch.empty((6, a.k), dtype=torch.float16, device="cuda")
    linear._ext().rot_in(x, a.suh, xh)
    assert torch.equal(linear.linear_rotated(c, xh, torch.float32), linear.linear_rotated(a, xh, torch.float32))


def test_dsv41_linear_subclass_and_the_prompt_gemm():
    """dsv41's weights.Linear converted: its decode rows and its prompt GEMM (prefill.matmul's unpack) on lanes."""

    pytest.importorskip("triton")
    from tensorfold.families.deepseek_v41.cuda import weights as W

    trellis, suh, svh, _ = _tensors(5, 64, 32, seed=9)
    a = W.Linear.from_tensors(trellis, suh, svh, "mul1")
    c = W.Linear.from_tensors(trellis, suh, svh, "mul1")
    lanes.convert(c, log=None)
    assert isinstance(c, W.Linear) and isinstance(c, lanes.LanesExl3Linear) and c.layout == "lanes"
    x = torch.randn((300, a.k), device="cuda").half()
    for m in (1, 6, 17):
        assert torch.equal(c(x[:m]), a(x[:m]))
    assert torch.equal(c(x), a(x))                                 # > 128 rows: the prompt GEMM
    W.Linear.prompt_mode = True
    try:
        assert torch.equal(c(x[:40]), a(x[:40]))
    finally:
        W.Linear.prompt_mode = False
