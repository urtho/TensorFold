"""TF_DSV41_ROT_FUSE (rot128.cuh): the rotated rows the attention merge writes for wo_a and wo_a's epilogue writes for
wo_b equal rot_in's of the stored outputs bit for bit, and the outputs themselves are unchanged: rot_exact against
rot_in, both merge kernels (merge_kernel: a CTA per stage; merge_flat: per group) in the cvt and lut builds over 1..16
rows and window-only layers, the grouped linear at SK == 1 and SK > 1 in every output dtype."""

from __future__ import annotations

import numpy as np
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import linear

H, D, W, E = 32, 512, 128, 3000


def _rot_in(x: torch.Tensor, suh: torch.Tensor) -> torch.Tensor:
    xh = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    linear._ext().rot_in(x, suh, xh)
    return xh


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _suh(k: int, seed: int) -> torch.Tensor:
    """suh of mixed magnitudes, the first quarter signs only (EXL3's plain sign flips)."""

    g = torch.Generator(device="cuda").manual_seed(seed)
    s = (torch.randn((k,), generator=g, device="cuda") *
         torch.exp2(torch.randint(-4, 5, (k,), generator=g, device="cuda").float())).half()
    s[: k // 4] = torch.where(s[: k // 4] < 0, -1.0, 1.0).half()
    return s


def test_rot_mode_found_and_exact():
    """A form equals rot_in on this build; rot_exact in it == rot_in on fresh rows (1..16 rows, every input dtype)."""

    mode = linear.rot_mode()
    assert mode is not None
    g = torch.Generator(device="cuda").manual_seed(7)
    k = H * D
    suh = _suh(k, 8)
    for R in range(1, 17):
        base = torch.randn((R, k), generator=g, device="cuda") * torch.exp2(
            torch.randint(-8, 9, (R, k), generator=g, device="cuda").float())
        for x in (base.to(torch.bfloat16), base.clamp(-6e4, 6e4).half(), base):
            got = torch.empty((R, k), dtype=torch.float16, device="cuda")
            linear._ext().rot_exact(x, suh, got, mode)
            assert torch.equal(_bits(got), _bits(_rot_in(x, suh))), (R, x.dtype)


# -- the attention merge ------------------------------------------------------------------------------------------
def _mqa_mods():
    if torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("mqa_fp4: sm_12x only")
    from tensorfold.families.deepseek_v41.cuda import kernels as K
    from tensorfold.families.deepseek_v41.cuda import mqa_fp4

    return K, mqa_fp4


def _tables(K, n=1 << 16):
    freqs = 1.0 / (160000.0 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    return K.rope_tables(freqs, n)


def _cache(K, seed, n=E):
    cos, sin = _tables(K)
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((n, D), generator=g, device="cuda") * torch.exp2(
        torch.randint(-3, 2, (n, 1), generator=g, device="cuda").float())
    rows = K.Fp4Rows(n, D, device="cuda")
    rows.store(torch.arange(n, device="cuda"), x.to(torch.bfloat16), torch.arange(n, device="cuda") * 3, cos, sin)
    return rows


@pytest.mark.parametrize("lut", [False, True])
@pytest.mark.parametrize("comp", [True, False])
@pytest.mark.parametrize("per_group", [False, True])        # False: merge_kernel (per 1), True: merge_flat (per group)
def test_merge_rot_out_equals_rot_in(lut, comp, per_group):
    K, mqa_fp4 = _mqa_mods()
    ext = mqa_fp4.ext(lut)
    mode = linear.rot_mode()
    rows = _cache(K, 1)
    cos, sin = _tables(K)
    suh = _suh(H * D, 3)
    group = mqa_fp4.GROUP
    per = group if per_group else 1
    parts = -(-mqa_fp4.stages(512 if comp else 0) // per)
    po = torch.empty((16 * parts * H * D,), device="cuda")
    pm, pl = torch.empty((16 * parts * H,), device="cuda"), torch.empty((16 * parts * H,), device="cuda")
    for R in range(1, 17):
        g = torch.Generator(device="cuda").manual_seed(100 + R)
        q = (torch.randn((R, H, D), generator=g, device="cuda") * 0.6).to(torch.bfloat16)
        pos = 5000 + torch.randint(0, 1000, (R,), generator=g, device="cuda")
        sink = torch.randn((H,), generator=g, device="cuda")
        swa = (torch.randn((4096, D), generator=g, device="cuda") * 2).to(torch.bfloat16)
        idx = torch.stack([torch.randperm(E, generator=g, device="cuda")[:512] for _ in range(R)]).int() \
            if comp else None
        if idx is not None:
            idx[:, -9:] = -1
        args = (q, rows.q if comp else None, rows.s if comp else None, idx, swa, pos, None, sink, cos, sin)
        tail = (po, pm, pl, swa.shape[0], group, per, D ** -0.5, 0)
        want = torch.empty((R, H, D), dtype=torch.bfloat16, device="cuda")
        ext.attend_rows(*args, want, *tail)
        out = torch.empty_like(want)
        xo = torch.full((R, H * D), float("nan"), dtype=torch.float16, device="cuda")
        ext.attend_rows_rot(*args, out, *tail, suh, xo, mode)
        assert torch.equal(_bits(out), _bits(want)), R
        assert torch.equal(_bits(xo), _bits(_rot_in(out.view(R, -1), suh))), R


def test_mqa_rot_flag(monkeypatch):
    """kernels.mqa: (o, True) with xo filled on the CUDA path; (o, False) on the Triton one; no RoPE: no xo."""

    K, _ = _mqa_mods()
    mode = linear.rot_mode()
    rows = _cache(K, 2)
    cos, sin = _tables(K)
    suh = _suh(H * D, 4)
    buf = K.AttnBuffers(32, H, D, 512 + W)
    g = torch.Generator(device="cuda").manual_seed(5)
    R = 3
    q = (torch.randn((R, H, D), generator=g, device="cuda") * 0.6).to(torch.bfloat16)
    pos = 5000 + torch.randint(0, 1000, (R,), generator=g, device="cuda")
    sink = torch.randn((H,), generator=g, device="cuda")
    swa = (torch.randn((4096, D), generator=g, device="cuda") * 2).to(torch.bfloat16)
    idx = torch.stack([torch.randperm(E, generator=g, device="cuda")[:512] for _ in range(R)]).int()
    monkeypatch.setattr(K, "CUDA_MQA", True)
    want = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin)
    xo = torch.empty((R, H * D), dtype=torch.float16, device="cuda")
    o, filled = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin, rot=(suh, xo, mode))
    assert filled and torch.equal(o, want)
    assert torch.equal(_bits(xo), _bits(_rot_in(o.view(R, -1), suh)))
    o, filled = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, rot=(suh, xo, mode))
    assert not filled and o.dtype == torch.float32
    monkeypatch.setattr(K, "CUDA_MQA", False)
    o, filled = K.mqa(q, rows, idx, swa, pos, sink, W, buf, D ** -0.5, cos, sin, rot=(suh, xo, mode))
    assert not filled


# -- wo_a's epilogue ----------------------------------------------------------------------------------------------
def _layer(seed: int, k: int, n: int, bits: int = 4, codebook: str = "mul1") -> linear.Exl3Linear:
    rng = np.random.default_rng(seed)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits)))
                               .astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    return linear.Exl3Linear.from_tensors(trellis, suh, svh, codebook)


@pytest.mark.parametrize("split", [None, (1, 4), (2, 4)])   # None: plan(): wo_a's real split (SK > 1 at 4096 x 1024)
@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_grouped_rot_out_equals_rot_in(split, out_dtype):
    groups, k, n = 4, 4096, 1024                              # a rank's wo_a: 4 groups of 8 heads x 512 -> 1024
    layers = [_layer(10 + i, k, n) for i in range(groups)]
    if split is not None:
        for m in layers:
            m.split = split
    g = linear.GroupedLinear(layers)
    mode = linear.rot_mode()
    suh_b = _suh(groups * n, 9)
    wo_b = _layer(20, groups * n, 1024)
    gen = torch.Generator(device="cuda").manual_seed(1)
    for R in list(range(1, 17)) + [32]:
        x = (torch.randn((R, groups * k), generator=gen, device="cuda")).to(torch.bfloat16)
        want = g(x, out_dtype=out_dtype)
        xh = _rot_in(x, g.suh)
        y = linear.grouped_rotated(g, xh, out_dtype)
        assert torch.equal(y.view(torch.uint8), want.view(torch.uint8)), R
        xo = torch.full((R, groups * n), float("nan"), dtype=torch.float16, device="cuda")
        y2 = linear.grouped_rotated(g, None, out_dtype, x=x, rot_out=(suh_b, xo, mode))
        assert torch.equal(y2.view(torch.uint8), want.view(torch.uint8)), R
        assert torch.equal(_bits(xo), _bits(_rot_in(want, suh_b))), R
        if out_dtype == torch.bfloat16:                        # the next layer from those rows: wo_b(z) itself
            xb = torch.empty((R, wo_b.k), dtype=torch.float16, device="cuda")
            linear.grouped_rotated(g, xh, out_dtype, rot_out=(wo_b.suh, xb, mode))
            got = linear.linear_rotated(wo_b, xb, torch.float32)
            assert torch.equal(got, wo_b(want, out_dtype=torch.float32)), R
