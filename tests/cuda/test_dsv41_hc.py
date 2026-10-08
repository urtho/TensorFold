"""DeepSeek-V4.1 fused hyper-connection kernels against the reference's PyTorch math."""

import json
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import triton
import triton.language as tl

from tensorfold.families.deepseek_v41 import reference as R
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import hc

CFG = Config.from_dict(json.loads((Path(__file__).parents[1] / "fixtures" / "deepseek_v41" / "config.json").read_text()))


class _Ck:
    def __init__(self, t):
        self.t = t

    def get(self, name, dtype=None):
        x = self.t[name.rsplit(".", 1)[-1]]
        return x if dtype is None else x.to(dtype)


@pytest.mark.parametrize("rows", [1, 3, 17])
def test_pre_and_post_match_the_reference(rows):
    g = torch.Generator(device="cuda").manual_seed(rows)
    D = CFG.hidden_size
    X = (torch.randn((rows, 4, D), generator=g, device="cuda") * 3).to(torch.bfloat16)
    fn = torch.randn((24, 4 * D), generator=g, device="cuda") * 0.01
    base = torch.randn((24,), generator=g, device="cuda")
    scale = torch.rand((3,), generator=g, device="cuda") + 0.5
    norm_w = (torch.rand((D,), generator=g, device="cuda") + 0.5).to(torch.bfloat16)
    pre_in = torch.rand((rows, 4), generator=g, device="cuda")
    ref = R.HCMix(_Ck({"hc_fn": fn, "hc_base": base, "hc_scale": scale}), "layers.0.hc")
    post_r, comb_r, x_r, pre_r = ref(X, pre_in, norm_w, CFG)
    buf = hc.HCBuffers(32, D)
    post_k, comb_k, x_k, pre_k = hc.pre(X, fn, base, scale, pre_in, norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps,
                                        CFG.hc_sinkhorn_iters)
    # the reference matmul runs in TF32 in NVIDIA's container (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1)
    torch.testing.assert_close(pre_k, pre_r, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(post_k, post_r, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(comb_k, comb_r, rtol=2e-3, atol=2e-3)
    assert (x_k.float() - x_r.float()).abs().max() <= 0.02 * x_r.float().abs().max()
    b = torch.randn((rows, D), generator=g, device="cuda").to(torch.bfloat16)
    y_r = R.hc_post(b, X, post_r, comb_r)
    y_k = hc.post(b, X, post_r, comb_r)
    assert (y_k.float() - y_r.float()).abs().max() <= 0.01 * y_r.float().abs().max() + 1e-3


def test_a_row_alone_equals_the_row_in_a_window():
    g = torch.Generator(device="cuda").manual_seed(7)
    D = CFG.hidden_size
    X = torch.randn((9, 4, D), generator=g, device="cuda").to(torch.bfloat16)
    fn = torch.randn((24, 4 * D), generator=g, device="cuda") * 0.01
    base, scale = torch.randn((24,), generator=g, device="cuda"), torch.ones((3,), device="cuda")
    norm_w = torch.ones((D,), dtype=torch.bfloat16, device="cuda")
    pre_in = torch.rand((9, 4), generator=g, device="cuda")
    buf = hc.HCBuffers(16, D)
    all_rows = hc.pre(X, fn, base, scale, pre_in, norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps, CFG.hc_sinkhorn_iters)
    one = hc.pre(X[4:5].contiguous(), fn, base, scale, pre_in[4:5], norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps,
                 CFG.hc_sinkhorn_iters)
    for a, b in zip(all_rows, one):
        assert torch.equal(a[4:5], b)


def test_post_adds_rank_partials_in_order_then_rounds():
    g = torch.Generator(device="cuda").manual_seed(3)
    D = CFG.hidden_size
    X = torch.randn((2, 4, D), generator=g, device="cuda").to(torch.bfloat16)
    parts = torch.randn((2, 2, D), generator=g, device="cuda")
    post_w = torch.rand((2, 4), generator=g, device="cuda")
    comb = torch.softmax(torch.randn((2, 4, 4), generator=g, device="cuda"), -1)
    want = hc.post((parts[0] + parts[1]).to(torch.bfloat16), X, post_w, comb)
    assert torch.equal(hc.post(parts, X, post_w, comb), want)


def test_post_pre_fused_matches_post_then_pre():
    """Prompt chunks: post_pre gives the same streams and the same pre() outputs as post then pre."""

    torch.manual_seed(7)
    R, D = 100, 5120
    dev = "cuda"
    X = (torch.randn(R, 4, D, device=dev) * 0.5).to(torch.bfloat16)
    parts = torch.randn(2, R, D, device=dev).to(torch.bfloat16)
    post_w = torch.rand(R, 4, device=dev) * 2
    comb = torch.softmax(torch.randn(R, 4, 4, device=dev), -1)
    fn = torch.randn(24, 4 * D, device=dev) * 0.01
    base = torch.randn(24, device=dev) * 0.1
    scale = torch.rand(3, device=dev)
    pre_in = torch.rand(R, 4, device=dev)
    norm_w = torch.rand(D, device=dev).to(torch.bfloat16)
    buf = hc.HCBuffers(R, D, device=dev)
    Y_ref = hc.post(parts, X, post_w, comb)
    ref = hc.pre(Y_ref, fn, base, scale, pre_in, norm_w, buf, 1e-6, 1e-6, 20)
    ref = [t.clone() for t in ref]
    for split in (R, 48):
        b = parts if split == R else hc.SplitPartials(torch.cat([parts[:, r0:r0 + split].reshape(-1)
                                                                 for r0 in range(0, R, split)]), 2, split)
        Y, got = hc.post_pre(b, X, post_w, comb, fn, base, scale, pre_in, norm_w, buf, 1e-6, 1e-6, 20)
        assert torch.equal(Y, Y_ref)
        for name, a, e in zip(("post", "comb", "x_in", "pre"), got, ref):
            assert torch.allclose(a.float(), e.float(), rtol=1e-4, atol=1e-5 if name != "x_in" else 1e-2), name   # mix blocks differ


# -- TF_DSV41_HC_SPLIT / HC_SIDE / HC_SIDE_PART: bit-equal to the one-program finish ----------------------------------
ROWS = [1, 2, 3, 4, 5, 6, 8, 16, 17, 32]
MODES = ["split", "kernels", "side", "side_sinkhorn"]


@triton.jit
def _pre_partial_bbc5ee8(X, FN, PART, WIDE: tl.constexpr, NBLK: tl.constexpr, SUBK: tl.constexpr):
    """hc._pre_partial as it was before its EVICT parameter (bbc5ee8), verbatim."""

    r = tl.program_id(0)
    b = tl.program_id(1)
    KB: tl.constexpr = WIDE // NBLK
    m = tl.arange(0, 32)
    k = tl.arange(0, SUBK)
    acc = tl.zeros((32,), dtype=tl.float32)
    ss = tl.zeros((SUBK,), dtype=tl.float32)
    for t in range(KB // SUBK):
        base = b * KB + t * SUBK
        x = tl.load(X + r * WIDE + base + k).to(tl.float32)
        w = tl.load(FN + m[:, None] * WIDE + base + k[None, :], mask=m[:, None] < 24, other=0.0)
        acc += tl.sum(w * x[None, :], axis=1)
        ss += x * x
    tl.store(PART + (r * NBLK + b) * 32 + m, acc, mask=m < 24)
    tl.store(PART + (r * NBLK + b) * 32 + 24, tl.sum(ss, axis=0))


def _inputs(rows, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    D = CFG.hidden_size
    X = (torch.randn((rows, 4, D), generator=g, device="cuda") * 3).to(torch.bfloat16)
    fn = torch.randn((24, 4 * D), generator=g, device="cuda") * 0.01
    base = torch.randn((24,), generator=g, device="cuda")
    scale = torch.rand((3,), generator=g, device="cuda") + 0.5
    norm_w = (torch.rand((D,), generator=g, device="cuda") + 0.5).to(torch.bfloat16)
    pre_in = torch.rand((rows, 4), generator=g, device="cuda")
    return X, fn, base, scale, pre_in, norm_w


def _pre(mode, X, fn, base, scale, pre_in, norm_w, buf, monkeypatch, side=None):
    """hc.pre with ``mode``'s switches (None: the default path), ``side`` joined before returning; "kernels": the
    two halves launched one after the other."""

    monkeypatch.setattr(hc, "HC_SPLIT", mode == "split")
    monkeypatch.setattr(hc, "HC_SIDE", mode in ("side", "side_sinkhorn"))
    monkeypatch.setattr(hc, "HC_SIDE_PART", mode != "side_sinkhorn")
    if mode == "kernels":
        R, _, D = X.shape
        post = torch.empty((R, 4), dtype=torch.float32, device="cuda")
        comb = torch.empty((R, 4, 4), dtype=torch.float32, device="cuda")
        pre_out = torch.empty((R, 4), dtype=torch.float32, device="cuda")
        x_in = torch.empty((R, D), dtype=torch.bfloat16, device="cuda")
        hc._pre_partial[(R, hc.NB)](X, fn, buf.part, WIDE=4 * D, NBLK=hc.NB, SUBK=hc.SUB, num_warps=4)
        hc._pre_sinkhorn[(R,)](buf.part, base, scale, pre_out, post, comb, CFG.rms_norm_eps, CFG.hc_eps, D=D,
                               NBLK=hc.NB, ITERS=CFG.hc_sinkhorn_iters, num_warps=8)
        hc._pre_collapse[(R,)](X, pre_in, norm_w, x_in, CFG.rms_norm_eps, D=D, CH=hc.CHUNK, num_warps=8)
        return post, comb, x_in, pre_out
    side = side if mode in ("side", "side_sinkhorn") else None
    out = hc.pre(X, fn, base, scale, pre_in, norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps, CFG.hc_sinkhorn_iters,
                 side=side)
    if side is not None:
        torch.cuda.current_stream().wait_stream(side)
    return out


@pytest.mark.parametrize("evict", ["", "evict_first"])
@pytest.mark.parametrize("rows", [1, 6, 32])
def test_partial_evict_is_bit_equal_to_bbc5ee8(rows, evict):
    X, fn, *_ = _inputs(rows, 40 + rows)
    D = CFG.hidden_size
    a = torch.full((rows * hc.NB * 32,), float("nan"), device="cuda")
    b = a.clone()
    _pre_partial_bbc5ee8[(rows, hc.NB)](X, fn, a, WIDE=4 * D, NBLK=hc.NB, SUBK=hc.SUB, num_warps=4)
    hc._pre_partial[(rows, hc.NB)](X, fn, b, WIDE=4 * D, NBLK=hc.NB, SUBK=hc.SUB, EVICT=evict, num_warps=4)
    assert torch.equal(a.nan_to_num(7.0), b.nan_to_num(7.0))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", ROWS)
def test_split_and_side_bit_equal(rows, mode, monkeypatch):
    inp = _inputs(rows, 100 + rows)
    buf = hc.HCBuffers(32, CFG.hidden_size)
    want = _pre(None, *inp, buf, monkeypatch)
    got = _pre(mode, *inp, buf, monkeypatch, side=torch.cuda.Stream())
    for name, a, b in zip(("post", "comb", "x_in", "pre"), want, got):
        assert a.dtype == b.dtype and torch.equal(a, b), name


def _chain(mode, X, fn, base, scale, pre_in, norm_w, branch, buf, monkeypatch, side):
    """Two sublayers as serial.layers runs them: pre, post (after the join), pre with the carried pre."""

    post, comb, x, pre_a = _pre(mode, X, fn, base, scale, pre_in, norm_w, buf, monkeypatch, side)
    X2 = hc.post(branch, X, post, comb)
    return (x, X2, *_pre(mode, X2, fn, base * 0.5, scale, pre_a, norm_w, buf, monkeypatch, side))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("rows", [1, 6, 17])
def test_chained_sublayers_bit_equal(rows, mode, monkeypatch):
    inp = _inputs(rows, 200 + rows)
    branch = torch.randn((rows, CFG.hidden_size), device="cuda").to(torch.bfloat16)
    buf = hc.HCBuffers(32, CFG.hidden_size)
    want = _chain(None, *inp, branch, buf, monkeypatch, None)
    got = _chain(mode, *inp, branch, buf, monkeypatch, torch.cuda.Stream())
    for a, b in zip(want, got):
        assert torch.equal(a, b)


@pytest.mark.parametrize("mode", ["split", "side", "side_sinkhorn"])
@pytest.mark.parametrize("rows", [1, 6])
def test_side_stream_in_a_graph(rows, mode, monkeypatch):
    """The chain captured into a CUDA graph (the side stream forked and joined inside it), replayed on new inputs:
    bit-equal to the eager default path."""

    X, fn, base, scale, pre_in, norm_w = _inputs(rows, 300 + rows)
    branch = torch.randn((rows, CFG.hidden_size), device="cuda").to(torch.bfloat16)
    buf = hc.HCBuffers(32, CFG.hidden_size)
    side = torch.cuda.Stream()
    sX, spre = X.clone(), pre_in.clone()
    warm = torch.cuda.Stream()
    warm.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm):
        _chain(mode, sX, fn, base, scale, spre, norm_w, branch, buf, monkeypatch, side)
    torch.cuda.current_stream().wait_stream(warm)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        outs = _chain(mode, sX, fn, base, scale, spre, norm_w, branch, buf, monkeypatch, side)
    for seed in range(5):
        X2, _, _, _, p2, _ = _inputs(rows, 400 + seed)
        sX.copy_(X2)
        spre.copy_(p2)
        gr.replay()
        want = _chain(None, X2, fn, base, scale, p2, norm_w, branch, buf, monkeypatch, None)
        torch.cuda.synchronize()
        for a, b in zip(want, outs):
            assert torch.equal(a, b)


@pytest.mark.parametrize("mode", MODES)
def test_a_row_alone_equals_the_row_in_a_window_under_the_switches(mode, monkeypatch):
    X, fn, base, scale, pre_in, norm_w = _inputs(9, 7)
    buf = hc.HCBuffers(16, CFG.hidden_size)
    side = torch.cuda.Stream()
    all_rows = _pre(mode, X, fn, base, scale, pre_in, norm_w, buf, monkeypatch, side)
    one = _pre(mode, X[4:5].contiguous(), fn, base, scale, pre_in[4:5].contiguous(), norm_w, buf, monkeypatch, side)
    for a, b in zip(all_rows, one):
        assert torch.equal(a[4:5], b)
