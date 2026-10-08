"""TF_DSV41_PDL: the small decode kernels as programmatic dependent launches give the same bits as plain launches,
inside a captured graph replayed many times, right behind a slow producer of their input (a missing griddepcontrol
.wait reads the poisoned buffer and shows); and the launch attribute survives torch's stream capture (a kernel
launched with it but without the wait does race)."""

import json
from pathlib import Path

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (9, 0):
    pytest.skip("CUDA sm_90+ only (griddepcontrol)", allow_module_level=True)

import triton
import triton.language as tl
from triton.language.extra.cuda import gdc_launch_dependents

from tensorfold.families.deepseek_v41 import reference as R
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import hc
from tensorfold.families.deepseek_v41.cuda import kernels as K

CFG = Config.from_dict(json.loads((Path(__file__).parents[1] / "fixtures" / "deepseek_v41" / "config.json").read_text()))
REPS = 1000
SPIN = 20000                                 # the producer's busy loop (tens of us: the consumer launches early)


@triton.jit
def _late_copy(SRC, DST, SINK, n, spin, BLOCK: tl.constexpr):
    """DST <- SRC bytes after a dependent integer loop (the stores wait on it: never hoisted). It lets the next
    kernel launch at once, so a dependent launch runs beside the loop and only its griddepcontrol.wait orders it."""

    gdc_launch_dependents()
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.program_id(0) + 1
    for _ in range(spin):
        acc = acc * 1103515245 + 12345
    v = tl.load(SRC + i, mask=i < n)
    tl.store(DST + i, v, mask=(i < n) & (acc != 0x12345))
    tl.store(SINK + tl.program_id(0), acc)


def late_copy(src: torch.Tensor, dst: torch.Tensor, sink: torch.Tensor) -> None:
    s, d = src.view(-1).view(torch.uint8), dst.view(-1).view(torch.uint8)
    _late_copy[(triton.cdiv(s.numel(), 1024),)](s, d, sink, s.numel(), SPIN, BLOCK=1024, num_warps=4)


@triton.jit
def _no_wait(X, Y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, tl.load(X + i, mask=i < n), mask=i < n)


def _outs(o):
    return [t for t in (o if isinstance(o, (tuple, list)) else (o,)) if isinstance(t, torch.Tensor)]


def _replays(op, src, inp, pdl, monkeypatch, reps=REPS):
    """``op(inp)`` captured behind late_copy(src -> inp) with TF_DSV41_PDL ``pdl``; each replay starts from a poisoned
    ``inp`` and must give the eager plain-launch bits. Returns the eager reference outputs."""

    sink = torch.empty(1 << 16, dtype=torch.int32, device="cuda")
    monkeypatch.setattr(K, "PDL", False)
    inp.copy_(src)
    want = [t.clone() for t in _outs(op(inp))]
    monkeypatch.setattr(K, "PDL", pdl)
    inp.copy_(src)
    for got, w in zip(_outs(op(inp)), want):                 # eager (and compiles the PDL kernels before capture)
        assert torch.equal(got, w)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        late_copy(src, inp, sink)
        outs = _outs(op(inp))
    poison = inp.view(-1).view(torch.uint8)
    for i in range(reps):
        poison.fill_(0xFF if i % 2 else 0x7F)               # NaNs / huge values: an early read cannot match
        for o in outs:
            o.view(-1).view(torch.uint8).fill_(0x55)
        g.replay()
        if i % 50 == 0 or i == reps - 1:
            torch.cuda.synchronize()
            for got, w in zip(outs, want):
                assert torch.equal(got, w), f"replay {i}"
    return want


def _gen(seed):
    return torch.Generator(device="cuda").manual_seed(seed)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [1, 6])
def test_rmsnorm_rope_rope_q(monkeypatch, pdl, rows):
    g = _gen(rows)
    eps = CFG.rms_norm_eps
    cos, sin = K.rope_tables(R.inv_freq(CFG, 0, "cuda"), 4096)
    w = (torch.rand((1280,), generator=g, device="cuda") + 0.5).to(torch.bfloat16)
    pos = torch.randint(0, 4096, (rows,), generator=g, device="cuda")
    src = (torch.randn((rows, 1280), generator=g, device="cuda") * 3).to(torch.bfloat16)
    _replays(lambda x: K.rmsnorm(x, w, eps), src, torch.empty_like(src), pdl, monkeypatch)
    src = torch.randn((rows, 8, 512), generator=g, device="cuda").to(torch.bfloat16)
    _replays(lambda x: (K.rope(x, pos, cos, sin), K.rope(x, pos, cos, sin, inverse=True, out_dtype=torch.float32)),
             src, torch.empty_like(src), pdl, monkeypatch)
    src = torch.randn((rows, 512), generator=g, device="cuda").to(torch.bfloat16)
    _replays(lambda x: K.rope_q(x, pos, cos, sin, "fp8"), src, torch.empty_like(src), pdl, monkeypatch)
    src = torch.randn((rows, 64, 128), generator=g, device="cuda").to(torch.bfloat16)
    _replays(lambda x: K.rope_q(x, pos, cos, sin, "mxfp4"), src, torch.empty_like(src), pdl, monkeypatch)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("group,scale,dim", [(16, "e4m3", 512), (32, "ue8m0", 128)])
def test_fp4_rows_store(monkeypatch, pdl, group, scale, dim):
    g = _gen(group)
    cos, sin = K.rope_tables(R.inv_freq(CFG, 4, "cuda"), 4096)
    rows = K.Fp4Rows(64, dim, group=group, scale=scale)
    slot = torch.tensor([3, 17, 40, 41, 9], device="cuda")
    pos = torch.tensor([7, 100, 2000, 2001, 4095], device="cuda")
    src = torch.randn((5, dim), generator=g, device="cuda").to(torch.bfloat16)

    def op(x):
        rows.q.zero_()
        rows.s.zero_()
        rows.store(slot, x, pos, cos, sin)
        return rows.q, rows.s

    _replays(op, src, torch.empty_like(src), pdl, monkeypatch, reps=300)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("rows", [1, 6])
def test_router_and_route(monkeypatch, pdl, rows):
    g = _gen(10 + rows)
    w = (torch.randn((384, 5120), generator=g, device="cuda") * 0.02).half()
    wi = (torch.randn((64, 5120), generator=g, device="cuda") * 0.02).half()
    bias = torch.randn((384,), generator=g, device="cuda") * 0.1
    src = torch.randn((rows, 5120), generator=g, device="cuda").to(torch.bfloat16)
    _replays(lambda x: (*K.route(K.router_logits(x, w, parts=True), bias, 6, 1.5), K.router_logits(x, wi)),
             src, torch.empty_like(src), pdl, monkeypatch)


def _hc_inputs(rows, seed):
    g = _gen(seed)
    D = CFG.hidden_size
    X = (torch.randn((rows, 4, D), generator=g, device="cuda") * 3).to(torch.bfloat16)
    fn = torch.randn((24, 4 * D), generator=g, device="cuda") * 0.01
    base = torch.randn((24,), generator=g, device="cuda")
    scale = torch.rand((3,), generator=g, device="cuda") + 0.5
    norm_w = (torch.rand((D,), generator=g, device="cuda") + 0.5).to(torch.bfloat16)
    pre_in = torch.rand((rows, 4), generator=g, device="cuda")
    return X, fn, base, scale, norm_w, pre_in


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("mode", ["side", "side_sinkhorn", "plain", "split"])
@pytest.mark.parametrize("rows", [1, 6])
def test_hc_pre(monkeypatch, pdl, mode, rows):
    X, fn, base, scale, norm_w, pre_in = _hc_inputs(rows, rows)
    buf = hc.HCBuffers(32, CFG.hidden_size)
    monkeypatch.setattr(hc, "HC_SPLIT", mode == "split")
    monkeypatch.setattr(hc, "HC_SIDE", mode.startswith("side"))
    monkeypatch.setattr(hc, "HC_SIDE_PART", mode != "side_sinkhorn")
    side = torch.cuda.Stream() if mode.startswith("side") else None

    def op(x):
        out = hc.pre(x, fn, base, scale, pre_in, norm_w, buf, CFG.rms_norm_eps, CFG.hc_eps, CFG.hc_sinkhorn_iters,
                     side=side)
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        return out

    _replays(op, X, torch.empty_like(X), pdl, monkeypatch, reps=500)


@pytest.mark.parametrize("pdl", [False, True])
@pytest.mark.parametrize("parts", [0, 2])
def test_hc_post(monkeypatch, pdl, parts):
    rows = 6
    X, *_ = _hc_inputs(rows, 3)
    g = _gen(4)
    D = CFG.hidden_size
    post_w = torch.rand((rows, 4), generator=g, device="cuda") * 2
    comb = torch.rand((rows, 4, 4), generator=g, device="cuda")
    if parts:
        src = torch.randn((parts, rows, D), generator=g, device="cuda")
    else:
        src = torch.randn((rows, D), generator=g, device="cuda").to(torch.bfloat16)
    _replays(lambda b: hc.post(b, X, post_w, comb), src, torch.empty_like(src), pdl, monkeypatch)


def test_launch_attribute_survives_capture():
    """A kernel launched with launch_pdl but no griddepcontrol.wait reads the producer's buffer before the producer
    is done, in a graph too: the programmatic edge is real (else TF_DSV41_PDL would only cost a specialization)."""

    n = 1 << 16
    src = torch.arange(n, dtype=torch.int32, device="cuda")
    inp, out = torch.empty_like(src), torch.empty_like(src)
    sink = torch.empty(1 << 16, dtype=torch.int32, device="cuda")
    for pdl in (False, True):
        g = torch.cuda.CUDAGraph()
        _no_wait[(n // 1024,)](inp, out, n, BLOCK=1024, launch_pdl=pdl)       # compile outside capture
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            late_copy(src, inp, sink)
            _no_wait[(n // 1024,)](inp, out, n, BLOCK=1024, launch_pdl=pdl)
        stale = 0
        for _ in range(20):
            inp.fill_(-1)
            g.replay()
            torch.cuda.synchronize()
            stale += int(not torch.equal(out, src))
        assert (stale > 0) == pdl, (pdl, stale)
