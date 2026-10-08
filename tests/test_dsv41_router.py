"""The router's slices summed inside the routing kernel (``router_logits(parts=True)`` + ``route``) == the summed
logits routed, and bf16 rows converted as loaded == ``x.half()`` first: the same picks and weights, bit for bit (GPU)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.families.deepseek_v41.cuda import kernels as K  # noqa: E402


@pytest.mark.parametrize("R", [1, 2, 6, 16, 33, 2048])
@pytest.mark.parametrize("E,D", [(384, 4096), (64, 4096)])
def test_parts_routed_equal_summed(R, E, D):
    g = torch.Generator(device="cuda").manual_seed(R + E)
    x = (torch.randn((R, D), generator=g, device="cuda")).to(torch.bfloat16)
    w = (torch.randn((E, D), generator=g, device="cuda") * 0.02).half()
    bias = torch.randn((E,), generator=g, device="cuda") * 0.1
    summed = K.router_logits(x, w)
    assert torch.equal(summed, K.router_logits(x.half(), w))
    pa, wa = K.route(summed, bias, 6, 2.5)
    pb, wb = K.route(K.router_logits(x, w, parts=True), bias, 6, 2.5)
    assert torch.equal(pa, pb) and torch.equal(wa.view(torch.int32), wb.view(torch.int32))


@pytest.mark.parametrize("R", [1, 6, 16])
def test_one_warp_route_equals_four(R, monkeypatch):
    """TF_DSV41_ROUTE_WARPS=1: the same picks and weights as 4 warps, also with tied score + bias (lowest id wins)."""

    g = torch.Generator(device="cuda").manual_seed(40 + R)
    E = 384
    logits = torch.randn((R, E), generator=g, device="cuda")
    logits[:, 100:110] = logits[:, 100:101]                         # ties
    bias = torch.randn((E,), generator=g, device="cuda") * 0.1
    bias[100:110] = 0.0
    out = {}
    for nw in (4, 1):
        monkeypatch.setattr(K, "ROUTE_WARPS", nw)
        out[nw] = K.route(logits, bias, 6, 2.5)
    assert torch.equal(out[4][0], out[1][0]) and torch.equal(out[4][1].view(torch.int32), out[1][1].view(torch.int32))


@pytest.mark.parametrize("R", [1, 2, 6, 16])
@pytest.mark.parametrize("E,D", [(384, 5120), (32, 5120)])
def test_router_be16_equals_be32(R, E, D, monkeypatch):
    """TF_ROUTER_BE=16: the same logits (summed and as slices) as 32 experts a program, bit for bit."""

    g = torch.Generator(device="cuda").manual_seed(70 + R + E)
    x = torch.randn((R, D), generator=g, device="cuda").to(torch.bfloat16)
    w = (torch.randn((E, D), generator=g, device="cuda") * 0.02).half()
    out = {}
    for be in (32, 16):
        monkeypatch.setattr(K, "ROUTER_BE", be)
        out[be] = (K.router_logits(x, w), K.router_logits(x, w, parts=True))
    for a, b in zip(out[32], out[16]):
        assert torch.equal(a.view(torch.int32), b.view(torch.int32))
