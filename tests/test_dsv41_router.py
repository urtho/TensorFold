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
