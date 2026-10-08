"""x3ld (TF_EXPERT_LOADS) routed experts give bit-identical results to upstream's grouped kernel (GPU)."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("needs CUDA", allow_module_level=True)

from tensorfold.cuda.exl3 import experts as ex3  # noqa: E402
from tensorfold.cuda.exl3 import x3ld  # noqa: E402


def layer(E, D, I, k2s, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def trellis(k, n, k2):
        return torch.randint(-32768, 32767, (k // 16, n // 16, 8 * k2), generator=g, device="cuda",
                             dtype=torch.int32).to(torch.int16)

    def sign(n):
        return (torch.randint(0, 2, (n,), generator=g, device="cuda") * 2 - 1).half()

    def scale(n):
        return (torch.rand((n,), generator=g, device="cuda") * 0.02 + 0.005).half()

    gate = [(trellis(D, I, k2s[e]), sign(D), scale(I)) for e in range(E)]
    up = [(trellis(D, I, k2s[e]), sign(D), scale(I)) for e in range(E)]
    down = [(trellis(I, D, k2s[e]), sign(I), scale(D)) for e in range(E)]
    return ex3.prepare(gate, up, down, ex3.CB_MUL1)


@pytest.mark.parametrize("k2s,I", [("mixed", 1024), ("mixed", 2048), ("uniform8", 1024)])
@pytest.mark.parametrize("cfg", [(8, 1), (8, 2), (4, 2)])
def test_x3ld_equals_grouped(k2s, I, cfg, monkeypatch):
    E, D, slots = 24, 4096, 6
    widths = [4 + 2 * (e % 4) for e in range(E)] if k2s == "mixed" else [8] * E
    ex = layer(E, D, I, widths, I + len(k2s))
    g = torch.Generator(device="cuda").manual_seed(5)
    taken = []
    real = x3ld.grouped
    monkeypatch.setattr(x3ld, "grouped", lambda *a, **k: taken.append(real(*a, **k)) or taken[-1])
    for R in (1, 2, 5, 16, 32):
        x = torch.randn((R, D), generator=g, device="cuda").to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:slots] for _ in range(R)]).int()
        wts = torch.rand((R, slots), generator=g, device="cuda")
        outs = []
        for on in (False, True):
            monkeypatch.setitem(x3ld.CFG, "on", on)
            monkeypatch.setitem(x3ld.CFG, "gu", cfg)
            monkeypatch.setitem(x3ld.CFG, "dn", cfg)
            s = ex3.Scratch(ex, 32, slots)
            outs.append(ex3.routed(x, pick, wts, ex, s, None, R).clone())
        assert torch.equal(outs[0], outs[1]), (R, k2s, I, cfg)
        assert taken[-2:] == [True, True]                # both launches of the "on" run went through x3ld
        assert torch.isfinite(outs[0]).all()


def _picks(R, E, slots, g, last):
    """top-(slots - 1) distinct experts a row plus ``last`` (the folded shared expert's id, or E: a skipped slot)."""

    pick = torch.stack([torch.randperm(E - 1, generator=g, device="cuda")[:slots - 1] for _ in range(R)]).int()
    return torch.cat([pick, torch.full((R, 1), last, dtype=torch.int32, device="cuda")], dim=1).contiguous()


@pytest.mark.parametrize("D,I", [(4096, 1024), (5120, 1152)])
@pytest.mark.parametrize("x3", [False, True])
@pytest.mark.parametrize("skip", [False, True])
def test_l2_discard_is_bit_identical(D, I, x3, skip, monkeypatch):
    """TF_DSV41_L2_DISCARD=moe: routed() with wts gives the same bits over 50 back-to-back calls on one Scratch (s.z
    holds the gate/up Z, then the down Z the PDL-launched down x3ld writes over it). Under TF_SKIP_SHARED the skipped
    slot (pick E) reads its y row, so y stays stored there."""

    E, slots = 24, 7
    ex = layer(E, D, I, [8] * E, D + I)
    g = torch.Generator(device="cuda").manual_seed(9)
    monkeypatch.setitem(x3ld.CFG, "on", x3)
    monkeypatch.setitem(x3ld.CFG, "gu", (8, 1))
    monkeypatch.setitem(x3ld.CFG, "dn", (8, 1))
    monkeypatch.setattr(ex3, "SKIP_SHARED", skip)
    for R in (1, 2, 6, 16):
        xs = [torch.randn((R, D), generator=g, device="cuda").to(torch.bfloat16) for _ in range(50)]
        picks = [_picks(R, E, slots, g, E if skip else E - 1) for _ in range(50)]
        wts = [torch.rand((R, slots), generator=g, device="cuda") for _ in range(50)]
        yfill = torch.randn((16 * slots, D), generator=g, device="cuda")
        outs, ys = {}, {}
        for on in (False, True):
            monkeypatch.setattr(ex3, "DISCARD", on)
            s = ex3.Scratch(ex, 16, slots)
            s.y.copy_(yfill)                               # what a skipped slot reads
            outs[on] = [ex3.routed(x, p, w, ex, s, None, R).clone() for x, p, w in zip(xs, picks, wts)]
            ys[on] = s.y.clone()
        for a, b in zip(outs[False], outs[True]):
            assert torch.equal(a, b), (R, D, x3, skip)
        assert torch.isfinite(outs[True][-1]).all()
        if skip:
            assert torch.equal(ys[False], ys[True])        # y stored as before


@pytest.mark.parametrize("on", [False, True])
def test_res_fold_equals_add(on, monkeypatch):
    """TF_DSV41_RES_FOLD: routed(res=shared) == routed() + shared bit for bit, R = 1..32, with the combine waiting on a
    side stream that makes ``res`` (and with the L2 discards on)."""

    E, D, I, slots = 24, 4096, 1024, 6
    ex = layer(E, D, I, [8] * E, 77)
    monkeypatch.setattr(ex3, "DISCARD", on)
    g = torch.Generator(device="cuda").manual_seed(13)
    s = ex3.Scratch(ex, 32, slots)
    side, main = torch.cuda.Stream(), torch.cuda.current_stream()
    for R in range(1, 33):
        x = torch.randn((R, D), generator=g, device="cuda").to(torch.bfloat16)
        pick = torch.stack([torch.randperm(E, generator=g, device="cuda")[:slots] for _ in range(R)]).int()
        wts = torch.rand((R, slots), generator=g, device="cuda")
        shared = torch.randn((R, D), generator=g, device="cuda") * 3
        want = ex3.routed(x, pick, wts, ex, s, None, R) + shared
        side.wait_stream(main)
        with torch.cuda.stream(side):
            res = shared * 1.0
        got = ex3.routed(x, pick, wts, ex, s, None, R, res=res, before_combine=lambda: main.wait_stream(side))
        assert torch.equal(got, want), R
