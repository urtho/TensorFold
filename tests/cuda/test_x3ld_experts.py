"""x3ld (TF_EXPERT_LOADS) routed experts give bit-identical results to upstream's grouped kernel, under every
TF_EXPERT_LOADS_CFG and TF_X3LD_ORDER (GPU)."""

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


# TF_X3LD_ORDER / deeper rings: Z bit for bit against today's (grid order, "4,2") at the model's per-rank shapes
# (gate/up 5120 -> 1152 at 4 K splits: 20 k steps a warp; down 1152 -> 5120 at 1: 18), every Z cell compared --
# the written ones and the ones left alone (a sentinel), so an order writes exactly the same cells.
DSV41_E, DSV41_D, DSV41_I, DSV41_SLOTS = 32, 5120, 1152, 6
SENTINEL = 1234.5


def picks(R, E, slots, g):
    """Distinct experts a row; odd rows from a pool of 8 (experts shared by rows); every third row a dead slot (E)."""

    rows = []
    for r in range(R):
        p = torch.randperm(8 if r % 2 else E, generator=g, device="cuda")[:slots].int()
        if r % 3 == 2:
            p[-1] = E
        rows.append(p)
    return torch.stack(rows)


def z_of(ex, R, x0, x1, pick, mats, cfg, order, probe=0):
    s = ex3.Scratch(ex, R, DSV41_SLOTS)
    ids, members = s.window(R)
    ex3._ext().group(pick, ids, s.count, members, R, DSV41_SLOTS, ex.count)
    z = torch.full_like(s.z, SENTINEL)
    P = R * DSV41_SLOTS
    if mats == 2:
        _, w, sk, _ = s.cfg_gu
        ok = x3ld.grouped(x0, x1, ex.gate_ptr, ex.up_ptr, ex.gate_k2, ex.up_k2, ids, s.count, members, z, 2,
                          ex.dims, ex.width, P, sk, DSV41_SLOTS, ex.cb, w, ex.k2_gu[0], ex.k2_gu[1], cfg=cfg,
                          order=order, probe=probe)
    else:
        _, w, sk, _ = s.cfg_d
        ok = x3ld.grouped(x0, x0, ex.down_ptr, ex.down_ptr, ex.down_k2, ex.down_k2, ids, s.count, members, z, 1,
                          ex.width, ex.dims, P, sk, DSV41_SLOTS, ex.cb, w, ex.k2_d[0], ex.k2_d[1], cfg=cfg,
                          order=order, probe=probe)
    assert ok, (mats, cfg, order)
    return z


@pytest.fixture(scope="module", params=["routed", "wide", "dspark"])
def dsv41_layer(request):
    E = DSV41_E
    widths = {"routed": [4 + 2 * (e % 2) for e in range(E)],          # 2/3-bit: the 4..6 instance at pd >= 3
              "wide": [4 + 2 * (e % 4) for e in range(E)],            # up to 5 bits: 2..10 (pd 4 spills there)
              "dspark": [8] * E}[request.param]                       # DSpark's 4-bit experts: 8..8
    return request.param, layer(E, DSV41_D, DSV41_I, widths, 11 + len(request.param))


GU_CFGS = [(4, 2), (4, 4), (8, 2), (8, 1)]
DN_CFGS = [(4, 2), (4, 3), (8, 3), (8, 2)]


@pytest.mark.parametrize("R", [1, 2, 4, 6, 16])
def test_orders_and_rings_give_todays_z(dsv41_layer, R):
    kind, ex = dsv41_layer
    g = torch.Generator(device="cuda").manual_seed(100 + R)
    pick = picks(R, ex.count, DSV41_SLOTS, g)
    P = R * DSV41_SLOTS
    xg = torch.randn((P, DSV41_D), generator=g, device="cuda").half()
    xu = torch.randn((P, DSV41_D), generator=g, device="cuda").half()
    xd = torch.randn((P, DSV41_I), generator=g, device="cuda").half()
    for mats, x0, x1, cfgs in ((2, xg, xu, GU_CFGS), (1, xd, xd, DN_CFGS)):
        ref = z_of(ex, R, x0, x1, pick, mats, (4, 2), 0)
        assert (ref != SENTINEL).any() and torch.isfinite(ref).all()
        for cfg in cfgs:
            for order in (0, 1):
                z = z_of(ex, R, x0, x1, pick, mats, cfg, order)
                assert torch.equal(z, ref), (kind, R, mats, cfg, order)


@pytest.mark.parametrize("order", ["grid", "expert"])
@pytest.mark.parametrize("cfg", ["4,2", "4,4/4,3", "8,2/8,3"])
def test_routed_with_order_and_cfg_equals_upstream(cfg, order, monkeypatch):
    """The whole routed() (group, rot_in, both launches, epilogues) under the switches as parsed, against upstream's."""

    monkeypatch.setenv("TF_EXPERT_LOADS_CFG", cfg)
    monkeypatch.setenv("TF_X3LD_ORDER", order)
    parsed = x3ld._parse()
    E = DSV41_E
    ex = layer(E, DSV41_D, DSV41_I, [4 + 2 * (e % 2) for e in range(E)], 3)
    g = torch.Generator(device="cuda").manual_seed(9)
    taken = []
    real = x3ld.grouped
    monkeypatch.setattr(x3ld, "grouped", lambda *a, **k: taken.append(real(*a, **k)) or taken[-1])
    for R in (1, 3, 6, 16):
        x = torch.randn((R, DSV41_D), generator=g, device="cuda").to(torch.bfloat16)
        pick = picks(R, E, DSV41_SLOTS, g)
        wts = torch.rand((R, DSV41_SLOTS), generator=g, device="cuda")
        outs = []
        for on in (False, True):
            for k, v in parsed.items():
                monkeypatch.setitem(x3ld.CFG, k, v)
            monkeypatch.setitem(x3ld.CFG, "on", on)
            s = ex3.Scratch(ex, 16, DSV41_SLOTS)
            outs.append(ex3.routed(x, pick, wts, ex, s, None, R).clone())
        assert torch.equal(outs[0], outs[1]), (R, cfg, order)
        assert taken[-2:] == [True, True]


@pytest.mark.parametrize("order", [0, 1])
@pytest.mark.parametrize("cfg", list(x3ld.PROBE_CFGS))
def test_probe_instances_launch(cfg, order):
    """PROBE 3 (timing only: a wrong Z by design) has its instances at the model's shapes."""

    E = DSV41_E
    ex = layer(E, DSV41_D, DSV41_I, [4 + 2 * (e % 2) for e in range(E)], 5)
    g = torch.Generator(device="cuda").manual_seed(1)
    R = 4
    pick = picks(R, E, DSV41_SLOTS, g)
    P = R * DSV41_SLOTS
    xg = torch.randn((P, DSV41_D), generator=g, device="cuda").half()
    xd = torch.randn((P, DSV41_I), generator=g, device="cuda").half()
    if x3ld.fits(DSV41_D, DSV41_I, 4, 4, cfg):
        z_of(ex, R, xg, xg, pick, 2, cfg, order, probe=3)
    if x3ld.fits(DSV41_I, DSV41_D, 1, 4, cfg):
        z_of(ex, R, xd, xd, pick, 1, cfg, order, probe=3)
    torch.cuda.synchronize()
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


def test_l2_discard_row_gate_on_one_scratch(monkeypatch):
    """TF_DSV41_L2_DISCARD_ROWS: one Scratch, calls alternating R in (1, 6, 2, 4) with the moe discard on for R >= 4
    only, give the bits of discard off throughout."""

    E, slots, D, I = 24, 7, 5120, 1152
    ex = layer(E, D, I, [8] * E, D + I)
    g = torch.Generator(device="cuda").manual_seed(11)
    calls = []
    for R in (1, 6, 2, 4) * 6:
        calls.append((R, torch.randn((R, D), generator=g, device="cuda").to(torch.bfloat16),
                      _picks(R, E, slots, g, E - 1), torch.rand((R, slots), generator=g, device="cuda")))
    outs = {}
    for on, rows in ((False, 0), (True, 4)):
        monkeypatch.setattr(ex3, "DISCARD", on)
        monkeypatch.setattr(ex3, "DISCARD_ROWS", rows)
        s = ex3.Scratch(ex, 16, slots)
        outs[on] = [ex3.routed(x, p, w, ex, s, None, R).clone() for R, x, p, w in calls]
    for a, b in zip(outs[False], outs[True]):
        assert torch.equal(a, b)
