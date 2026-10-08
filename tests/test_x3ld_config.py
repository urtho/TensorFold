"""x3ld's switches (TF_EXPERT_LOADS_CFG, TF_X3LD_ORDER, TF_X3LD_PROBE), the width ranges, the instances x3ld.cu
builds for them, and the expert-major order's index map (CPU; the kernels' Z: tests/cuda/test_x3ld_experts.py)."""

import re
from pathlib import Path

import pytest

from tensorfold.cuda.exl3 import x3ld

CU = (Path(x3ld.__file__).parent / "x3ld.cu").read_text()


@pytest.fixture
def env(monkeypatch):
    for k in ("TF_EXPERT_LOADS", "TF_EXPERT_LOADS_CFG", "TF_X3LD_ORDER", "TF_X3LD_PROBE"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_defaults_are_todays(env):
    assert x3ld._parse() == {"on": True, "gu": (4, 2), "dn": (4, 2), "order": 0, "probe": 0}


def test_order_and_deeper_rings_parse(env):
    env.setenv("TF_X3LD_ORDER", "expert")
    env.setenv("TF_EXPERT_LOADS_CFG", "4,4/4,3")
    c = x3ld._parse()
    assert (c["order"], c["gu"], c["dn"]) == (1, (4, 4), (4, 3))
    env.setenv("TF_X3LD_ORDER", "grid")
    env.setenv("TF_EXPERT_LOADS_CFG", "8,3")
    c = x3ld._parse()
    assert (c["order"], c["gu"], c["dn"]) == (0, (8, 3), (8, 3))


@pytest.mark.parametrize("name,value", [("TF_X3LD_ORDER", "fast"), ("TF_EXPERT_LOADS_CFG", "4,5"),
                                        ("TF_X3LD_PROBE", "1")])
def test_bad_values_are_refused(env, name, value):
    env.setenv(name, value)
    with pytest.raises(ValueError):
        x3ld._parse()


def test_probe_only_where_built(env):
    env.setenv("TF_X3LD_PROBE", "3")
    env.setenv("TF_EXPERT_LOADS_CFG", "4,4/4,3")
    assert x3ld._parse()["probe"] == 3
    env.setenv("TF_EXPERT_LOADS_CFG", "4,4/8,3")
    with pytest.raises(ValueError):
        x3ld._parse()


def test_model_shapes_take_the_deeper_rings():
    # gate/up: K 5120 at 4 splits, 20 k steps a warp; down: K 1152 at 1, 18
    assert x3ld.fits(5120, 1152, 4, 4, (4, 4)) and x3ld.fits(5120, 1152, 4, 4, (4, 2))
    assert x3ld.fits(1152, 5120, 1, 4, (4, 3)) and x3ld.fits(1152, 5120, 1, 4, (8, 3))
    assert not x3ld.fits(1152, 5120, 1, 4, (4, 4))     # 18 % 4: upstream's kernel (the same Z) would run


def test_width_ranges():
    for pd in (1, 2):                                   # today's instances, whatever the widths
        assert x3ld.k2_range(4, 6, pd) == (2, 10)
        assert x3ld.k2_range(8, 8, pd) == (8, 8)
    for pd in (3, 4):
        assert x3ld.k2_range(4, 6, pd) == (4, 6)
        assert x3ld.k2_range(4, 4, pd) == (4, 6)
        assert x3ld.k2_range(4, 10, pd) == (2, 10)     # TF_FOLD_SHARED's wider layers
        assert x3ld.k2_range(8, 8, pd) == (8, 8)
        assert x3ld.k2_range(2, 12, pd) is None
    assert x3ld.k2_range(4, 6) == (2, 10)


def test_grouped_passes_order_probe_and_range(monkeypatch):
    calls = []

    class Ext:
        def grouped(self, *a):
            calls.append(a)

    monkeypatch.setattr(x3ld, "_ext", lambda: Ext())
    monkeypatch.setitem(x3ld.CFG, "on", True)
    monkeypatch.setitem(x3ld.CFG, "order", 1)
    monkeypatch.setitem(x3ld.CFG, "probe", 0)
    monkeypatch.setitem(x3ld.CFG, "gu", (4, 4))
    monkeypatch.setitem(x3ld.CFG, "dn", (4, 3))
    args = [None] * 10
    assert x3ld.grouped(*args, 2, 5120, 1152, 36, 4, 6, 2, 4, 4, 6)
    assert x3ld.grouped(*args, 1, 1152, 5120, 36, 1, 6, 2, 4, 4, 6)
    assert not x3ld.grouped(*args, 1, 1152, 5120, 36, 1, 6, 2, 4, 4, 6, cfg=(4, 4))
    assert x3ld.grouped(*args, 1, 1152, 5120, 36, 1, 6, 2, 4, 4, 6, cfg=(4, 2), order=0)
    # (nt, pd, probe, lo, hi, pdl, order)
    assert [c[17:] for c in calls] == [(4, 4, 0, 4, 6, False, 1), (4, 3, 0, 4, 6, False, 1),
                                       (4, 2, 0, 2, 10, False, 0)]


def test_every_cfg_is_instantiated():
    body = CU[CU.index("bool dispatch_cfg("):]
    body = body[:body.index("\n}\n")]
    built = {(int(n), int(p)) for n, p in re.findall(r"nt == (\d+) && pd == (\d+)", body)}
    assert built == set(x3ld.CFGS)
    assert "(8, 3): no probe instance" in body and (8, 3) not in x3ld.PROBE_CFGS
    assert set(x3ld.PROBE_CFGS) == built - {(8, 3)}
    assert "lo == 4 && hi == 6" in CU and "PD >= 3" in CU


def test_extension_name_bumped():
    assert 'name="tf_x3ld_v2"' in Path(x3ld.__file__).read_text()


def program(order, x, y, z, X, Y, Z):
    """x3ld.cu program<ORDER, D>() for D = 0, 1, 2 (unsigned arithmetic as the kernel's)."""

    if order == 0:
        return x, y, z
    lin = x + X * (y + Y * z)
    return lin // (Y * Z), lin % (Y * Z) % Y, lin % (Y * Z) // Y


@pytest.mark.parametrize("X,Y,Z", [(6, 9, 4), (36, 9, 4), (36, 40, 1), (1, 1, 1), (24, 18, 8), (7, 3, 5)])
def test_expert_major_is_a_permutation_with_dead_slots_last(X, Y, Z):
    seen = []
    for lin in range(X * Y * Z):                        # the hardware's launch order: x fastest
        x, rest = lin % X, lin // X
        y, z = rest % Y, rest // Y
        u, by, bz = program(1, x, y, z, X, Y, Z)
        assert 0 <= u < X and 0 <= by < Y and 0 <= bz < Z
        seen.append((u, by, bz))
    assert sorted(seen) == sorted(program(0, x, y, z, X, Y, Z) for x in range(X) for y in range(Y) for z in range(Z))
    us = [u for u, _, _ in seen]
    assert us == sorted(us)                             # expert-major: one expert's programs adjacent
    for cnt in range(X + 1):                            # every live count: the dead slots are the tail
        dead = [u >= cnt for u in us]
        assert dead == sorted(dead)
    assert [by for _, by, _ in seen[:Y]] == list(range(Y))   # n blocks fastest within an expert
