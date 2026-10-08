"""TF_DSV41_L2_DISCARD (po / moe) and TF_DSV41_RES_FOLD without a GPU: the switches default off, routed() hands the
epilogues the right flags (a recording stand-in for the extension), and the launchers pass the switches to both ranks.
The bit-for-bit checks on/off run on the GPU: tests/cuda/test_x3ld_experts.py, tests/cuda/test_dsv41_mqa_fp4.py."""

import importlib
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import experts as ex3
from tensorfold.families.deepseek_v41.cuda import mqa_fp4

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reload(monkeypatch):
    """Re-import a module under the given environment; the default-environment module comes back afterwards."""

    mods = []

    def go(mod, **env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        mods.append(mod)
        return importlib.reload(mod)

    yield go
    monkeypatch.undo()
    for m in mods:
        importlib.reload(m)


@pytest.mark.parametrize("value,po,moe", [("", False, False), ("po", True, False), ("moe", False, True),
                                          ("po,moe", True, True), ("all", True, True), ("lin", False, False),
                                          ("po, moe", True, True), (" moe ,", False, True)])
def test_switch_tokens(reload, value, po, moe):
    assert reload(mqa_fp4, TF_DSV41_L2_DISCARD=value).DISCARD is po
    assert reload(ex3, TF_DSV41_L2_DISCARD=value).DISCARD is moe


@pytest.mark.parametrize("mod", ["mqa_fp4", "ex3"])
def test_switch_rejects_unknown(reload, mod):
    with pytest.raises(ValueError, match="moee"):
        reload(mqa_fp4 if mod == "mqa_fp4" else ex3, TF_DSV41_L2_DISCARD="po,moee")


def test_defaults_off(monkeypatch, reload):
    monkeypatch.delenv("TF_DSV41_L2_DISCARD", raising=False)
    monkeypatch.delenv("TF_DSV41_RES_FOLD", raising=False)
    assert reload(mqa_fp4).DISCARD is False
    assert reload(ex3).DISCARD is False
    src = (ROOT / "src/tensorfold/families/deepseek_v41/cuda/serial.py").read_text()
    assert 'RES_FOLD = os.environ.get("TF_DSV41_RES_FOLD") == "1"' in src


def test_extension_names_bumped():
    """experts.cu / experts.cpp and mqa_fp4.cu changed signatures: a stale build under the old name must not load."""

    assert 'name="tensorfold_exl3_experts_v2"' in (ROOT / "src/tensorfold/cuda/exl3/experts.py").read_text()
    src = (ROOT / "src/tensorfold/families/deepseek_v41/cuda/mqa_fp4.py").read_text()
    assert '"tf_dsv41_mqa_fp4_lut_v8" if lut else "tf_dsv41_mqa_fp4_v8"' in src


def test_merge_flat_barrier_before_discard():
    """merge_flat's threads load their own dims while a discarded line spans 8 threads: a barrier must come first."""

    cu = (ROOT / "src/tensorfold/families/deepseek_v41/cuda/mqa_fp4.cu").read_text()
    flat = cu[cu.index("merge_flat(const float*"):cu.index("__global__ void table_kernel")]
    tail = flat[flat.index("if (discard) {"):]
    assert tail.index("__syncthreads();") < tail.index("discard_po(")
    kern = cu[cu.index("merge_kernel(const float*"):cu.index("merge_flat(const float*")]
    assert kern.index("__syncthreads();") < kern.index("if (discard) discard_po(")


def test_launchers_pass_the_switches():
    run2 = (ROOT / "tools/dsv41_run2.sh").read_text().splitlines()
    ranks = [ln for ln in run2 if "dsv41_serial_run.py" in ln and "docker exec" in ln]
    assert len(ranks) == 2
    for name in ("TF_DSV41_L2_DISCARD", "TF_DSV41_RES_FOLD"):
        assert all(f"-e {name}=${{{name}:-}}" in ln for ln in ranks), name
        assert f"{name}: ${{{name}:-}}" in (ROOT / "deploy/dsv41-tp2/docker-compose.yaml").read_text()
        assert f"-e {name}=${{{name}:-}}" in (ROOT / "tools/dsv41_serve2.sh").read_text()


class _Ext:
    """Records the epilogue launches routed() makes."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*a, **k):
            self.calls.append((name, a, k))
        return call


def _layer(E=8, D=256, I=128):
    t = torch.zeros(1)
    h = torch.zeros((E, D), dtype=torch.float16)
    return ex3.Exl3RoutedExperts(t, t, t, t, t, t, h, h, torch.zeros((E, I), dtype=torch.float16),
                                 torch.zeros((E, I), dtype=torch.float16), torch.zeros((E, I), dtype=torch.float16), h,
                                 E, D, I, ex3.CB_MUL1, (8, 8), (8, 8), t)


@pytest.mark.parametrize("discard,skip,with_wts", [(False, False, True), (True, False, True), (True, True, True),
                                                   (True, False, False)])
def test_routed_flags(monkeypatch, discard, skip, with_wts):
    E, D, I, R, slots = 8, 256, 128, 3, 4
    ex = _layer(E, D, I)
    s = ex3.Scratch(ex, 4, slots, device="cpu")
    rec = _Ext()
    monkeypatch.setattr(ex3, "_ext", lambda: rec)
    monkeypatch.setattr(ex3, "DISCARD", discard)
    monkeypatch.setattr(ex3, "SKIP_SHARED", skip)
    monkeypatch.setitem(ex3.x3ld.CFG, "on", False)
    x = torch.zeros((R, D), dtype=torch.bfloat16)
    pick = torch.zeros((R, slots), dtype=torch.int32)
    wts = torch.ones((R, slots)) if with_wts else None
    res = torch.zeros((R, D)) if with_wts else None
    order = []
    ex3.routed(x, pick, wts, ex, s, None, R, limit=math.inf, res=res,
               before_combine=(lambda: order.append(len(rec.calls))) if with_wts else None)
    names = [c[0] for c in rec.calls]
    gu = rec.calls[names.index("gateup_epilogue")][1]
    on = int(discard and with_wts)
    P = R * slots
    assert gu[14] is s.xg and gu[15] is s.xu
    assert gu[16] == s.cfg_d[2] * P * D and gu[17] == on            # the down Z's end: lines below it not dropped
    if not with_wts:
        assert "down_combine" not in names and "down_epilogue" in names
        return
    i = names.index("down_combine")
    assert order == [i]                                             # before_combine right before the combine
    dc = rec.calls[i][1]
    assert dc[12] is res
    assert dc[13] == (0 if discard and not skip else 1)             # y stored unless dropped (kept under SKIP_SHARED)
    assert dc[14] is s.xd and dc[15] == on


def test_routed_res_needs_wts(monkeypatch):
    ex = _layer()
    s = ex3.Scratch(ex, 4, 4, device="cpu")
    monkeypatch.setattr(ex3, "_ext", lambda: _Ext())
    monkeypatch.setitem(ex3.x3ld.CFG, "on", False)
    with pytest.raises(ValueError):
        ex3.routed(torch.zeros((2, 256)), torch.zeros((2, 4), dtype=torch.int32), None, ex, s, None, 2,
                   res=torch.zeros((2, 256)))
