"""TF_DSV41_ROT_FUSE without a GPU: the switch's tokens and default (off), the launchers passing it, the extension name
bumps, where the producers sit in the kernels, and the linear wrappers handing the extension the right rows (a
recording stand-in). The bit-for-bit checks against rot_in run on the GPU: tests/cuda/test_dsv41_rot_fuse.py."""

import importlib
from pathlib import Path

import pytest

pytest.importorskip("triton")
torch = pytest.importorskip("torch")

from tensorfold.cuda.exl3 import linear
from tensorfold.families.deepseek_v41.cuda import serial as S

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def reload(monkeypatch):
    def go(**env):
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        return importlib.reload(S)

    yield go
    monkeypatch.undo()
    importlib.reload(S)


@pytest.mark.parametrize("value,want", [("", set()), ("0", set()), ("attn", {"attn"}), ("wob", {"wob"}),
                                        ("attn,wob", {"attn", "wob"}), ("1", {"attn", "wob"}),
                                        ("all", {"attn", "wob"}), (" wob , attn", {"attn", "wob"})])
def test_switch_tokens(reload, value, want):
    assert reload(TF_DSV41_ROT_FUSE=value).ROT_FUSE == want


def test_switch_rejects_unknown(reload):
    with pytest.raises(ValueError, match="q"):
        reload(TF_DSV41_ROT_FUSE="attn,q")


def test_default_off(monkeypatch, reload):
    monkeypatch.delenv("TF_DSV41_ROT_FUSE", raising=False)
    assert reload().ROT_FUSE == set()
    assert S.SerialEngine._rot_fuse == frozenset()          # (engines built without __init__ too)


def test_launchers_pass_the_switch():
    name = "TF_DSV41_ROT_FUSE"
    ranks = [ln for ln in (ROOT / "tools/dsv41_run2.sh").read_text().splitlines()
             if "dsv41_serial_run.py" in ln and "docker exec" in ln]
    assert len(ranks) == 2 and all(f"-e {name}=${{{name}:-}}" in ln for ln in ranks)
    assert f"{name}: ${{{name}:-}}" in (ROOT / "deploy/dsv41-tp2/docker-compose.yaml").read_text()
    assert f"-e {name}=${{{name}:-}}" in (ROOT / "tools/dsv41_serve2.sh").read_text()


def test_extension_names_bumped():
    """linear.cu / linear.cpp and mqa_fp4.cu changed signatures: a stale build under the old name must not load."""

    assert 'name="tensorfold_exl3_linear_v7"' in (ROOT / "src/tensorfold/cuda/exl3/linear.py").read_text()
    src = (ROOT / "src/tensorfold/families/deepseek_v41/cuda/mqa_fp4.py").read_text()
    assert '"tf_dsv41_mqa_fp4_lut_v8" if lut else "tf_dsv41_mqa_fp4_v8"' in src


def test_producers_after_the_stored_values():
    """Both merges rotate inside the bf16 (RoPE) branch, after the stored row; both linear finish sites after store4."""

    cu = (ROOT / "src/tensorfold/families/deepseek_v41/cuda/mqa_fp4.cu").read_text()
    for a, b in (("merge_kernel(const float*", "merge_flat(const float*"),
                 ("merge_flat(const float*", "__global__ void table_kernel")):
        k = cu[cu.index(a):cu.index(b)]
        rope = k[k.index("if (COS != nullptr)"):k.index("} else {", k.index("if (COS != nullptr)"))]
        assert rope.index("__floats2bfloat162_rn(o[2], o[3])") < rope.index("rot128_bf16_store(o,")
    lin = (ROOT / "src/tensorfold/cuda/exl3/linear.cu").read_text()
    body = lin[lin.index("linear_kernel("):lin.index("unpack_kernel(")]
    assert body.count("store_rot(v, y_dtype") == 2
    rot = (ROOT / "src/tensorfold/cuda/exl3/rot128.cuh").read_text()
    core = rot[rot.index("void pair("):rot.index("void rot128_store(")]
    code = [ln.split("//")[0] for ln in core.splitlines() if not ln.strip().startswith(("for", "//"))]
    for op in (" * ", " + ", " - ", "*=", "+=", "-="):          # every float operation an explicit intrinsic
        assert not any(op in ln for ln in code), op


class _Ext:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def call(*a, **k):
            self.calls.append((name, a, k))
        return call


def _grouped(groups=2, k=256, n=128):
    layers = []
    for _ in range(groups):
        words = torch.zeros((n // 128, k // 16, 8, 32), dtype=torch.int32)
        layers.append(linear.Exl3Linear(words, torch.ones(k, dtype=torch.float16), torch.ones(n, dtype=torch.float16),
                                        None, 4, "mul1", k, n))
    return linear.GroupedLinear(layers)


def test_grouped_rotated_calls(monkeypatch):
    rec = _Ext()
    monkeypatch.setattr(linear, "_ext", lambda: rec)
    g = _grouped()
    xh = torch.zeros((3, 512), dtype=torch.float16)
    y = linear.grouped_rotated(g, xh, torch.bfloat16)
    assert [c[0] for c in rec.calls] == ["linear"] and y.shape == (3, 256) and y.dtype == torch.bfloat16
    assert rec.calls[0][1][0] is xh
    rec.calls.clear()
    xo, suh = torch.zeros((3, 256), dtype=torch.float16), torch.ones(256, dtype=torch.float16)
    x = torch.zeros((3, 512), dtype=torch.bfloat16)
    linear.grouped_rotated(g, None, torch.bfloat16, x=x, rot_out=(suh, xo, 4))
    assert [c[0] for c in rec.calls] == ["rot_in", "linear_rot_out"]
    a = rec.calls[1][1]
    assert a[-3] is xo and a[-2] is suh and a[-1] == 4 and a[0] is rec.calls[0][1][2]
    rec.calls.clear()
    g(x)                                                       # the default path: unchanged (rot_in + linear)
    assert [c[0] for c in rec.calls] == ["rot_in", "linear"]


def test_linear_rotated_calls(monkeypatch):
    rec = _Ext()
    monkeypatch.setattr(linear, "_ext", lambda: rec)
    lay = _grouped(1, 512, 128)
    m = linear.Exl3Linear(lay.words, lay.suh, lay.svh, None, 4, "mul1", 512, 128)
    xh = torch.zeros((2, 512), dtype=torch.float16)
    y = linear.linear_rotated(m, xh, torch.float32)
    assert [c[0] for c in rec.calls] == ["linear"] and y.dtype == torch.float32 and rec.calls[0][1][0] is xh
    with pytest.raises(ValueError):
        linear.linear_rotated(m, xh.float(), torch.float32)
