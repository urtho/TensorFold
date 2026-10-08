"""TF_RDMA_TRACE: the phase medians from the kernel's and the proxy's rings, the ext / ABI bumps that come with them."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("torch")

import tensorfold.cuda.rdma as R
from tensorfold.cuda.rdma import trace_phases

HERE = Path(R.__file__).parent


def _gpu_entry(seq, t0, stage=3000, wait=5000, copy=2000, tail=500, polls=7):
    rung = t0 + stage
    return [seq, t0, t0 + stage // 2, rung, rung + wait, rung + wait + copy, rung + wait + copy + tail, polls]


def test_phases_medians():
    n = 8
    gpu = [[0] * 8 for _ in range(n)]
    proxy = [[0] * 4 for _ in range(n)]
    for k, seq in enumerate(range(11, 16)):                      # seq 11..15 completed, 10 before the window
        gpu[seq & 7] = _gpu_entry(seq, 1_000_000 * (k + 1), wait=4000 + 1000 * k)
        proxy[seq & 7] = [seq, 100 + k, 300 + k, 2300 + k]
    gpu[10 & 7] = _gpu_entry(10, 0, wait=999_000)               # before ``first``: left out
    st = trace_phases(gpu, proxy, 10, 15)
    assert st["gathers"] == 5
    assert st["stage"] == pytest.approx(3.0) and st["copy"] == pytest.approx(2.0) and st["tail"] == pytest.approx(0.5)
    assert st["wait"] == pytest.approx(6.0)                      # 4, 5, 6, 7, 8 us
    assert st["max_wait"] == pytest.approx(8.0)
    assert st["total"] == pytest.approx(11.5) and st["polls"] == 7
    assert st["post"] == pytest.approx(0.2) and st["ack"] == pytest.approx(2.0)


def test_phases_latest_lap_and_partial_entries():
    n = 4
    gpu = [[0] * 8 for _ in range(n)]
    for seq in range(1, 11):                                     # ten gathers through a ring of four
        gpu[seq & 3] = _gpu_entry(seq, 10_000 * seq)
    gpu[9 & 3][6] = 0                                            # seq 9 never finished (a stop): left out
    st = trace_phases(gpu, [], 0, 10)
    assert st["gathers"] == 3                                    # 7, 8, 10 (the ring holds 7..10)
    assert "post" not in st


def test_phases_peer_flag_before_ring():
    """Block 0 may see an early peer's flag before another block rings: a negative wait, still counted."""

    e = _gpu_entry(1, 1000)
    e[4] = e[2] + 10                                             # flag seen after block 0 staged, before the ring
    st = trace_phases([[0] * 8, e], [], 0, 1)
    assert st["gathers"] == 1 and st["wait"] < 0


def test_phases_empty():
    assert trace_phases([[0] * 8] * 4, [], 5, 5) == {"gathers": 0.0}


def test_ext_name_bumped(monkeypatch):
    """gather.cu changed: a new extension name, or the boxes reuse a stale cached build."""

    import tensorfold.cuda.build as B

    seen = {}
    monkeypatch.setattr(B, "load", lambda name, sources, **kw: seen.setdefault("name", name))
    R._ext.cache_clear()
    try:
        R._ext()
    finally:
        R._ext.cache_clear()
    assert seen["name"] == "tensorfold_rdma_gather_v4"


def test_proxy_abi_and_trace_symbol():
    src = (HERE / "rdma_proxy.c").read_text()
    assert "#define TF_RDMA_ABI 2" in src and "void tf_rdma_trace(" in src
    assert "lib.tf_rdma_abi() != 2" in (HERE / "__init__.py").read_text()


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
def test_proxy_builds_with_trace(tmp_path):
    """The proxy with the trace ring compiles and reports ABI 2 (where the libibverbs headers are installed)."""

    out = tmp_path / "p.so"
    done = subprocess.run(["gcc", "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(out), str(HERE / "rdma_proxy.c"),
                           "-libverbs", "-lpthread"], capture_output=True, text=True, check=False)
    if done.returncode:
        pytest.skip(f"no libibverbs to build against: {done.stderr.strip().splitlines()[-1:]}")
    lib = ctypes.CDLL(str(out))
    assert lib.tf_rdma_abi() == 2
    assert hasattr(lib, "tf_rdma_trace")
