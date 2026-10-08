"""Small all-gathers between two machines over RoCE, through pinned host memory (no GPUDirect: GB10 has none).

NCCL's small all-gathers between two DGX Sparks cost ~19 us of fixed latency each (two per layer per decode step).
Here the GPU stages its shard in pinned host memory, a host thread RDMA-writes it and a flag to the peer, and the
peer's GPU polls the flag in its own pinned memory: one kernel per gather, CUDA-graph replayable, the same output
layout (rank order) and bits as NCCL. ``RdmaComm`` sends gathers that fit a slot this way and the rest through NCCL.
Host side: ``rdma_proxy.c``; GPU side: ``gather.cu``.

The protocol is b12x's "RoCEnante" one-shot all-gather (https://github.com/local-inference-lab/b12x,
b12x/comm/roce/ at commit 8a99d639410e; Apache License 2.0, Luke Alonso and the b12x contributors), as ported in
MiaAI-Lab's GLM-5.3-Flash TensorFold recipe (patch 0006-cuda-roce-allgather, roce.py; Apache License 2.0, Copyright
2026 MiaAI-Lab). Reduced to two ranks and one QP; the NCCL_IB_HCA parse, the NCCL fall-through and the connect error
follow that port.

TF_RDMA_TRACE=N (a power of two; off by default): each gather's phases stamped into rings of N entries, the kernel's
in %globaltimer ns (``gather.cu``), the proxy's in CLOCK_REALTIME ns (``rdma_proxy.c``); ``RdmaGather.trace_stats``
reads them as per-phase medians (``tools/dsv41_serial_run.py --decode-bench`` prints them on both ranks).
"""

from __future__ import annotations

import ctypes
import hashlib
import ipaddress
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

import torch

HERE = Path(__file__).parent
SLOT_ALIGN = 4096
INFO_WORDS = 7                       # tf_rdma_info: qpn, psn, rkey, addr, mtu, gid_hi, gid_lo
SPIN = 20_000_000                    # flag polls before a wait gives up (~20 s): the peer died or never sent
TRACE = int(os.environ.get("TF_RDMA_TRACE") or 0)       # trace ring entries (0: off)
_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def proxy() -> ctypes.CDLL:
    """The host proxy, built with the C compiler and libibverbs once per source hash."""

    src = HERE / "rdma_proxy.c"
    tag = hashlib.sha256(src.read_bytes()).hexdigest()[:12]
    out = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "tensorfold" / f"tf_rdma-{tag}.so"
    with _LOCK:
        if not out.exists():
            cc = shutil.which(os.environ.get("CC", "gcc")) or shutil.which("cc")
            if cc is None:
                raise RuntimeError("the RoCE all-gather needs a C compiler and the libibverbs headers")
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(f".{os.getpid()}.tmp")
            done = subprocess.run([cc, "-O2", "-std=gnu11", "-shared", "-fPIC", "-o", str(tmp), str(src), "-libverbs",
                                   "-lpthread"], capture_output=True, text=True, check=False)
            if done.returncode:
                raise RuntimeError(f"building the RoCE proxy failed:\n{done.stderr}")
            tmp.replace(out)                             # atomic: the other process may build the same file
    lib = ctypes.CDLL(str(out))
    u64, vp, i = ctypes.c_uint64, ctypes.c_void_p, ctypes.c_int
    for name, res, args in (("tf_rdma_abi", i, []), ("tf_rdma_layout", None, [u64, ctypes.POINTER(u64)]),
                            ("tf_rdma_create", vp, [ctypes.c_char_p, i, vp, u64, u64, ctypes.c_char_p, u64]),
                            ("tf_rdma_local", None, [vp, ctypes.POINTER(u64)]),
                            ("tf_rdma_connect", i, [vp, ctypes.POINTER(u64)]), ("tf_rdma_start", i, [vp]),
                            ("tf_rdma_failed", i, [vp]), ("tf_rdma_error", ctypes.c_char_p, [vp]),
                            ("tf_rdma_counter", u64, [vp, i]), ("tf_rdma_destroy", None, [vp]),
                            ("tf_rdma_trace", None, [vp, vp, u64])):
        fn = getattr(lib, name)
        fn.restype, fn.argtypes = res, args
    if lib.tf_rdma_abi() != 2:
        raise RuntimeError("unexpected RoCE proxy ABI")
    return lib


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    return load("tensorfold_rdma_gather_v3", [str(HERE / "gather.cpp"), str(HERE / "gather.cu")],
                extra_cuda_cflags=["-O3"])


def _device() -> str:
    """This rank's RoCE device: TF_RDMA_HCA, else the first of NCCL_IB_HCA (``=dev[:port],...``)."""

    raw = os.environ.get("TF_RDMA_HCA") or os.environ.get("NCCL_IB_HCA") or ""
    names = [x.strip().lstrip("=^").split(":")[0] for x in raw.split(",") if x.strip()]
    if not names:
        raise RuntimeError("the RoCE all-gather needs TF_RDMA_HCA or NCCL_IB_HCA to name the RoCE device")
    return names[0]


def roce_v2_ipv4_gids(device: str, port: int = 1, root: str = "/sys/class/infiniband") -> list[tuple[int, str]]:
    """The port's RoCE v2 GIDs that carry an IPv4 address, as (index, address), lowest index first."""

    base = Path(root) / device / "ports" / str(port)
    found = []
    try:
        names = sorted((p.name for p in (base / "gids").iterdir() if p.name.isdigit()), key=int)
    except OSError:
        return []
    for name in names:
        try:                                     # an empty or stale slot reads as zeros or fails (EINVAL)
            gid = (base / "gids" / name).read_text().strip()
            kind = (base / "gid_attrs" / "types" / name).read_text().strip()
        except OSError:
            continue
        raw = bytes.fromhex(gid.replace(":", ""))
        if kind == "RoCE v2" and len(raw) == 16 and raw[:12] == bytes(10) + b"\xff\xff":
            found.append((int(name), str(ipaddress.IPv4Address(raw[12:]))))
    return found


def gid_index(device: str, port: int = 1, root: str = "/sys/class/infiniband") -> int:
    """This rank's source GID: TF_RDMA_GID_INDEX if set, else the port's RoCE v2 IPv4 GID found by type and address
    (inside TF_RDMA_ADDR_RANGE or NCCL_IB_ADDR_RANGE when one is set), else NCCL_IB_GID_INDEX.

    The index of that GID is not stable: the kernel keeps a deleted GID's slot while a queue pair still uses it, so
    an address re-added under a live connection (the peer rebooting drops the link; NetworkManager removes and
    re-adds the address) takes the next free slot (3 becomes 4)."""

    explicit = os.environ.get("TF_RDMA_GID_INDEX")
    if explicit:
        return int(explicit)
    found = roce_v2_ipv4_gids(device, port, root)
    span = os.environ.get("TF_RDMA_ADDR_RANGE") or os.environ.get("NCCL_IB_ADDR_RANGE")
    if span:
        net = ipaddress.ip_network(span.strip(), strict=False)
        found = [(i, a) for i, a in found if ipaddress.ip_address(a) in net]
    if found:
        return found[0][0]
    fallback = os.environ.get("NCCL_IB_GID_INDEX")
    if fallback:
        return int(fallback)
    raise RuntimeError(f"{device} port {port} has no RoCE v2 IPv4 GID{' in ' + span if span else ''} (an address "
                       "on its interface?); TF_RDMA_GID_INDEX names one")


# TF_RDMA_HOST=register (default): the region is page-aligned anonymous memory, locked and registered with
# cudaHostRegister. GB10's GPU reaches cudaHostAlloc memory (pin_memory) over an uncached path, registered memory
# over the coherent cached one (measured by bertholomus/TensorFold bd0024d on 4 MB: loads 6.3 vs 20.4 us, stores 6.2
# vs 20.7 us), so the gather's stage, flag polls and peer reads wait less behind DRAM traffic. pinned: pin_memory.
HOST = os.environ.get("TF_RDMA_HOST") or "register"


def _region(total: int) -> tuple[torch.Tensor, str]:
    """The zeroed host region the kernel and the NIC share, and how it was made (``register`` or ``pinned``)."""

    if HOST == "register":
        import mmap

        mem = mmap.mmap(-1, total, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        region = torch.frombuffer(mem, dtype=torch.uint8)
        ptr = region.data_ptr()
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mlock(ctypes.c_void_p(ptr), ctypes.c_size_t(total))         # (the NIC's registration pins it too)
        if torch.cuda.cudart().cudaHostRegister(ptr, total, 3) == 0:      # Portable | Mapped
            if _ext().device_pointer(ptr) == ptr:                           # the kernel takes the host address
                region._tf_keep = mem                                       # the mapping lives as long as the tensor
                return region, "register"
            torch.cuda.cudart().cudaHostUnregister(ptr)
        del region                                    # (the mapping is freed with its last export)
    return torch.zeros(total, dtype=torch.uint8).pin_memory(), "pinned"


class RdmaGather:
    """One rank's pinned region, queue pair and proxy thread; ``all_gather`` launches the gather kernel."""

    def __init__(self, nccl, rank: int, max_bytes: int) -> None:
        if nccl.world != 2:
            raise ValueError("the RoCE all-gather joins exactly two ranks")
        lib = proxy()
        self.lib, self.rank = lib, rank
        self.slot_bytes = -(-int(max_bytes) // SLOT_ALIGN) * SLOT_ALIGN
        lay = (ctypes.c_uint64 * 5)()
        lib.tf_rdma_layout(self.slot_bytes, lay)
        self.flag_off, self.send_off, self.recv_off, total = int(lay[1]), int(lay[2]), int(lay[3]), int(lay[4])
        self.region, self.host = _region(total)                               # ctrl and flags start at 0
        self.ctrl = self.region[:64].view(torch.int32).numpy()
        self.state = torch.zeros(4, dtype=torch.int32, device="cuda")          # epoch, arrivals, departures, stopped
        self.trace = self.ptrace = None
        if TRACE:                                        # (gather.cu / rdma_proxy.c list the entries' words)
            if TRACE & (TRACE - 1):
                raise ValueError(f"TF_RDMA_TRACE={TRACE}: a power of two (the ring's entries)")
            self.trace = torch.zeros((TRACE, 8), dtype=torch.int64, device="cuda")
            self.ptrace = torch.zeros((TRACE, 4), dtype=torch.int64)
        self.device = _device()
        err = ctypes.create_string_buffer(256)
        try:
            self.gid = gid_index(self.device)
        except RuntimeError as exc:                      # both ranks still meet in the exchange below
            self.gid, self.ctx = -1, None
            err.value = str(exc).encode()[:255]
        else:
            self.ctx = lib.tf_rdma_create(self.device.encode(), self.gid, ctypes.c_void_p(self.region.data_ptr()),
                                          total, self.slot_bytes, err, len(err))
        mine = torch.zeros(INFO_WORDS + 2, dtype=torch.int64)
        if self.ctx:
            info = (ctypes.c_uint64 * INFO_WORDS)()
            lib.tf_rdma_local(self.ctx, info)
            mine[:INFO_WORDS] = torch.tensor([v - (1 << 64) if v >= 1 << 63 else v for v in map(int, info)],
                                             dtype=torch.int64)       # uint64 words as int64 for the exchange
        mine[INFO_WORDS] = 0 if self.ctx else 1
        mine[INFO_WORDS + 1] = self.slot_bytes
        both = torch.empty(2 * mine.numel(), dtype=torch.int64, device="cuda")
        nccl.all_gather(mine.cuda(), both)
        rows = both.view(2, -1).cpu()
        if int(rows[:, INFO_WORDS].sum()) or int(rows[0, INFO_WORDS + 1]) != int(rows[1, INFO_WORDS + 1]):
            why = err.value.decode(errors="replace") if not self.ctx else "the ranks' slot sizes differ"
            self.close()
            raise RuntimeError(f"RoCE all-gather setup failed on a rank: {why}")
        _ext()                                           # build the kernel before the connection's verdict
        if self.ptrace is not None:
            lib.tf_rdma_trace(self.ctx, ctypes.c_void_p(self.ptrace.data_ptr()), TRACE)
        peer = rows[1 - rank, :INFO_WORDS].tolist()
        words = (ctypes.c_uint64 * INFO_WORDS)(*[v & 0xFFFFFFFFFFFFFFFF for v in peer])
        ok = lib.tf_rdma_connect(self.ctx, words) == 0 and lib.tf_rdma_start(self.ctx) == 0
        verdict = torch.tensor([0 if ok else 1], dtype=torch.int64, device="cuda")
        votes = torch.empty(2, dtype=torch.int64, device="cuda")
        nccl.all_gather(verdict, votes)
        if int(votes.sum()):
            why = lib.tf_rdma_error(self.ctx).decode(errors="replace") if self.ctx else ""
            self.close()
            raise RuntimeError(f"RoCE all-gather could not connect: {why}")

    def fits(self, send: torch.Tensor, recv: torch.Tensor) -> bool:
        n = send.numel() * send.element_size()
        return (0 < n <= self.slot_bytes and n % 16 == 0 and send.is_contiguous() and recv.is_contiguous()
                and send.data_ptr() % 16 == 0 and recv.data_ptr() % 16 == 0
                and recv.numel() * recv.element_size() == 2 * n)

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        _ext().gather(send, recv, self.region.data_ptr(), self.flag_off, self.send_off, self.recv_off, self.slot_bytes,
                      self.state, SPIN, self.rank, 0 if self.trace is None else self.trace.data_ptr(), max(TRACE - 1, 0))
        if not torch.cuda.is_current_stream_capturing():
            self.check()

    def trace_seq(self) -> int:
        """The last completed gather's sequence number (syncs): the start of a ``trace_stats`` window."""

        return int(self.state[0].item())

    def trace_stats(self, since: int) -> dict[str, float] | None:
        """Per-phase medians of the gathers after ``since`` (TF_RDMA_TRACE; None: off), see ``trace_phases``."""

        if self.trace is None:
            return None
        torch.cuda.synchronize()
        return trace_phases(self.trace.cpu().tolist(), self.ptrace.tolist(), since, self.trace_seq())

    def trace_dump(self, path) -> None:
        """Both raw rings as JSON (TF_RDMA_TRACE)."""

        import json

        if self.trace is not None:
            torch.cuda.synchronize()
            Path(path).write_text(json.dumps({"rank": self.rank, "gpu": self.trace.cpu().tolist(),
                                              "proxy": self.ptrace.tolist()}))

    def check(self) -> None:
        """Raise if a GPU wait gave up (ctrl[4]) or the proxy thread failed (ctrl[3])."""

        if int(self.ctrl[4]) or int(self.ctrl[3]) or (self.ctx and self.lib.tf_rdma_failed(self.ctx)):
            why = self.lib.tf_rdma_error(self.ctx).decode(errors="replace") if self.ctx else ""
            raise RuntimeError(f"RoCE all-gather failed (wait gave up at seq {int(self.ctrl[4])}, rung "
                               f"{int(self.ctrl[0])}, posted {self.lib.tf_rdma_counter(self.ctx, 0)}, completed "
                               f"{self.lib.tf_rdma_counter(self.ctx, 1)}){': ' + why if why else ''}")

    def close(self) -> None:
        if getattr(self, "ctx", None):
            self.lib.tf_rdma_destroy(self.ctx)
            self.ctx = None


def _median(xs: list[float]) -> float:
    xs = sorted(xs)
    n = len(xs)
    return (xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2) if n else float("nan")


def trace_phases(gpu: list[list[int]], proxy_ring: list[list[int]], first: int, last: int) -> dict[str, float]:
    """Median us of each phase over the gathers with first < seq <= last whose entries are whole (the ring's latest
    lap only). gpu entries: seq, start, staged, rung, flag, copied, end, polls (``gather.cu``); proxy: seq, seen,
    posted, completed (``rdma_proxy.c``). stage = start..rung (the local shard staged, fences, doorbell); wait =
    rung..flag (the proxy, the wire and the peer's lateness: the rank that waits less is the later one; below 0 when
    the peer's flag was in before another block rang); copy = flag..copied (block 0's copy-out); tail = copied..end;
    post = seen..posted and ack = posted..completed on the proxy (a write's round trip). ``p90_total`` and
    ``max_wait`` show the outliers."""

    mask = len(gpu) - 1
    rows = [gpu[q & mask] for q in range(max(first + 1, last - mask), last + 1)]
    rows = [r for r in rows if r[0] > first and 0 < r[1] <= r[2] <= r[3] and r[2] <= r[4] <= r[5] <= r[6]]
    out: dict[str, float] = {"gathers": float(len(rows))}
    if not rows:
        return out
    us = 1e-3
    for name, a, b in (("stage", 1, 3), ("wait", 3, 4), ("copy", 4, 5), ("tail", 5, 6), ("total", 1, 6)):
        out[name] = _median([(r[b] - r[a]) * us for r in rows])
    totals = sorted((r[6] - r[1]) * us for r in rows)
    out["p90_total"] = totals[min(len(totals) - 1, int(0.9 * len(totals)))]
    out["max_wait"] = max((r[4] - r[3]) * us for r in rows)
    out["polls"] = _median([float(r[7]) for r in rows])
    pmask = len(proxy_ring) - 1 if proxy_ring else 0
    seqs = {r[0] for r in rows}
    prox = [proxy_ring[q & pmask] for q in seqs] if proxy_ring else []
    prox = [p for p in prox if p[0] in seqs and 0 < p[1] <= p[2] <= p[3]]
    if prox:
        out["post"] = _median([(p[2] - p[1]) * us for p in prox])
        out["ack"] = _median([(p[3] - p[2]) * us for p in prox])
    return out


class RdmaComm:
    """An NCCL communicator whose small all-gathers go over RoCE: the same calls, output layout and bits."""

    def __init__(self, nccl, rank: int, max_bytes: int | None = None) -> None:
        self.nccl, self.rank, self.world = nccl, rank, nccl.world
        kb = int(os.environ.get("TF_RDMA_MAX_KB") or 704) if max_bytes is None else max_bytes >> 10
        self.rdma = RdmaGather(nccl, rank, kb << 10)
        self.settled = False
        self.small = self.large = 0

    def settle(self) -> None:
        """Startup is over (graphs captured): eager gathers no longer meet the peer in an NCCL barrier first. Until
        then the ranks may drift apart by whole kernel builds, longer than a RoCE wait allows."""

        self.settled = True

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if self.rdma.fits(send, recv):
            self.small += 1
            if not self.settled and not torch.cuda.is_current_stream_capturing():
                self.nccl.barrier()
            self.rdma.all_gather(send, recv)
        else:
            self.large += 1
            self.nccl.all_gather(send, recv)

    def barrier(self) -> None:
        self.nccl.barrier()

    def check(self) -> None:
        self.rdma.check()

    def __getattr__(self, name):
        return getattr(self.nccl, name)
