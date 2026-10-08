"""TF_RDMA_TRACE on the GPU, one process: the host plays the peer (its payload in recv[slot], then its flag), so the
gather kernel runs without a NIC. With the trace ring the kernel gives the same bytes (output, send slot, ctrl words,
timeout record) as without it, eager and graph-replayed, and its stamps come in phase order."""

from __future__ import annotations

import pytest
import torch

from tensorfold.cuda import rdma as R

SLOT = 128 << 10
FLAG, SEND, RECV = 64, 4096, 4096 + 2 * SLOT
TOTAL = 4096 + 4 * SLOT
SIZES = [16, 1024, 16 << 10, 96 << 10, SLOT]       # 16 KiB: one fp32 row (R=1), 96 KiB: R=6, a whole slot
TRACE = 16


class Side:
    """One fake rank: its pinned region, device state and (optionally) trace ring."""

    def __init__(self, trace: bool) -> None:
        self.region = torch.zeros(TOTAL, dtype=torch.uint8).pin_memory()
        self.host = self.region.numpy()
        self.state = torch.zeros(4, dtype=torch.int32, device="cuda")
        self.trace = torch.zeros((TRACE, 8), dtype=torch.int64, device="cuda") if trace else None
        self.seq = 0

    def peer(self, payload: torch.Tensor, flag: bool = True) -> None:
        """The peer's next message: payload into recv[slot], then its seq flag (as the NIC writes them)."""

        self.seq += 1
        slot = self.seq & 1
        raw = payload.view(torch.uint8).cpu().numpy()
        self.host[RECV + slot * SLOT:RECV + slot * SLOT + raw.size] = raw
        if flag:
            self.host[FLAG + slot * 64:FLAG + slot * 64 + 4].view("<u4")[0] = self.seq

    def gather(self, x: torch.Tensor, out: torch.Tensor, rank: int, spin: int = 50_000_000) -> None:
        ptr, mask = (0, 0) if self.trace is None else (self.trace.data_ptr(), TRACE - 1)
        R._ext().gather(x, out, self.region.data_ptr(), FLAG, SEND, RECV, SLOT, self.state, spin, rank, ptr, mask)

    def ctrl(self) -> list[int]:
        return self.host[:20].view("<u4").tolist()

    def sent(self, nbytes: int) -> bytes:
        slot = self.seq & 1
        return self.host[SEND + slot * SLOT:SEND + slot * SLOT + nbytes].tobytes()


def _payload(nbytes: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(-2 ** 31, 2 ** 31 - 1, (nbytes // 4,), dtype=torch.int32, generator=g).cuda()


def _check_stamps(side: Side) -> None:
    for e in side.trace.cpu().tolist():
        if e[0] == 0:
            continue
        _, start, staged, rung, flag, copied, end, _ = e
        assert 0 < start <= staged <= rung and staged <= flag <= copied <= end, e


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("nbytes", SIZES)
def test_trace_on_equals_off(nbytes, rank):
    off, on = Side(False), Side(True)
    for i in range(2 * TRACE + 3):                  # the trace ring wraps twice
        x, theirs = _payload(nbytes, 2 * i), _payload(nbytes, 2 * i + 1)
        outs = []
        for side in (off, on):
            side.peer(theirs)
            out = torch.empty(2 * nbytes // 4, dtype=torch.int32, device="cuda")
            side.gather(x, out, rank)
            outs.append(out)
        torch.cuda.synchronize()
        want = torch.cat([x, theirs] if rank == 0 else [theirs, x])
        assert torch.equal(outs[0], want) and torch.equal(outs[1], want)
        assert off.sent(nbytes) == on.sent(nbytes) == x.view(torch.uint8).cpu().numpy().tobytes()
        assert off.ctrl() == on.ctrl() and off.ctrl()[0] == off.seq
        assert off.state.tolist() == on.state.tolist() and int(on.state[0]) == on.seq
    _check_stamps(on)
    st = R.trace_phases(on.trace.cpu().tolist(), [], on.seq - TRACE, on.seq)
    assert st["gathers"] == TRACE


def test_trace_graph_replay_equals_eager():
    nbytes, rank = 16 << 10, 0
    eager, graphed = Side(False), Side(True)
    x = torch.empty(nbytes // 4, dtype=torch.int32, device="cuda")
    out = torch.empty(2 * nbytes // 4, dtype=torch.int32, device="cuda")
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):                          # (the capture runs nothing: replays take seq 1, 2, ...)
        graphed.gather(x, out, rank)
    for i in range(40):
        xi, theirs = _payload(nbytes, 3 * i), _payload(nbytes, 3 * i + 1)
        eager.peer(theirs)
        ref = torch.empty_like(out)
        eager.gather(xi, ref, rank)
        graphed.peer(theirs)
        x.copy_(xi)
        g.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, ref)
        assert eager.ctrl() == graphed.ctrl() and eager.state.tolist() == graphed.state.tolist()
    _check_stamps(graphed)


def test_trace_timeout_records_as_before():
    """The peer never flags: both kernels record the sequence in ctrl[4] and state[3] and stop further gathers."""

    nbytes = 1024
    sides = [Side(False), Side(True)]
    for side in sides:
        side.peer(_payload(nbytes, 1), flag=False)
        out = torch.zeros(2 * nbytes // 4, dtype=torch.int32, device="cuda")
        side.gather(_payload(nbytes, 2), out, 0, spin=2000)
        torch.cuda.synchronize()
        side.peer(_payload(nbytes, 3))
        side.gather(_payload(nbytes, 4), out, 0)       # stopped: returns at once
        torch.cuda.synchronize()
    assert sides[0].ctrl() == sides[1].ctrl() and sides[0].ctrl()[4] == 1
    assert sides[0].state.tolist() == sides[1].state.tolist() == [0, 8, 8, 1]
