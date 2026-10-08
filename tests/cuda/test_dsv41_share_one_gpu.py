"""TF_DSV41_SEND=one on the GPU (one process, a loopback all-gather): rank 0's non-blocking pinned copies deliver
every message's exact words in stream order while the host refills the ring behind a busy GPU, and a send off the
default stream or inside a capture is refused."""

from __future__ import annotations

import pytest
import torch

from tensorfold.families.deepseek_v41.cuda.share import MSG, RING, OneExchange


class Loopback:
    """Rank 0's all-gather with a silent peer: keeps a device copy of every exchange's first half (stream order)."""

    def __init__(self) -> None:
        self.seen: list[torch.Tensor] = []

    def all_gather(self, send, recv):
        n = send.numel()
        recv[:n].copy_(send)
        recv[n:].zero_()
        self.seen.append(recv[:n].clone())


def test_words_arrive_behind_a_busy_gpu():
    comm = Loopback()
    one = OneExchange(comm, 0)
    msgs = [[(31 * i + j) % 100003 - 50000 for j in range(i % (MSG + 3))] for i in range(200)]
    for i, m in enumerate(msgs):
        if i % 25 == 0:
            torch.cuda._sleep(20_000_000)             # the GPU busy: the copies queue, the ring must wait
        assert one.share(m) == m
    torch.cuda.synchronize()
    assert one.waits > 0
    k = 0
    for m in msgs:
        head = comm.seen[k].tolist()
        k += 1
        if len(m) < MSG:
            assert head[0] == len(m) and head[1:1 + len(m)] == m
        else:
            assert head[0] == -len(m)
            assert comm.seen[k].tolist() == m        # the continuation exchange
            k += 1
    assert k == len(comm.seen)


def test_refused_off_the_default_stream():
    one = OneExchange(Loopback(), 0)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side), pytest.raises(RuntimeError, match="default stream"):
        one.share([1, 2, 3])
    assert one.share([4]) == [4]


def test_refused_inside_a_capture():
    one = OneExchange(Loopback(), 0)
    g = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError), torch.cuda.graph(g):
        one.share([1])


def test_ring_size():
    assert RING == 4 and MSG == 64
