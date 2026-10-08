"""TF_DSV41_SEND=one (``share.OneExchange``): rank 1 receives exactly rank 0's messages, in order, for every length
(empty, the one-exchange limit, the two-exchange continuation), and a staging buffer is reused only after its copy."""

from __future__ import annotations

import threading

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda.share import MSG, RING, OneExchange


class PairComm:
    """Two ranks' all-gathers in one process (CPU tensors): each call meets the other rank's call of the same order."""

    def __init__(self) -> None:
        self.bar = threading.Barrier(2)
        self.slot: list = [None, None]
        self.calls = [[], []]                      # (send numel) per rank, the exchange sequence

    def rank(self, r: int):
        pair = self

        class Side:
            def all_gather(self, send, recv):
                pair.calls[r].append(send.numel())
                pair.slot[r] = send.clone()
                pair.bar.wait()
                if pair.slot[0].numel() != pair.slot[1].numel():
                    raise RuntimeError("the ranks' exchanges are out of step")
                recv.copy_(torch.cat([pair.slot[0], pair.slot[1]]))
                pair.bar.wait()

        return Side()


class FakeEvent:
    def __init__(self, log: list) -> None:
        self.log, self.done = log, True

    def record(self) -> None:
        self.log.append("record")
        self.done = False                          # a copy in flight until someone waits

    def query(self) -> bool:
        return self.done

    def synchronize(self) -> None:
        self.log.append("wait")
        self.done = True


LENGTHS = [0, 1, 2, MSG - 2, MSG - 1, MSG, MSG + 1, 5000, 0, 34, 3]


def _run(messages: list[list[int]]):
    comm, log = PairComm(), []
    a = OneExchange(comm.rank(0), 0, device="cpu", event=lambda: FakeEvent(log))
    b = OneExchange(comm.rank(1), 1, device="cpu")
    got0, got1 = [], []

    def follower():
        for _ in messages:
            got1.append(b.share(None))

    t = threading.Thread(target=follower)
    t.start()
    for m in messages:
        got0.append(a.share(m))
    t.join(timeout=30)
    assert not t.is_alive()
    return got0, got1, comm, a, log


def test_messages_round_trip():
    msgs = [[(7 * i + j) * (-1) ** j for j in range(n)] for i, n in enumerate(LENGTHS)]
    msgs[3][0] = 2 ** 63 - 1                       # the int64 range survives
    msgs[4][-1] = -2 ** 63
    got0, got1, _, _, _ = _run(msgs)
    assert got1 == msgs and got0 == msgs
    assert all(type(v) is int for m in got0 + got1 for v in m)


def test_exchange_sequence():
    """One MSG-word exchange a message; a long one (MSG values or more) adds one exchange of its length."""

    msgs = [list(range(n)) for n in LENGTHS]
    _, _, comm, _, _ = _run(msgs)
    want = []
    for n in LENGTHS:
        want += [MSG] if n < MSG else [MSG, n]
    assert comm.calls[0] == want and comm.calls[1] == want


def test_ring_reuse_waits_on_its_event():
    msgs = [[i] for i in range(3 * RING)]
    _, got1, _, a, log = _run(msgs)
    assert got1 == msgs
    # the first RING sends find fresh buffers; every later one waits for its buffer's copy (never completed here)
    assert a.waits == 2 * RING
    assert log[:RING] == ["record"] * RING and log[RING:RING + 2] == ["wait", "record"]


def test_values_like_ints():
    """bools and numpy ints are sent as their int values, as torch.tensor(values, int64) did."""

    np = pytest.importorskip("numpy")
    got0, got1, _, _, _ = _run([[True, np.int64(5), 3]])
    assert got0 == got1 == [[1, 5, 3]]


def test_engine_switch_default_is_two(monkeypatch):
    import importlib

    import tensorfold.families.deepseek_v41.cuda.engine as E

    monkeypatch.delenv("TF_DSV41_SEND", raising=False)
    assert importlib.reload(E).SEND == "two"
    monkeypatch.setenv("TF_DSV41_SEND", "three")
    with pytest.raises(ValueError):
        importlib.reload(E)
    monkeypatch.setenv("TF_DSV41_SEND", "one")
    assert importlib.reload(E).SEND == "one"
    monkeypatch.delenv("TF_DSV41_SEND")
    importlib.reload(E)
