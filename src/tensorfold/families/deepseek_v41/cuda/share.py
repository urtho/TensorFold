"""``TF_DSV41_SEND=one``: rank 0's messages to rank 1 (``Dsv41Engine._share``) in one exchange, without a host sync.

The default (``two``) sends each message as two blocking exchanges: its length, then its values, each a pageable
H2D copy, a gather and a ``.tolist()`` on both ranks (0.31-0.44 ms of rank 0's host time a ROUND message, DEV.md).
Here a message of up to MSG - 1 values goes as one fixed MSG-word exchange, ``[len, values..., stale]``:

- rank 0 writes it into the next of RING pinned host buffers (waiting first on that buffer's event, recorded after
  its last H2D copy), copies it to the device without blocking, gathers and returns its values without a sync;
- rank 1 gathers zeros, reads rank 0's MSG words (its sync) and takes ``len`` values; len 0 is the empty message;
- a longer message sends ``-len`` as its header, then its values in the default's second exchange (both ranks
  decide from the header alone).

Same messages, same order, same contents on rank 1, so the same rounds and tokens. Rank 0's copy and gather are no
longer ordered before the next graph replay by a host sync, only by stream order, and the RoCE gather's device
sequence counter is shared with the graphs' gathers: so both must run on the stream the graphs replay on (the default
stream, checked on every send). Rank 1 stays on the critical path (one exchange, its read, its parse).

After jayleaton/deepseek-v41-tensorfold-spark's plan link (``TF_DSV41_PLAN_LINK``, G14; Apache-2.0): the idea of one
non-blocking exchange a round, no code copied.
"""

from __future__ import annotations

MSG = 64        # int64 words an exchange: a header and up to 63 values (a ROUND of 16 streams is 34)
RING = 4        # pinned staging buffers rank 0 cycles through


class OneExchange:
    """One rank's side of ``TF_DSV41_SEND=one``; ``share(values)`` on rank 0, ``share(None)`` on rank 1.

    ``device="cpu"`` (tests) skips the pinned memory and the stream check; ``event`` makes each buffer's event."""

    def __init__(self, comm, rank: int, *, device: str = "cuda", event=None) -> None:
        import torch

        self.torch, self.comm, self.rank, self.device = torch, comm, rank, device
        cuda = device != "cpu"
        self.bufs = [torch.zeros(MSG, dtype=torch.int64, pin_memory=cuda) for _ in range(RING)]
        self.views = [b.numpy() for b in self.bufs]
        make = event if event is not None else (torch.cuda.Event if cuda else None)
        self.events = [make() if make is not None else None for _ in range(RING)]
        self.used = [False] * RING
        self.next = 0
        self.send = torch.zeros(MSG, dtype=torch.int64, device=device)     # rank 1 sends these zeros, always
        self.recv = torch.empty(2 * MSG, dtype=torch.int64, device=device)
        self.stream = torch.cuda.default_stream() if cuda else None
        self.waits = 0                                   # sends that found their buffer's last copy still running

    def _check_stream(self) -> None:
        torch = self.torch
        if self.stream is not None and (torch.cuda.current_stream() != self.stream
                                        or torch.cuda.is_current_stream_capturing()):
            raise RuntimeError("TF_DSV41_SEND=one: a message sent off the default stream (or inside a capture); its "
                               "copy and gather must be stream-ordered with the graph replays")

    def share(self, values: list[int] | None) -> list[int]:
        self._check_stream()
        if self.rank == 0:
            return self._send(values)
        return self._receive()

    def _send(self, values: list[int]) -> list[int]:
        n = len(values)
        i = self.next
        self.next = (i + 1) % RING
        buf, view, ev = self.bufs[i], self.views[i], self.events[i]
        if self.used[i] and ev is not None and not ev.query():    # its previous H2D copy may still be reading it
            self.waits += 1
            ev.synchronize()
        self.used[i] = True
        short = n < MSG
        view[0] = n if short else -n
        if short and n:
            view[1:1 + n] = values
        self.send.copy_(buf, non_blocking=True)
        if ev is not None:
            ev.record()
        self.comm.all_gather(self.send, self.recv)
        if not n and self.stream is not None:            # the end message (shutdown, _warm) blocks as the default's
            self.torch.cuda.current_stream().synchronize()   # .tolist() did: rank 1 has joined before rank 0 goes on
        if short:
            return view[1:1 + n].tolist()
        torch = self.torch                               # (the default's second exchange, synced as it is)
        got = torch.empty((2 * n,), dtype=torch.int64, device=self.device)
        self.comm.all_gather(torch.tensor(values, dtype=torch.int64, device=self.device), got)
        return got[:n].tolist()

    def _receive(self) -> list[int]:
        self.comm.all_gather(self.send, self.recv)
        head = self.recv[:MSG].tolist()
        n = head[0]
        if n >= 0:
            return head[1:1 + n]
        torch, n = self.torch, -n
        got = torch.empty((2 * n,), dtype=torch.int64, device=self.device)
        self.comm.all_gather(torch.zeros((n,), dtype=torch.int64, device=self.device), got)
        return got[:n].tolist()
