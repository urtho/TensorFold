"""Triton kernels whose binary is first loaded after the server said ready.

Triton loads a compiled kernel into the CUDA context lazily, on its first launch (``CompiledKernel._init_handles``).
A specialization the warm-up never ran therefore loads mid-serving: a pause on that request, and on GB10 a load has
failed there (CUDA 800, "operation not permitted", after 15.5 h of serving in a peer engine: bertholomus/TensorFold#1),
which takes the lane down. ``arm()`` after the warm-up prints one line for every such load, so the warm-up can be made
to cover it; ``late`` holds them (kernel name -> loads) for /health. Nothing changes when Triton lacks the hook.
"""

from __future__ import annotations

import threading
import time

late: dict[str, int] = {}
_lock = threading.Lock()
_armed: list[float] = []


def _hook(module, function, name, metadata_group, hash) -> None:
    if not _armed:
        return
    with _lock:
        key = f"{name}:{str(hash)[:12]}"
        late[key] = late.get(key, 0) + 1
    print(f"[tensorfold] late kernel load {time.monotonic() - _armed[0]:.0f}s after ready: {key}", flush=True)


def arm() -> bool:
    """Start reporting (idempotent); False when this Triton has no load hook."""

    try:
        from triton import knobs

        chain = knobs.runtime.kernel_load_start_hook
    except (ImportError, AttributeError):
        return False
    if not _armed:
        chain.add(_hook)
        _armed.append(time.monotonic())
    return True
