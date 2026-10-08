"""Triton kernels whose binary is first loaded after the server said ready.

Triton loads a compiled kernel into the CUDA context lazily, on its first launch (``CompiledKernel._init_handles``).
A specialization the warm-up never ran therefore loads mid-serving: a pause on that request, and on GB10 a load has
failed there (CUDA 800, "operation not permitted", after 15.5 h of serving in a peer engine: bertholomus/TensorFold#1),
which takes the lane down. ``arm()`` after the warm-up prints one line for every such load, so the warm-up can be made
to cover it; ``late`` holds them (kernel name -> loads) for /health. Nothing changes when Triton lacks the hook.

``trace()`` (a boot's warm-up audit, TF_DSV41_WARM_TRACE): every load from then on is also recorded, first loads
only, and ``take()`` hands back those since the last call, so each startup stage and warm-up wave can name the
kernels it loaded first. Host-only (a list append under a lock): safe inside a graph capture.
"""

from __future__ import annotations

import threading
import time

late: dict[str, int] = {}
_lock = threading.Lock()
_armed: list[float] = []
_tracing: list[bool] = []
_seen: set[str] = set()
_new: list[str] = []


def _key(name, hash) -> str:
    return f"{name}:{str(hash)[:12]}"


def _trace_hook(module, function, name, metadata_group, hash) -> None:
    key = _key(name, hash)
    with _lock:
        if key not in _seen:
            _seen.add(key)
            _new.append(key)


def _hook(module, function, name, metadata_group, hash) -> None:
    if not _armed:
        return
    with _lock:
        key = _key(name, hash)
        late[key] = late.get(key, 0) + 1
    print(f"[tensorfold] late kernel load {time.monotonic() - _armed[0]:.0f}s after ready: {key}", flush=True)


def _chain():
    try:
        from triton import knobs

        return knobs.runtime.kernel_load_start_hook
    except (ImportError, AttributeError):
        return None


def trace() -> bool:
    """Record every kernel's first load from now on (idempotent); False when this Triton has no load hook."""

    chain = _chain()
    if chain is None:
        return False
    if not _tracing:
        chain.add(_trace_hook)
        _tracing.append(True)
    return True


def take() -> list[str]:
    """The kernels first loaded since the last call (in load order; [] when not tracing)."""

    with _lock:
        out = list(_new)
        _new.clear()
    return out


def arm() -> bool:
    """Start reporting (idempotent); False when this Triton has no load hook."""

    chain = _chain()
    if chain is None:
        return False
    if not _armed:
        chain.add(_hook)
        _armed.append(time.monotonic())
    return True
