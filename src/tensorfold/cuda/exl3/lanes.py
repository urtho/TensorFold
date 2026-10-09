"""TF_EXL3_LANES: the dense EXL3 linears over the "lanes" layout (``lanes.cu``; jayleaton's DENSE_V3, MIT): the same
bits as ``linear.cu``'s kernel, other loads.

The layout is a load-time transform, never stored: a prepared (fastboot) tree keeps strips words, and ``convert``
repacks a loaded tree's words in place (same storage, so data pointers the L2 tables hold stay valid) and swaps each
converted object's class to a lanes subclass (``LanesExl3Linear`` / ``LanesGroupedLinear``, or a subclass of the
object's own class such as dsv41's ``weights.Linear``) whose ``__call__`` / ``unpack`` read lanes. ``Exl3Linear``
and ``GroupedLinear`` (whose sources the fastboot key digests) are untouched, and the strips module functions
(``linear_rotated``, ``grouped_rotated``, ``prefill.matmul``) dispatch on ``layout == "lanes"``.

Rules (each one can change tokens if broken):
- convert once per storage: the wo_a slices are views of their GroupedLinear's words; the group's storage is
  repacked once, then the group and every slice inside it are marked lanes;
- convert before anything holds strips decodes or kernels: CUDA graph captures, warm-ups, prefill workspaces
  (``Workspace.held``) — ``convert`` is called at the top of the engine's constructor;
- set identically on both ranks (deploy compose, tools/dsv41_run2.sh, tools/dsv41_serve2.sh);
- TF_EXL3_LANES_TWO (default off) picks lanes.cu's two-group kernel for 17-32-row calls (same bits), taken by the
  extension at load; set it identically on both ranks too (the engine checks);
- never with PDL around these kernels (jayleaton saw DENSE_V3 + PDL not bit-exact); lanes.cu has no griddepcontrol.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

from .linear import CODEBOOK_IDS, Exl3Linear, GroupedLinear
from .linear import _ext as _strips_ext

# on by default since 2026-10-09 (DeepSeek-V4.1 converts at load; jaybench +1%, with TF_L2_ATTN_WQB_MB=8 +1.7-2%,
# out/perf/ab10-*); TF_EXL3_LANES=0: strips as before
ENABLED = os.environ.get("TF_EXL3_LANES", "1") != "0"
HEAD = os.environ.get("TF_EXL3_LANES_HEAD", "0") == "1"     # the vocabulary head too (off: it stays on strips)
# 17-32-row calls in one pass of two 16-row groups: each weight tile read and decoded once for both (lanes.cu NG 2;
# same bits; default off until measured on GB10)
TWO = os.environ.get("TF_EXL3_LANES_TWO", "0") == "1"
K2S = (4, 6, 8, 10, 12)                                      # lanes.cu's widths (even, take_bits' 64-bit reach)
WKS = (4, 8)


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(__file__).parent
    ext = load(name="tensorfold_exl3_lanes_v2", sources=[str(here / "lanes.cpp"), str(here / "lanes.cu")],
               extra_include_paths=[str(here)], extra_cuda_cflags=["-O3", "--expt-relaxed-constexpr"],
               verbose=False)
    ext.set_two(TWO)
    return ext


def why_not(k2: int, k: int, split: tuple[int, int] | None) -> str | None:
    """None when a layer of width K2, K inputs and its actual (K splits, warps) can take the lanes kernel."""

    if k2 not in K2S:
        return f"K2 {k2} (lanes: {K2S})"
    if split is None:
        return "no split"
    sk, wk = split
    kt = k // 16
    if wk not in WKS:
        return f"{wk} warps (lanes: {WKS})"
    if kt % (sk * wk):
        return f"K/16 {kt} over {sk} x {wk}"
    if (kt // (sk * wk)) % 2:
        return f"{kt // (sk * wk)} k steps a warp (odd: lanes groups 2)"
    return None


def lanes_ok(k2: int, k: int, split: tuple[int, int] | None) -> bool:
    return why_not(k2, k, split) is None


def strips_strides(k: int, k2: int) -> tuple[int, int]:
    tw = 4 * k2
    return 8 * tw, (k // 16) * 8 * tw


def relayout_(words: torch.Tensor, k2: int, k: int, to_lanes: bool = True) -> None:
    """``words`` (whole strips, [.., K/16, 8, 4 K2] int32) repacked in place: same storage and data pointer."""

    tmp = torch.empty_like(words)
    _ext().relayout(words, tmp, k2, k, to_lanes)
    words.copy_(tmp)
    del tmp


class LanesExl3Linear(Exl3Linear):
    """An Exl3Linear whose words are in the lanes layout (made by ``convert``; never stored)."""

    @property
    def strides(self) -> tuple[int, int]:
        return strips_strides(self.k, self.k2)

    def __call__(self, x: torch.Tensor, out: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
                 xh: torch.Tensor | None = None, z: torch.Tensor | None = None) -> torch.Tensor:
        # Exl3Linear.__call__ with the lanes kernel; ROT_FUSE's in-kernel rotation is not taken (it is bit-identical
        # to rot_in + linear by construction, so the outputs are the same)
        if x.dim() != 2 or x.shape[1] != self.k or not 1 <= x.shape[0] <= 128:
            raise ValueError(f"x must be [1..128, {self.k}], got {tuple(x.shape)}")
        x = x.contiguous()
        m = x.shape[0]
        if out is None:
            out = torch.empty((m, self.n), dtype=out_dtype or x.dtype, device=x.device)
        sk, wk = self.split
        if xh is None:
            xh = torch.empty((m, self.k), dtype=torch.float16, device=x.device)
        if sk > 1 and z is None:
            z = torch.empty((sk * m * self.n,), dtype=torch.float32, device=x.device)
        _strips_ext().rot_in(x, self.suh, xh)
        _ext().linear(xh, self.words, *self.strides, self.svh, self.bias, out, z if sk > 1 else None, self.counters,
                      self.k2, CODEBOOK_IDS[self.codebook], sk, wk, 0, 0)
        return out

    def unpack(self, out: torch.Tensor | None = None) -> torch.Tensor:
        w = out if out is not None else torch.empty((self.k, self.n), dtype=torch.float16, device=self.words.device)
        _ext().unpack(self.words, w, *self.strides, self.k2, CODEBOOK_IDS[self.codebook])
        return w


class LanesGroupedLinear(GroupedLinear):
    """A GroupedLinear whose shared words are in the lanes layout."""

    def __call__(self, x: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
        m, N = x.shape[0], self.groups * self.n
        if x.dim() != 2 or x.shape[1] != self.groups * self.k or not 1 <= m <= 128:
            raise ValueError(f"x must be [1..128, {self.groups * self.k}], got {tuple(x.shape)}")
        x = x.contiguous()
        out = torch.empty((m, N), dtype=out_dtype or x.dtype, device=x.device)
        xh = torch.empty((m, self.groups * self.k), dtype=torch.float16, device=x.device)
        sk, wk = self.split
        z = torch.empty((sk * m * N,), dtype=torch.float32, device=x.device) if sk > 1 else None
        _strips_ext().rot_in(x, self.suh, xh)
        _ext().linear(xh, self.words, *self.strides, self.svh, None, out, z, self.counters, self.k2,
                      CODEBOOK_IDS[self.codebook], sk, wk, self.k, self.n)
        return out


_CLASSES: dict[type, type] = {Exl3Linear: LanesExl3Linear, GroupedLinear: LanesGroupedLinear}


def lanes_class(cls: type) -> type:
    """The lanes class of an Exl3Linear / GroupedLinear class: for a subclass (dsv41 ``weights.Linear``, whose
    ``__call__`` reaches ``super().__call__``), class(cls, Lanes*) so the subclass's own logic stays first and its
    super() calls land on the lanes methods."""

    if cls in _CLASSES.values() or getattr(cls, "_lanes", False):
        return cls
    got = _CLASSES.get(cls)
    if got is None:
        base = LanesGroupedLinear if issubclass(cls, GroupedLinear) else LanesExl3Linear
        got = type(f"Lanes{cls.__name__}", (cls, base), {"_lanes": True, "__module__": __name__})
        _CLASSES[cls] = got
    return got


def is_lanes(obj) -> bool:
    return getattr(obj, "layout", None) == "lanes"


def _span(t: torch.Tensor) -> tuple[int, int]:
    return t.data_ptr(), t.data_ptr() + t.numel() * t.element_size()


def collect(tree, skip: set[int] | None = None) -> tuple[list[GroupedLinear], list[Exl3Linear]]:
    """Every GroupedLinear and Exl3Linear reachable from ``tree`` (dataclasses, lists, tuples, dicts), once each."""

    seen: set[int] = set() if skip is None else skip
    groups, layers = [], []

    def walk(o) -> None:
        if o is None or isinstance(o, (torch.Tensor, str, bytes, int, float, bool)) or id(o) in seen:
            return
        seen.add(id(o))
        if isinstance(o, GroupedLinear):
            groups.append(o)
            return
        if isinstance(o, Exl3Linear):
            layers.append(o)
            return
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)
        elif hasattr(o, "__dataclass_fields__"):
            for name in o.__dataclass_fields__:
                walk(getattr(o, name, None))

    walk(tree)
    return groups, layers


def convert(tree, *, exclude: tuple = (), log=print) -> dict:
    """Repack every eligible strips layer of ``tree`` to lanes in place, once per storage (TF_EXL3_LANES); objects
    in ``exclude`` (and those sharing their storage) stay on strips. Returns counts. Call before any graph capture,
    warm-up or prefill workspace use; resets dsv41's prefill workspace if one exists."""

    skip = {id(o) for o in exclude}
    groups, layers = collect(tree, set(skip))
    done: list[tuple[int, int]] = []                       # repacked storage spans
    kept: list[tuple[int, int]] = [_span(o.words) for o in exclude if hasattr(o, "words")]
    n_mat = n_bytes = n_skip = 0
    why: dict[str, int] = {}

    def overlaps(span, spans) -> bool:
        return any(span[0] < b and a < span[1] for a, b in spans)

    def inside(span, spans) -> bool:
        return any(a <= span[0] and span[1] <= b for a, b in spans)

    for g in groups:
        if is_lanes(g):
            continue
        sp = _span(g.words)
        reason = why_not(g.k2, g.k, g.split) or ("excluded" if overlaps(sp, kept) else None)
        if reason is None and overlaps(sp, done):
            raise RuntimeError("lanes: a grouped linear's words overlap words already repacked")
        if reason is not None:
            kept.append(sp)
            why[reason] = why.get(reason, 0) + 1
            n_skip += 1
            continue
        relayout_(g.words, g.k2, g.k)
        done.append(sp)
        g.__class__ = lanes_class(type(g))
        g.layout = "lanes"
        n_mat += 1
        n_bytes += g.words.numel() * 4
    for m in layers:
        if is_lanes(m):
            continue
        if m.layout != "strips":
            why[f"layout {m.layout}"] = why.get(f"layout {m.layout}", 0) + 1
            n_skip += 1
            continue
        sp = _span(m.words)
        if inside(sp, done):                               # a slice of a repacked group: mark only
            if why_not(m.k2, m.k, m.split) is not None:
                raise RuntimeError(f"lanes: a slice of a repacked group cannot take lanes ({why_not(m.k2, m.k, m.split)})")
            m.__class__ = lanes_class(type(m))
            m.layout = "lanes"
            continue
        if overlaps(sp, done):
            raise RuntimeError("lanes: a linear's words partly overlap words already repacked")
        reason = why_not(m.k2, m.k, m.split) or ("excluded" if overlaps(sp, kept) else None)
        if reason is not None:
            why[reason] = why.get(reason, 0) + 1
            n_skip += 1
            continue
        relayout_(m.words, m.k2, m.k)
        done.append(sp)
        m.__class__ = lanes_class(type(m))
        m.layout = "lanes"
        n_mat += 1
        n_bytes += m.words.numel() * 4
    torch.cuda.synchronize()
    _reset_workspaces()
    stats = {"matrices": n_mat, "bytes": n_bytes, "kept": n_skip, "why": why}
    if log is not None:
        log(f"[tensorfold] TF_EXL3_LANES: {n_mat} matrices ({n_bytes / 2**30:.2f} GiB) repacked to lanes, {n_skip} "
            f"kept on strips ({', '.join(f'{k}: {v}' for k, v in sorted(why.items())) or 'none'})", flush=True)
    return stats


def _reset_workspaces() -> None:
    import sys

    w = sys.modules.get("tensorfold.families.deepseek_v41.cuda.weights")
    ws = getattr(w, "_WORKSPACE", None) if w is not None else None
    if ws is not None:
        ws.held = None
