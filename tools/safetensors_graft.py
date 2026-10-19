#!/usr/bin/env python3
"""Graft a sidecar's tensors into a copy of a safetensors checkpoint, standard library only (no torch).

  python3 -I tools/safetensors_graft.py --src STOCK_DIR --sidecar SIDECAR.safetensors --dst NEW_DIR [--meta META.json]

Every sidecar tensor must already exist in the checkpoint with the same dtype and shape; it replaces that tensor's
bytes. Shards holding none of them are hard-linked from --src (copied if --dst is on another filesystem); shards
holding some are rewritten with the replaced bytes and a recomputed header; every other file is copied. --src is never
written. After writing, each grafted tensor is read back from --dst and compared byte for byte with the sidecar.
Used for drowzeys' Keys overlays (attn.wo_b of layers 10-35) on DeepSeek-V4.1-Flash EXL3 packs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from collections import defaultdict
from pathlib import Path


def header(path: Path) -> tuple[dict, int]:
    """The safetensors header (dict) and the byte offset where tensor data starts."""

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(path: Path, name: str) -> bytes:
    h, base = header(path)
    a, b = h[name]["data_offsets"]
    with open(path, "rb") as f:
        f.seek(base + a)
        return f.read(b - a)


def rewrite(src: Path, dst: Path, graft: dict[str, bytes], side_h: dict) -> None:
    """dst = src with the tensors in ``graft`` replaced; offsets recomputed in src's tensor order."""

    h, base = header(src)
    meta = h.pop("__metadata__", None)
    order = sorted(h, key=lambda k: h[k]["data_offsets"][0])
    out, pos = {}, 0
    for k in order:
        e = dict(h[k])
        if k in graft:
            s = side_h[k]
            if (s["dtype"], s["shape"]) != (e["dtype"], e["shape"]):
                raise SystemExit(f"{k}: sidecar {s['dtype']} {s['shape']} != checkpoint {e['dtype']} {e['shape']}")
            size = len(graft[k])
        else:
            size = e["data_offsets"][1] - e["data_offsets"][0]
        e["data_offsets"] = [pos, pos + size]
        out[k] = e
        pos += size
    if meta is not None:
        out = {"__metadata__": meta, **out}
    raw = json.dumps(out, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)                       # the data starts 8-byte aligned, as the writers do
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        fo.write(struct.pack("<Q", len(raw)))
        fo.write(raw)
        for k in order:
            if k in graft:
                fo.write(graft[k])
            else:
                a, b = h[k]["data_offsets"]
                fi.seek(base + a)
                left = b - a
                while left:
                    chunk = fi.read(min(left, 64 << 20))
                    fo.write(chunk)
                    left -= len(chunk)
    os.chmod(tmp, 0o644)
    tmp.replace(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--sidecar", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--meta", type=Path, help="a JSON file copied into --dst as ABLIT_META.json")
    a = ap.parse_args()
    if a.src.resolve() == a.dst.resolve():
        raise SystemExit("refusing to write into --src")
    side_h, _ = header(a.sidecar)
    side_h.pop("__metadata__", None)
    graft_keys = set(side_h)
    wm = json.loads((a.src / "model.safetensors.index.json").read_text())["weight_map"]
    missing = sorted(graft_keys - set(wm))
    if missing:
        raise SystemExit(f"{len(missing)} sidecar tensors are not in the checkpoint, e.g. {missing[:4]}")
    by_shard = defaultdict(set)
    for k in graft_keys:
        by_shard[wm[k]].add(k)

    a.dst.mkdir(parents=True, exist_ok=True)
    linked = copied = 0
    for p in sorted(a.src.iterdir()):
        d = a.dst / p.name
        if p.name.startswith(".") or d.exists():
            continue
        if p.is_dir():
            shutil.copytree(p, d)
        elif p.name.startswith("model-") and p.name.endswith(".safetensors") and p.name not in by_shard:
            try:
                os.link(p, d)
                linked += 1
            except OSError:
                shutil.copy2(p, d)
                copied += 1
        elif not p.name.endswith(".safetensors"):
            shutil.copy2(p, d)
    for shard, keys in sorted(by_shard.items()):
        graft = {k: read_tensor(a.sidecar, k) for k in keys}
        rewrite(a.src / shard, a.dst / shard, graft, side_h)
        print(f"rewrote {shard}: {len(keys)} tensors", flush=True)
    for k in sorted(graft_keys):                        # read back
        if read_tensor(a.dst / wm[k], k) != read_tensor(a.sidecar, k):
            raise SystemExit(f"read-back mismatch: {k}")
    if a.meta:
        shutil.copy2(a.meta, a.dst / "ABLIT_META.json")
    print(f"done: {len(graft_keys)} tensors grafted into {len(by_shard)} shards, {linked} shards linked, "
          f"{copied} copied, read-back equal", flush=True)


if __name__ == "__main__":
    main()
