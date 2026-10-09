"""TF_EXL3_LANES microbenchmark: one synthetic EXL3 layer, linear.cu (strips) vs lanes.cu (lanes), the same plan;
microseconds a call and weight GB/s for each row count, and whether the outputs are bit-identical. Rows 17-32 also
time lanes' two-group kernel (TF_EXL3_LANES_TWO: each weight tile decoded once for both 16-row groups) against its
per-pass kernel ("two" column, its change vs per-pass lanes).

    python tools/exl3_lanes_bench.py --k 4096 --n 8192 --bits 5 3 --rows 1 2 6 16
    python tools/exl3_lanes_bench.py --dsv41            # DSV4.1's per-rank dense shapes at 5 bits
    python tools/exl3_lanes_bench.py --dsv41 --rows 16 17 24 32      # per-pass vs two-group lanes

Cold weights: a scratch buffer larger than L2 is rewritten between calls (``--hot`` skips it)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfold.cuda.exl3 import format as fmt
from tensorfold.cuda.exl3 import lanes, linear

DSV41 = [("wq_a|wkv", 5120, 1536 + 512), ("wq_b", 1280, 16384), ("wo_a(slice)", 4096, 1024), ("wo_b", 4096, 5120),
         ("shared w1", 5120, 1152), ("shared w2", 1152, 5120)]


def layer_pair(k: int, n: int, bits: float):
    rng = np.random.default_rng(k * 7 + n)
    trellis = torch.from_numpy(rng.integers(-2**15, 2**15, size=(k // 16, n // 16, fmt.tile_words(bits))).astype(np.int16))
    suh = torch.from_numpy((rng.standard_normal(k) * 0.05).astype(np.float16))
    svh = torch.from_numpy((rng.standard_normal(n) * 0.05).astype(np.float16))
    a = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    c = linear.Exl3Linear.from_tensors(trellis, suh, svh, "mul1")
    lanes.convert(c, log=None)
    return a, c


def time_us(fn, reps: int, flush: torch.Tensor | None) -> float:
    ts = []
    for _ in range(reps):
        if flush is not None:
            flush.add_(1)
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return float(np.median(ts))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--bits", type=float, nargs="+", default=[5, 3])
    ap.add_argument("--rows", type=int, nargs="+", default=[1, 2, 6, 16])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--dsv41", action="store_true")
    args = ap.parse_args()
    flush = None if args.hot else torch.empty((256 << 20) // 4, dtype=torch.float32, device="cuda")
    shapes = [(nm, k, n, 5) for nm, k, n in DSV41] if args.dsv41 else \
        [(f"{args.k}x{args.n}", args.k, args.n, b) for b in args.bits]
    print(f"{torch.cuda.get_device_name()}  {'hot' if args.hot else 'cold'} weights, median of {args.reps}")
    for nm, k, n, bits in shapes:
        a, c = layer_pair(k, n, bits)
        why = lanes.why_not(a.k2, a.k, a.split)
        if why is not None:
            print(f"{nm:12s} {bits:g}b split {a.split}: strips only ({why})")
            continue
        mb = a.nbytes() / 1e6
        ext = lanes._ext()
        was = ext.get_two()
        for m in args.rows:
            x = torch.randn((m, k), device="cuda").half()
            ext.set_two(False)
            ya, yc = a(x), c(x)
            same = torch.equal(ya, yc)
            ta = time_us(lambda a=a, x=x, ya=ya: a(x, out=ya), args.reps, flush)
            tc = time_us(lambda c=c, x=x, yc=yc: c(x, out=yc), args.reps, flush)
            two = ""
            if 16 < m <= 32:
                ext.set_two(True)
                yt = c(x)
                same = same and torch.equal(yt, ya)
                tt = time_us(lambda c=c, x=x, yt=yt: c(x, out=yt), args.reps, flush)
                two = f"  two {tt:7.1f} us {mb / tt * 1e3:6.1f} GB/s ({(tt - tc) / tc * 100:+5.1f}% vs lanes)"
            ext.set_two(was)
            print(f"{nm:12s} {bits:g}b split {a.split} rows {m:3d}: strips {ta:7.1f} us {mb / ta * 1e3:6.1f} GB/s"
                  f"  lanes {tc:7.1f} us {mb / tc * 1e3:6.1f} GB/s  ({(tc - ta) / ta * 100:+5.1f}%){two}"
                  f"  bit-identical {same}")


if __name__ == "__main__":
    main()
