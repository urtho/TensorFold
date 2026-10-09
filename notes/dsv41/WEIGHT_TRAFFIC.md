# Decode weight-traffic audit — dual DGX Spark (SM121)

2026-10-09. Source review at `a9e2457`, including the working tree's calibration experiments.
These are code findings and proposed measurements, not new cluster benchmark results.

## Question

How much decode weight traffic is redundant, and how much can we remove without changing outputs?

The existing ~3.7–4.2 GB/rank weight budget is a useful rough estimate, but does not establish actual DRAM
traffic or an unavoidable latency floor. Distinguish:

1. Unique packed weight bytes needed by the selected rows and experts.
2. Bytes requested by load instructions, including overlapping lanes and repeated row tiles.
3. Bytes actually fetched from DRAM after cache reuse and prefetching.
4. Bytes per committed output token, including work on rejected speculative rows.

Fewer load instructions need not mean proportionally fewer DRAM bytes. Likewise, total kernel self-times
cannot be added into a round latency when kernels overlap on side streams.

## Concrete findings

### Dense EXL3: overlapping lane loads

[`linear.cu`](../../src/tensorfold/cuda/exl3/linear.cu), `load_step`, and
[`decode.cuh`](../../src/tensorfold/cuda/exl3/decode.cuh), `tile_words` / `lane_words`:

- A 5-bit tile (`K2=10`) contains 40 packed 32-bit words.
- Each of 32 lanes requests 3 words spanning its trellis windows: 96 word requests for 40 unique words.
- This is 2.4× requested words, **not evidence of 2.4× DRAM traffic**. Coalescing and caches can serve overlaps.
- The 1-bit and 2-bit cases already have a cooperative load/shuffle path.

Experiment: extend cooperative packed-word loading or prototype the proposed dense lane layout, then distribute
words through shuffles/shared memory. Preserve decoded values and reduction order. Measure load overhead,
cache traffic, registers and spills alongside wall time, with the existing L2 pacing enabled.

### Dense EXL3: repeated passes above 16 rows

`linear_kernel` in `linear.cu` loops over rows in passes of 16; each pass runs the weight load/decode loop again.
A 17–32-row round therefore scans the dense weights twice at the instruction level. Actual DRAM rereads depend
on whether the weights remain cached.

Experiment: reuse a loaded weight tile across two row groups. The tradeoff is more accumulators and register
pressure; preserve each row's MMA and reduction order. Test 16 versus 17 rows explicitly. This primarily targets
concurrent decode or long copy windows, not the usual 1–6-row DSpark window.

### Routed experts: reuse already exists, with a 16-member boundary

[`experts.py`](../../src/tensorfold/cuda/exl3/experts.py), `routed`, groups picks into unique experts and member lists.
[`x3ld.cu`](../../src/tensorfold/cuda/exl3/x3ld.cu), `ld_kernel`, processes 16 members per tile.

- Six rows selecting the same expert share that expert's packed weights within one member tile.
- Estimate compulsory expert bytes from **unique selected experts**, not `rows × top-k`.
- An expert with more than 16 members gets another member tile that independently loads its weights.
- Empty member tiles return before loading the expert trellis; allocated grid size is not weight traffic.

Experiment: share weights across member tiles or improve their cache locality through scheduling. This only
helps when expert member counts warrant it. Expert-major ordering has already shown marginal gains in earlier
A/Bs; do not assume a large benefit from changing order alone.

### mHC: the same matrix requested by every decode row

[`hc.py`](../../src/tensorfold/families/deepseek_v41/cuda/hc.py), `_pre_partial`, launches per row and K block.
Each row requests the same `24 × 20480` FP32 mix matrix (~1.97 MB). Across 80 sublayer boundaries, the nominal
matrix requests are ~157 MB for one row and ~944 MB for six rows, versus ~157 MB of unique matrices. These are
approximate byte counts; first-layer special handling and cache hits must be accounted for in measurements.

Experiment: process several rows with one loaded matrix tile while retaining each row's current arithmetic order.
The prompt path already shares tiles using `tl.dot`, but directly substituting that path can change numerics.
Check whether existing side-stream overlap leaves any of this work on the critical path.

## Other suspects

- **Prefetch effectiveness.** The saved profile reports ~25.7 MiB per layer across the router/shared-expert,
  next-attention and wo_a prefetch sites. Early eviction or prefetch arriving after the consumer could add traffic.
  Existing timing A/Bs favor pacing, so compare DRAM bytes and consumer cache hits before changing it. A prefetch
  followed by a cache-hit demand load is successful reuse, not two DRAM reads.
- **Rejected drafts.** Expert weights unique to rejected rows contribute no committed output. Draft policy should
  minimize time/bytes per committed token, not merely verification-round cost. All planned rows currently run
  through the target before acceptance is known; eliminating that waste requires a scheduling tradeoff.
- **Replication across ranks.** Some small projections and mHC weights are replicated, while large attention,
  expert and vocabulary matrices are sharded. Removing replication adds communication and may increase latency;
  count it separately from redundant rereads within a rank.
- **Reuse across rounds.** A later token may select the same expert, but weights must survive the intervening
  layers and other traffic to hit cache. Their presence in unified system memory does not eliminate the GPU's
  LPDDR reads. Do not assume a cross-round expert cache will help without reuse-distance measurements.

## Measurement plan

Use fixed real-token rounds, with identical routing and draft plans between alternatives. Keep adaptive-policy
comparisons separate from kernel comparisons. Include 1, 6, 16, 17 and 32 rows, both consecutive tokens from one
stream and independent streams, at representative short and long contexts.

For each rank and weight family, record:

- Unique packed matrix bytes, plus scale/rotation metadata separately.
- Unique experts per layer and their member-count distribution; expected scans scale with `ceil(members / 16)`.
- Dense scan count (`ceil(rows / 16)` for the current decode kernel).
- Measured DRAM reads, L2 traffic/hit behavior, and scratch/activation traffic separately where attribution permits.
- Round wall time, committed tokens, acceptance, and bytes per committed token.

Start with isolated kernels for attribution, then confirm in the full engine: isolation changes cache residency
and overlap. A profiling replay must reproduce routing and state; counter collection can perturb timings, so
report ordinary unprofiled A/B wall times separately.

Run prefetch on/off controls and examine the **16→17-row discontinuity**. For dense loads, determine whether
overlapping lane requests mostly cost instructions/cache transactions or actually amplify DRAM traffic.

Priorities:

1. Single-request decode: overlapping dense loads and prefetch effectiveness.
2. Concurrent throughput: repeated dense passes and experts with more than 16 members.
3. mHC row sharing if counters and the critical path show an opportunity.

Require existing exactness checks (quick-tier fingerprints and serial/drafted reply equality); use the long tier
when a changed layout or dispatch also affects prefill or long-context paths. Accept improvements on end-to-end
wall time, not byte reductions alone.

## Existing evidence

- [Building-block benchmarks](BENCH.md)
- [Dev measurement workflow](DEV.md)
- [Phase 9 design and critiques](../../out/perf/design.json)
- [Recent calibration and send A/B](../../out/perf/ab4-summary.txt)
- [Saved pre-Phase-9 kernel profile](../../out/perf/prof-bbc5ee8.txt)

The `out/` artifacts are local and may not accompany a published checkout.
