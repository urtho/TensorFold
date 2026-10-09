# DeepSeek-V4.1-Flash EXL3 on 2× DGX Spark — TensorFold port

Branch `dsv41-cuda` on top of upstream `ashhart/TensorFold` 0.6.2 (remote `upstream`; rebased 2026-10-02).
Target: `Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (mul1, 196 GiB, 39 shards) + Engram shards 47/48 of
`deepseek-ai/DeepSeek-V4.1-Flash`, TP=2 over CX7 on `aiai` (rank 0, 10.42.0.1) and `aiai2` (rank 1).

Baseline to beat (vLLM recipe on the same pair, DSpark k=3): decode 31.6 tok/s ×1, 23 tok/s serial,
aggregate 113.7 at ×6; prefill ~710 tok/s measured at 2K–32K (2026-10-01; ~1,000 quoted to 128k).
Per-token weight floor ≈ 3.7 GB/rank → ~17 ms, so serial ≈ 45–55 tok/s is the ceiling.
Where we are (2026-10-05, compose 16 × 614400, fp4 KV, carveout): serial ~37.8 tok/s, C1 98.0 (fp8 105.1; ~4% behind
since CUDA decode attention), hard prose 37.0, 16 clients 123.5 tok/s aggregate (fp8 128.0), PP8192 1,508,
100K prompt 2,286 tok/s; shared pool 6.52M tokens (fp8 2.38M).

Clean-room rule: the MiaAI-Lab vLLM recipe overlay and coolbho3k's `display_kv.c` (DeepSeek-v4.1-Flash-2x-DGX-Spark)
are **AGPL-3.0**. Read vLLM (Apache-2.0) and DeepSeek's reference (MIT) freely; take only ideas from AGPL code and
copy none of it into this Apache-2.0 tree.

## Phase 0 — groundwork

- [x] Clone upstream, branch `dsv41-cuda`, this TODO
- [x] NVIDIA container `nvcr.io/nvidia/pytorch:26.07-py3` on both nodes; install TensorFold; build kernels on sm_121
  (dev container `tf-dev` on aiai: repo at /tf, models at /models, Engram at /engram-src)
- [x] Toolchain smoke: `python -m tensorfold.cuda.exl3.inspect` — all 47,900 groups readable (`notes/dsv41/inspect.txt`)
- [x] Toolchain smoke: `tests/cuda/test_exl3_*` on GB10 — 114 passed, 54 skipped (need other checkpoints)
- [x] **DRM scanout carveout allocator, independent re-implementation of coolbho3k's display_kv idea**
  (`tensorfold/cuda/carveout.py`, pure ctypes, Apache-2.0, no AGPL code)
  - [x] DRM dumb buffer create/map, `cudaHostRegister(DEVICEMAP)`, torch tensor view, process-lifetime owner
  - [x] Opt-in via `TF_CARVEOUT=1`, size `TF_CARVEOUT_BYTES` (default 1792 MiB), card `TF_DRM_CARD`
  - [x] Probe CLI: `python -m tensorfold.cuda.carveout probe [--cuda]`
  - [x] Verify on GB10 with vLLM stopped: 1792 MiB, 0 MiB host RAM, round trip ok, 59 vs 122 GB/s (`notes/dsv41/BENCH.md`)
  - [x] Carveout holds the V4.1 compressed-KV pools (`TF_CARVEOUT=1`, largest source first; indexer keys stay in
        ordinary memory): prefill speed and long parity unchanged, admission credits the 1.75 GiB
- [x] Capacity: host reserve configurable (upstream 0.6.2: `TENSORFOLD_MEMORY_RESERVE_GIB`, >= 2 GiB; default max(4 GiB, 10 %)),
      carveout bytes counted as room outside MemAvailable
- [ ] Container flags doc: `--device /dev/dri/card0`, `nvidia_drm modeset=1 fbdev=0`, no display in use
      (`make up` already refuses with modeset off; the doc is still missing)

## Phase 1 — family skeleton + reference

- [x] `families/deepseek_v41`: MODEL_TYPES, config parsing (text_config), EXL3 check, `cuda_engine` stub
- [x] Architecture spec from vLLM reference: `notes/dsv41/ARCH.md` (open questions in §13)
- [x] Weight-streaming floor per rank: 482 µs/layer, ~49 tok/s serial ceiling (`tools/dsv41_bench_layer.py`, BENCH.md)
- [ ] Checkpoint map: EXL3 groups vs plain tensors per layer; per-rank split plan (experts by id, attention by head)
- [ ] Per-rank weight budget (must stay ≤ ~99.5 GiB/rank like vLLM)
- [x] Engram hashing/token map/bucket layout (`families/deepseek_v41/engram.py`; multipliers, primes, 99,092 ids verified)
- [x] Single-GPU layer-streaming reference forward, T ≤ 512 (`families/deepseek_v41/reference.py`)
- [x] Goldens: vLLM `prompt_logprobs` for 7 prompts (`tools/dsv41_golden.py`) → `notes/dsv41/golden.json`
- [x] Reference vs goldens: 94.2% top-1 / NLL 1.400 vs 1.383 on 365 tokens after dropping q per-head RMS
      (found with vLLM activation dumps: a private vLLM recipe checkout, eager mode,
      hook in `overlay/patch_memory_log.py`; `tools/dsv41_dump_diff.py`). vLLM itself is noisy on short
      prompts (eager vs CUDA graphs differ by up to 0.6 mean |Δlogprob|).
- [x] Chat template: the checkpoint's `chat_template.jinja` (V4.1 encoder port; DSML tool calls, reasoning_effort)

## Spec findings that change the plan (ARCH.md)

- Not an encoder/decoder: all layers causal. Only layers 2, 8, 14 (ratio 2) and 20 (ratio 1) own long-range KV;
  the rest read their source's cache. ~1.8 KB/token long-range KV (fp8) → 600k context ≈ 1.1 GB. KV is cheap.
- Top-k computed once per index source (2, 8, 14, 20, 24, 28, 32, 36) and reused by following layers.
- Below 512 tokens nothing is dropped: indexer/candidates can be deferred for first parity.
- Engram tables are 101 GB/layer fp8 in the original shards; 24 random 256 B rows/token/layer.
- The served checkpoint is the in-place abliterated variant (`ABLIT_META.json`, wo_b of layers 10–35).

## Phase 2 — serial TP=2 engine (go/no-go)

- [x] Embedding, hyper-connections (hc_mult 4, 20 Sinkhorn iters) on CUDA
- [x] Attention: q/kv low-rank, 64 heads × 512 shared KV, window 128 + sinks, RoPE/YaRN, grouped low-rank output
- [x] CSA2 compressors (ratios 2 / 1 per layer), compressed pools, cross-layer KV sharing (`kv_source_layer_ids`)
- [x] Indexer (32×128, top-512, bf16 keys) shared by `index_source_layer_ids`; rings for window/raw caches;
      long-context parity to 16K tokens (NLL equal to vLLM; BENCH.md)
- [x] Candidate blocks (layer 20, 2048×8): parity to 40K tokens (NLL 1.614 vs 1.613)
- [x] Position-keyed sampling (`tensorfold.cuda.sampling.sample_rows`), DSpark acceptance against keyed samples
- [x] MoE: sqrt-softplus router, noaux_tc top-6 of 384, shared expert; routed via `cuda/exl3/experts`
- [x] Engram layers 1/14: n-gram hashing, FP8 e4m3 rows, pread row store (page cache / NVMe)
- [ ] Mapped-table accounting in capacity
- [x] 2-rank split + NCCL rank-order reduction; lm_head (6-bit)
- [x] First serial TP=2 engine (`cuda/serial.py`, `tools/dsv41_serial_run.py`): eager PyTorch, 96.1 GiB/rank,
      365-token prefill 94.5% top-1 vs reference (NLL 1.420 vs 1.400), coherent greedy text; decode 7.3 tok/s,
      prefill 237 tok/s (2026-09-30)
- [ ] Engine vs reference agreement should be ~99%: layer-diff the engine against reference dumps
- [x] Fused HC pre/post Triton kernels (`cuda/hc.py`, tests/cuda/test_dsv41_hc.py) + one-row decode CUDA graph:
      decode 7.3 → 18.4 tok/s; profile 38 ms GPU/step: weights ~22, small torch ops ~7, NCCL 2.7, HC 1.7, host ~5
- [x] Fused MQA attention / table RoPE / RMSNorm kernels (`cuda/kernels.py`, tests/cuda/test_dsv41_kernels.py)
- [x] Concurrent Engram preads + GPU dequant: decode **26.9 tok/s** serial (vLLM serial 23) — BENCH.md
- [x] Native Engram reader (pthreads pread), 3 decode graphs with overlapped reads, argmax in graph,
      rank-order partial sums fused into HC post, fused router (fp16 mm → fp32, no TF32), attention chunk skip,
      Engram wkv split: **33.5 tok/s** serial (vLLM 23 serial / 31.6 DSpark)
- [x] Small linears in parallel on side streams (q / window KV / compressor, 8 wo_a slices, shared expert)
- [x] Shared expert folded into the grouped expert call (measured slower; off by default)
- [x] **Go/no-go**: serial 34–35 tok/s vs vLLM 23 at matching quality (target ≥ 40 still open)

## Phase 3 — drafting, long context, prefill

- [x] DSpark on CUDA (`cuda/dspark.py`): taps = entry streams of layers 37–39 (V4.1 semantics), Markov head,
      draft + (N+1)-row verify graphs. Acceptance equals vLLM's on the same prompts; ~1.3× vLLM tok/s (BENCH.md)
- [x] Exact verify: bit-identical to one-row decode (Triton router matmul + Engram gate); DSpark == serial tokens
- [x] Adaptive draft length (`DraftPolicy`, k = 0..N by expected tokens/ms): 1.39–1.49× vLLM on the 3 cases
- [x] Rank-deterministic draft policy (rank 0 decides k each round; ranks chose different k from own clocks → deadlock)
- [x] DSpark == serial tokens greedy and sampled (temperature 0.8) on all cases
- [x] Round costs from real tokens (capture zeros understated multi-row verify); policy picks k by true costs
- [x] Split Engram reads across ranks (bit-identical; 16 clients 114.9 -> 117.1 tok/s)
- [x] Shared expert folded into the grouped call: implemented, measured slower (35.0 vs 36.0 serial), off (`TF_FOLD_SHARED=1`)
- [x] Serial decode >= 40 tok/s (2026-10-09: 1-row window 22.0 ms = 45 tok/s with the Phase 9/10 defaults; was ~37.8 with x3ld + fast top-k + RoCE, GPU-bound 96 %): EXL3 decode GEMV toward ~240 GB/s (now ~215 large,
      110-180 small, 199 routed), fuse input rotation into the linear kernel, HC pre/post into one, router + route
  - [x] (measured Phase 9: -0.1..-0.5 ms, marginal, left off) x3ld expert-major order (dead slots last) + pd 3/4 rings on a 4..6 width instance: built, off until A/B
        (`TF_X3LD_ORDER=expert`, `TF_EXPERT_LOADS_CFG=4,4/4,3`; same Z; `TF_X3LD_PROBE=3` times the load path alone)
- [x] Decode graphs at narrow key widths on one graph memory pool (`TF_DSV41_WIDTHS`, default one 64K width;
      4.05 GiB for 32 graphs -> 0.27 GiB for 128): 16 clients at 16 × 614400 101.2 -> 122.5 tok/s (2026-10-03)
- [ ] Eager == graph checks per width; more narrow widths once graph driver memory allows (see Open)
- [x] Prefill 330 → 486 tok/s: 1,024-row chunks, EXL3 prompt GEMM for dense linears, prompt grouped expert kernel
      (`cuda/experts_prompt.cu`), bf16 prompt partials
- [x] Expert prompt kernel v2 (smem-staged activations, 2 member tiles/decode, 8 warps), 2,048-row chunks,
      last-row-only prompt logits: prefill **622 tok/s**
- [x] Prefill 1,252 tok/s steady (whole prompts 1,376 at 8K, 1,160 at 32K) vs vLLM ~710 (BENCH.md)
- [x] Long prompts to 600k: 4 sessions x 614,400 tokens admitted (fp8 caches), a 592,960-token prompt answered
- [x] KV format ~1.8 KB/token (fp8); carveout-backed pools
- [x] `--parallel N` shared rounds (see Phase 4)

## Phase 4 — serving

- [x] `cuda/engine.py` `Dsv41Engine`: `tensorfold serve --tp 2` on both nodes (rank 1 `follow()`s rank 0's
      requests: header + prompt over NCCL, rank 0's stop on the per-round agreement), warm-up before serving
- [x] OpenAI chat/completions through `cuda/server.App`: reasoning split, streaming, stop strings, seeded sampling,
      `draft: false` == drafted output, client disconnect stops within a round
- [x] DSML tool calls: V4.1 writes `<｜DSML｜ calls>` (space, no `tool_`); server + CUDA reply parsers read both forms
- [x] Launcher `tools/dsv41_serve2.sh [--port P] [--context N]` (dual-link NCCL env, Engram dir)
- [x] Prompt reuse: the live caches serve a prompt that extends them (common prefix >= 64 tokens, ring-safe);
      a follow-up turn of 8.5K tokens: 8,503 cached, 0.6 s instead of 8.0 s
- [x] Shared prompt-state pool `tensorfold/cuda/kv_pool.py` (model-agnostic: LCP match, LRU in a byte budget, rank-
      deterministic) + V4.1 adapter (`save_prefix` / `load_prefix`: per-position caches + rings' last window); budget
      from free memory (7 GiB ~ 2.35M tokens at context 40,960); switching conversations resumes in < 1 s, retries too
- [ ] Move GLM-5 / Nemotron-H snapshots onto `kv_pool` (see Open)
- [x] Concurrent decoding (`--parallel N`): stream slots, stream-aware decode graphs, MultiDecoder + shared Scheduler,
      cost-based draft allocation; 16 clients: 115.5 tok/s aggregate (vLLM recipe baseline 113.7 at x6); outputs ==
      sequential
- [x] Up to 32 rows a round + 15 MB decode rings a slot: 32 clients 143.3 tok/s aggregate
- [x] Batched DSpark drafting across streams (8 clients 82.5 -> 88.9 tok/s)
- [x] DSpark up to 5 drafts (reasoning 71.6, code 57.6 tok/s single stream)
- [x] Copy drafts, single stream (edit requests 80 -> 117 tok/s; after MiaAI-Lab's GLM recipe); [ ] in concurrent rounds
- [x] RoCE all-gather (`tensorfold.cuda.rdma`, from b12x RoCEnante via MiaAI-Lab 0006; exact): serial +1.7%;
      default since 28e8205 (see below); [ ] two rails
- [x] `--context` admission before any cache exists (both ranks' free memory; refuses with the largest that fits;
      default 40,960 shrinks to fit); expandable-segments allocator (prompt transients 3.6 -> 1.6 GiB at 38K)
- [x] Structured output: `response_format` json_schema / json_object, guided_choice / regex / grammar (xgrammar),
      masks on verify rows (DSpark drafts cut to the grammar's prefix), rank 1 compiles the same grammar
- [x] Compose project in place of the vLLM recipe: `deploy/dsv41-tp2` (image with deps baked in, `make swap-in` / `swap-out`, API key)
- [x] vLLM-compatible `/tokenize`; `/v1/completions` and `/tokenize` take token-id prompts
- [x] Draft allocator looks ahead (a stream's next 1..cap drafts at once): recipe bench C3 57 -> 112, C4 93 -> 132 tok/s
- [x] Hard prose at 4 × 614K 30.6 -> 33.2 tok/s: indexer scores only visible keys, Engram cached rows inline
      (`preadv2 RWF_NOWAIT`), hot pool helpers, no readahead on the tables

## Phase 5 — shared KV pool and kept prompts (2026-10-02 .. 10-03)

- [x] Shared KV pool (`--parallel > 1`): every stream's compressed entries and indexer keys in one arena (first-fit
      2048-aligned extents, design after MiaAI-Lab's GLM recipe), per-slot base table in the decode graphs, rank 0
      sends placements in ADMIT, NoRoom holds requests; concurrent == sequential
- [x] Kept prompts in the pool's free rows (extents outlive their stream, window rings in a bank of
      `TF_DSV41_KEPT_ENTRIES`, now 32); resume by takeover or copy; LRU eviction of whole extents (EVICT ops);
      the copy-out PrefixPool is off with the shared pool; 24-request suite == a no-keep server
- [x] Growth and yields: admission reserves prompt + `TF_DSV41_GROW_AHEAD` (4096); extents grow / move / evict kept
      states before each round (GROW/MOVE/EVICT ahead of ROUND); a stuck stream waits, the newest background stream
      yields and replays exactly; foreground requests ask `yield_for`; outputs == an unconstrained server
- [x] Decode graphs' driver memory (~40 MB a graph outside the allocator) charged in the pool size
      (`GRAPH_EXEC_BYTES`), `TF_DSV41_MEMLOG=1` per startup stage: pool 2.03M -> 3.08M tokens at 16 × 614400 (fp8)
      with 3.4-3.6 GiB left
- [x] Release after a long prompt only below `TF_DSV41_RELEASE_BELOW_GIB` (2.5) available (handing ~1 GiB back per
      prompt fragmented memory: 8K prompts 650-1000 -> 1550-1830 tok/s); glibc heap trimmed
- [x] Chunk-invariant prefill (`Linear.prompt_mode`, no chunk <= 32 rows): rows bit-identical whatever the chunking
      (chunk-test tails 1..1060, chunks 33..2048); kept prompts resume at any position their window covers
- [x] Exact bounded tail (layers 21-39 only over a prompt's last `tail_min` rows; idea from jayleaton's CED replay,
      exact form ours; `TF_DSV41_BOUNDED_TAIL=0` off): prefill 8K 1283 -> 1676, 16K 1279 -> 1995,
      32K 1249 -> 2165 tok/s; tail-test at 12K/30K/39K bit-equal
- [x] THP guards (`NUMPY_MADVISE_HUGEPAGE=0`, `MIMALLOC_ALLOW_THP=0`) in the compose environment
- [x] Defragmented starts: `make up` drops caches + `vm.compact_memory` on both nodes (fragmented host > 35 min load
      vs ~100 s clean; engine ready 59 s instead of ~88 s); `[boot]` timeline line
- [x] NVMe kept tier (`TF_DSV41_DISK`, `kvdisk.py` after Jay Leaton's sessdisk, MIT): evicted kept states spilled
      both ranks in step, restored on admission (32K: ~45 MB read in fp4 instead of ~30 s prefill), PERSIST on
      SIGTERM, key intersection at start; `KV_DISK_DIR` / `STOP_GRACE_S` 150 s in compose

## Phase 6 — speed, boot, serving hardening (2026-10-04)

- [x] Fast top-k: bounded radix-select indexer top-k for decode/verify rows (`topk.py`, after Jay Leaton's dtopk,
      our code; `TF_DSV41_FAST_TOPK=0` off): 16 clients 116.8 -> 123.0, serial 36.2 -> 36.8 tok/s; same entries
- [x] x3ld routed-expert load path (Jay Leaton's x3ld.cu, MIT; bit-identical Z), default on (4,2): 16 clients
      122.8 -> 128.2, 4 clients 79.8 -> 81.7, serial 36.9 -> 37.8 tok/s; bulk L2 prefetch sites opt-in
      (`TF_L2_PREFETCH=bulk`)
- [x] Fast boot: prepared per-rank weight folders (`TF_DSV41_PREPARED`, O_DIRECT parallel readers, per-chunk SHA-256,
      `make prepare`), cached Engram token map and calibration, lighter warm-up with `--parallel`
- [x] `make prebuild` compiles CUDA extensions (and the RoCE proxy, x3ld) before any model loads
      (compiling beside a loaded 4 × 600K model OOM-killed the server)
- [x] DSML parser: one lenient parser for streamed and whole replies (unclosed think, V4 spellings, cut invokes,
      `parallel_tool_calls=false`; adapted from Jay Leaton's dsml.py, MIT)
- [x] Tool grammar (`TF_DSV41_TOOL_GRAMMAR=required`, on in compose): required / named / strict tools as xgrammar's
      deepseek_v4_1 structural tag; streamed == whole, required == auto when auto calls first, alone == beside 3 streams
- [x] Health: `/health` reports fatal / stalled (`TF_STALL_S`), `TF_HEALTH=strict` answers 503
- [x] Watchdog: systemd user timer (`watchdog.sh`, `make watch-install`), lease / lock / stop marker, idle 1-token
      probe, heals with `WATCH_HEAL=1` (default 0: alert only)
- [x] Soak / stress / structured harnesses (`tools/dsv41_soak.py`, `dsv41_stress.py`, `dsv41_structured.py`,
      `make soak|stress|structured`, key only from the environment); `tools/dsv41_clients.py` N clients × T s
- [x] RoCE all-gathers by default (`TF_COMM=nccl` opts out; both ranks vote, any failure keeps NCCL): hard prose
      35.8 -> 37.7, C1 97.2 -> 97.7, C4 149.5 -> 154.0, 16 clients 127.3 -> 128.0 tok/s
- [x] Attribution (2740427): THIRD_PARTY_NOTICES.md, NOTICE and headers credit Jay Leaton, MiaAI-Lab GLM recipe,
      b12x RoCEnante, coolbho3k (ideas only), vLLM, DeepSeek, ExLlamaV3, xgrammar (FlashInfer added with mqa_fp4)
- [x] Draft PR ashhart/TensorFold#342 (head urtho:dsv41-cuda, draft) opened 2026-10-04; **on hold**: no pushes, no
      ready-for-review, no comments until the user pushes (e1baa98 and later are local only)

## Phase 7 — FP4 KV (2026-10-04 .. 10-05)

- [x] Torch port of V4.1's KV quantizers (NVFP4 / MXFP4 / FP8 ue8m0) against a NumPy oracle
- [x] `QRows` byte planes; `Fp4Rows` (NVFP4 entries 288 B, MXFP4 keys 68 B) + RoPE/quantize kernel, byte-equal on 1M rows
- [x] Attention and indexer read packed fp4 bit-equal to their bf16 dequant; prompt top-k ties to the lower index
- [x] `TF_DSV41_KV=bf16|fp8|fp4` (+ `IQ_FP4`, `SWA_FP8`, `COMP_BF16`); ranks agree on the format and knobs;
      fp8 1845 B a token, fp4 890
- [x] Invariant tools `--views-test`, `--resume-test`, `--tf-compare`, `--needle`, `--decode-bench`: all pass in fp4
- [x] Tie-keyed prompt top-k in one Triton pass (`_tie_pick`): fp4 prefill 100K 2085 vs fp8 2160, 128K 1993 vs 2133,
      600K 1270 vs 1343-1467 tok/s (was 256K 788 vs 1921)
- [x] Quality: teacher-forced vs bf16 over 24.5K positions: fp4 top-1 96.9%, dNLL +0.0016; fp8 97.8%, +0.0003;
      needles at 128K and 590K pass
- [x] **fp4 default** (11418ba, 2026-10-04): pool at 16 × 614400 2.38M -> 5.93M tokens (6.52M on the compose server);
      A/B fp4 / fp8: C1 98.0 / 105.1, hard 37.0 / 38.4, PP8192 1508 / 1596, 100K 2286 / 2428, 16 clients 123.5
- [x] CUDA decode attention for fp4 (`mqa_fp4.cu`, after FlashInfer's Cake DSv4.1 decode, Apache-2.0; default on,
      `TF_DSV41_CUDA_MQA=0` Triton): attention a step 0.33-0.36 ms (Triton fp4 1.56-1.60, fp8 0.86-0.90) at 4K-600K;
      verify R=4 0.51-0.57 ms; teacher-forced top-1 96.86%, dNLL +0.0018 (within noise)
- [x] Shared-tile fp4 indexer (`_index_scores_tile`, `TF_DSV41_SHARED_IK`, default on): MXFP4 key tile decoded once
      per stream for all its rows, tiles past visible keys skipped; torch.equal to the per-row kernel

## Phase 8 — adopted from the peer recipes (2026-10-08, notes/dsv41/PEERS.md)

Every item exact (replies' token sha unchanged; quick tier fingerprints equal to 9695a8b's), each behind a switch.
Jaybench on 9695a8b -> this tree (single requests, 3 reps, the same shas): code 82.80 -> 86.34, prose 45.30 -> 47.31,
structured 119.31 -> 124.85, c1 106.67 -> 109.60, hard 50.70 -> 53.01 tok/s.

- [x] Drafter Markov steps as kernels (`markov.py`, after bertholomus bd0024d; `TF_DSV41_MARKOV`): each rank scores its
      vocabulary half, one 16-byte-a-row gather a step, 256 cached fp32 bias rows (`markov_tokens.py`: 58% of a code +
      English sample); the fp32 bias equals cuBLAS's bit for bit (tests/test_dsv41_markov.py), the same drafts.
      Drafter graph 5.00 -> 3.32 ms; +66 MB a rank (charged in `engine.pool_tokens`)
- [x] RoCE gather (`TF_RDMA_HOST=register`): the region mmap'ed, locked and cudaHostRegister'ed (GB10 reaches
      cudaHostAlloc memory uncached), one system fence a block, the own shard copied before the wait, four peer loads
      in flight. Windows 1 / 2 / 4 / 6 rows vs pinned: 23.45 / 27.83 / 35.69 / 40.08 -> 23.08 / 27.35 / 35.53 / 40.01 ms;
      `TF_RDMA_MAX_KB=4352` (the head over RoCE at every width): no gain, stays 704
- [x] Router: the K slices summed inside `_route` (one launch less a layer), bf16 rows converted as loaded (no
      `.half()` copy); tests/test_dsv41_router.py
- [x] Prompt chunks without host syncs (`TF_DSV41_GROUP_LIST`): `serial.group_device` + `work_list` (after bertholomus
      v0.5 and jayleaton PR #17) replace bincount / `int(max)` / nonzero (3 syncs a layer); the grouped kernel walks the
      device-built (place, member group) list; tables equal group_members' (tests/test_dsv41_group_list.py).
      Prefill 8K 1726 -> 1749, 64K 2314 -> 2313 tok/s (chunk-prefill fingerprint unchanged)
- [x] Candidate-only reindex (`TF_DSV41_CAND_ONLY`, past the 16,384-entry pool): layers 24-36 score only layer 20's
      2048 blocks (`kernels.index_scores_cand`, `topk` MODE 3); scores and choice equal the masked full width's
      (tests/test_dsv41_fp4.py). Long tier vs 9695a8b: decode after 128K 29.90 -> 27.05 ms a step (32K 25.90 ->
      25.51), needles 6/6, tf-compare equal on the shared documents
- [x] Late Triton loads (`cuda/late_kernels.py`): a kernel first loaded after ready is printed and counted in /health
      (`late_kernel_loads`; bertholomus#1: such a load failed with CUDA 800 after 15.5 h); the serving-path warm-up
      (`engine._warm_serving`, `TF_DSV41_WARM_SERVING`): 1-32-row prompts, 33 / 48-row tails, a resumed kept prompt,
      streams decoding while others fill, a sampled reply
- [x] Loop guard (`call_gate.ThinkLoop`, `TF_LOOP_GUARD`, the request's "loop_guard"; after bertholomus dfbe519):
      3 dry 1,024-token windows (< 2% new 8-grams) close the thinking; tests/test_loop_guard.py
- [x] Kept-prompt trimming (`TF_DSV41_KEEP_SHRINK`, default on): prompts >= 128K also kept (exactly) at 3/4 and 7/8;
      room is made by dropping loose extents' longest states (`multi._trim`) before whole extents are evicted

## Phase 9 — performance items and reasoning loops (2026-10-08, out/perf/)

Designed by a 40-agent workflow (out/perf/design.json: 13 items, each with an exactness and a gain critic), built on
perf/* branches by a second one, merged in perf/integrate -> dsv41-cuda. Same-session A/B (out/perf/ab2-*: bbc5ee8 /
merged-off / on, alternating x2; the quick tier with every switch on gives bbc5ee8's fingerprints; reply shas equal).

- [x] Round index memo (`TF_DSV41_IDX_BASE`) + mHC partials and Sinkhorn on a side stream (`TF_DSV41_HC_SIDE`), on by
      default (c906e95): windows 1 / 2 / 4 / 6 rows 23.6-24.3 / 28.1 / 36.2 / 40.7 -> 23.15 / 27.3 / 35.0 / 39.9 ms;
      jaybench code 83.8 -> 85.8, prose 46.0 -> 47.0, structured 121.1 -> 124.2, c1 105.8 -> 107.8
- [x] Measured, left off: `TF_DSV41_HC_SPLIT` and `HC_SIDE_PART=0` (no gain), `TF_X3LD_ORDER=expert` (-0.1..-0.5 ms,
      marginal), `TF_DSV41_L2_DISCARD=po` (0) / `po,moe` (R=1 +0.4, R=6 -0.9: mixed), `TF_DSV41_RES_FOLD` (0),
      `TF_DSV41_MQA_FEW` 2 / 6 (worse than 1)
- [x] Copy drafts in concurrent rounds (`TF_MULTI_COPY`), on by default (26570cc): jaybench edit 107.5 -> 178.1, docs
      94.9 -> 115.1, code unchanged, the same reply shas; served soak + stress clean. 16-client run not done
- [x] Phase 9 workflow (out/perf/phase9.json, branches perf9/*, merged in perf9/integrate): every switch exact on GB10
      (GPU unit tests 335 pass; quick tier fingerprints equal with ROT_FUSE=1 PDL=1; reply shas equal in every arm).
      Same-session A/B, alternating x2 (out/perf/ab3-*), windows 1 / 6 rows and jaybench code / prose / structured / c1:
      base 22.9-23.1 / 39.7-39.8 ms, 82.5-85.3 / 45.4-46.7 / 122.2-122.7 / 101.6-107.3; no arm outside that noise:
  - `TF_DSV41_ROT_FUSE` (merge writes wo_a's rotated rows, wo_a's epilogue wo_b's; 80 rot_in launches a step fewer):
    22.9-23.6 / 40.0-40.3 ms. Left off; the q-norm part not built (the design's >= 0.1 ms bar not met)
  - `TF_DSV41_PDL` (Triton decode kernels as programmatic dependent launches): 22.8-22.9 / 39.8-40.0 ms. Left off
  - `TF_DSV41_COSTS=depth` (one stream's verify curve at 2K depth): still prices 5 / 6 rows at 43.8 / 46.3 ms against
    decode-bench 38 / 40 (it times the whole step_multi call); jaybench unchanged. Left off. Over-pricing stays open:
    time bare replays at depth, or correct online from measured rounds
  - [x] Serving warm-up trim, default since this round: GB10 audit boot "0 kernels only the full battery loads"; trim
    16.8 s vs 26.7 (boot 61.5 s vs 92), 15-minute soak (288 requests, 0 errors, 0 sha splits) + stress clean, no
    late kernel loads. `TF_DSV41_WARM_SERVING=full` keeps the old battery. Not run: a >128K request after a trim
    boot (warm_rare's shapes), encode (BOUNDED_TAIL) chunks in either battery
- [x] `TF_DSV41_SEND=one`, `TF_RDMA_TRACE`, norm + rot_in fusion, dense lanes, RoCE two rails, first-start PP dip:
      measured or built in Phase 10 below
- [x] Grammar-constrained streams draft (`TF_DSV41_DRAFT_GRAMMAR`, 6542b0d; on in the deployment): before, every
      response_format / tool-grammar reply verified one row a round, its reasoning included (accepted=0/0). Served:
      the same replies (12 verdicts' tokens and content equal; structured schemas PASS), ~55% of drafts kept, but at 6
      concurrent the aggregate is bandwidth-bound: per-stream 14-19 tok/s either way (median 87 -> 96 s); the gain is
      for 1-2 concurrent constrained requests
- [x] Reasoning loops (algolabel "smart" prompts, out/debug/smart-reasoning; tools/dsv41_reasoning_probe.py): the
      control prompt at temperature 0.1 looped to the 16K cap 4 / 4 (0.6: 1 / 4, 1.0 — the server default — 0 / 4;
      finished replies need 5-9K reasoning tokens). Loop guard under a grammar (f343663) cut 3 / 4 at 4-5K with valid
      JSON; numbers as one symbol (ec31852) also catches the 4th (a numbered 13-line cycle). Verdicts: no loops, but
      4 / 12 deliberate past max_tokens 4000 (empty content) at 6 concurrent; TF_THINK_RESERVE (thinking closed with
      room for the answer) turns those into answers


## Phase 10 — peer decode gap and the remaining TODO levers (2026-10-09, out/perf/)

Gap analysis vs bertholomus (6-row verify 36.8 vs our 39.7 ms): out/perf/gap-plan.md, gap.json (44-agent design
workflow), gap-attn.md (attention / indexer pass). Every lever exact, behind a switch, GPU unit tests on GB10, quick
tier fingerprints equal; A/Bs alternating, jaybench reply shas equal. Decode-bench windows sometimes read ~3 ms high
for a whole process regardless of switches: judged on jaybench and repeats.

- [x] On by default (2f88d69), confirmation A/B out/perf/ab8-* (3 reps): code 84.3-85.0 -> 85.9-86.6, structured
      122.4-123.3 -> 125.8-126.7, c1 105.8-106.3 -> 108.6-110.2, edit 178.8-180.4 -> 184.1-184.9, docs 116.5-116.9 ->
      119.3-120.0 tok/s; 4 / 6-row windows 35.0 / 39.7 -> 34.3 / 38.9-39.1 ms:
  - moe L2 discard in 4-32-row calls (`TF_DSV41_L2_DISCARD` default moe, `_ROWS` 4-32): the main window gain
  - candidate top-k bound (`TF_DSV41_CAND_BOUND`): 65536-key graphs scanned 16384 candidate lanes at any context
  - indexer q / weights beside the attention branches (`TF_DSV41_IDX_FORK`)
  - predicated gather copy-out (`TF_RDMA_COPY=pred`), parallel expert grouping (`TF_EXPERT_GROUP=par`), grouped
    drafter linears (`TF_DSV41_DRAFT_GROUP`): neutral alone, in the winning set
- [x] Calibration rows 1..8 best-of-16 (c4402f9): at best-of-4 the 4-6-row costs varied per process (row 4
      38.98-40.31 ms) enough to plan one draft fewer for a process's life (structured 112 vs 124 tok/s, same tokens);
      the server caches the curve per revision, so one bad timing held for a whole deployment
- [x] Built, exact, no gain, left off: `TF_X3LD_PDL`, `TF_DSV41_ROUTE_WARPS=1`, `TF_ROUTER_BE=16`,
      `TF_DSV41_COST_CURVE` (measured curve: none; flat: -15% prose/docs), `TF_DSV41_DRAFT_BORROW`, `TF_DSV41_SEND=one`
- [x] Rejected: `TF_DSV41_TOPK_DIGITS=11` (2048-bin histograms cost more than the pass saved: code -3%)
- [x] Dense lanes (`TF_EXL3_LANES`, jayleaton's DENSE_V3 layout, own extension tensorfold_exl3_lanes_v1; 1b5cc22
      on by default) + `TF_L2_ATTN_WQB_MB=8` (the next layer's wq_b prefix at the attention site, default 8): GB10
      A/B out/perf/ab10-* (3 reps, on top of the set above) code 86.9-87.3 -> 88.5-88.9, structured 126.3-126.9 ->
      129.2-129.4, c1 109.7-110.1 -> 111.9-112.3, docs 119.5-120.0 -> 121.4-122.1; lanes alone about half. The gate:
      an isolated 5-bit 4096x8192 layer ran 194 GB/s (3-bit 154) flat over 1-6 rows, so the time was in the k loop.
      Per call bit-identical, wo_a slices -10%, wq_b -4.5%, wo_b -2..-6.5%; 319 matrices (2.1 GiB) repacked at load,
      47 kept on strips (odd k steps a warp, 8-bit); quick-tier fingerprints equal. Never with PDL (refused). Served
      on the defaults (image 56916dd8): fingerprints equal, 15-minute soak 297 requests 0 errors 0 sha splits, stress
      ok. Boot 115.6 s on that first start ("caches" 21.2 s); the next image (read-ahead + two-group on) booted in
      76.3 s with "caches" 2.9 s, so the repack itself is cheap. Final served check on the complete defaults: fingerprints
      equal, soak 315 requests 0 errors 0 sha splits, stress ok
- [x] Measured, closed: narrow widths 16384 (no gain at 2K, ~1.2 GB); mHC row sharing (`_pre_partial` flat 1.5-1.7 ms
      over 1-6 rows); RoCE MTU (4096 already); 16 clients served: copy drafts on/off 127.3-127.6 / 127.9-128.1, fp4
      127.3-128.1 vs fp8 124.3-125.6 (the old fp8 lead was a stale baseline); norm + rot_in fusion (ROT_FUSE's 80
      launches unmeasurable, this targets less)
- [x] Gather trace (TF_RDMA_TRACE, out/perf/m8/trace.txt): per gather at R=1 2K stage 3.2, wait 12.5, copy 0.9 us;
      R=6 copy 1.7, ack 14 (R=1 5.9). Copy-out gate failed (matches its null A/B); two rails would save <= ~0.25 ms a
      6-row window / ~2% at 16 clients for an ABI + proxy change: deferred. The wait is protocol latency on both
      ranks (no skew)
- [x] Node trace (nsys, out/perf/m8/nsys_step.sqlite): rot_in -> linear_kernel gaps 0.19 us median (the replay
      floor): dense PDL (D2) closed
- [x] Weight-traffic audit (notes/dsv41/WEIGHT_TRAFFIC.md): dense 16 -> 17 rows +3.7 ms a round (dense +4.4; 17-32
      rows, concurrency only); a two-group pass needs > 255 registers (hot instances 213-235 + 64 accumulators):
      deferred to a kernel restructure. Expert member tiles > 16: concurrency only
- [x] Cold prefill (`TF_DSV41_PREFILL_PROF`): no first-start penalty left (6.3 vs 5.9-6.2 s at 8K); every prefill
      call waited on its first chunk's Engram reads (8K: 2 calls, ~0.8 s of 6.2 s TTFT). `TF_DSV41_PREFILL_AHEAD`
      reads the next call's first chunk during this call's last (waits -> 0 ms, reply shas equal); TTFT 6.14 -> 5.71
      s (8K) in one run, within the ~1-3 s run-to-run noise in another: off until a longer A/B
- [x] Two RoCE rails: closed (one port, see Open). Draft-cost curve over-pricing: closed (TF_DSV41_COST_CURVE with the
      measured curve gave no throughput change, out/perf/ab4-*)
- [x] Dense two-group pass for 17-32-row rounds (`TF_EXL3_LANES_TWO`, on by default): the lanes kernel decodes each
      tile once for two 16-row accumulator groups (lanes_v2; 4 warps 235-255 registers, no spills). GB10: exact
      (229 tests, fingerprints equal), per call -5..-12% vs per-pass lanes at 17-32 rows; windows 17 / 24 / 32 rows
      79.2 / 98.9 / 124.3 -> 77.7 / 97.4 / 121.8 ms, 16 rows unchanged (out/perf/two/); served 16 clients within
      noise (most rounds <= 16 rows)
- [x] Decode weight-traffic audit (done in Phase 10): distinguish unique weights, repeated load requests and actual DRAM reads;
      test dense 16→17-row passes, expert member tiles, mHC row sharing and prefetch effectiveness.
      Findings and measurement plan: [notes/dsv41/WEIGHT_TRAFFIC.md](notes/dsv41/WEIGHT_TRAFFIC.md) (2026-10-09).
- [x] RoCE GID index robustness (70abae0): no fixed index; NCCL (`NCCL_IB_ADDR_FAMILY=AF_INET`, RoCE v2, 10.42.0.0/15)
      and the RoCE all-gather find the IPv4 RoCE v2 GID per device (a reboot under a live QP moved aiai2's to 4)
- [x] FP4 prefill gap (67d6b36, bit-identical): row-tiled segment scores, one tie-keyed pick a pass, prompt entries
      decoded once a chunk: PP8192 1649 (fp8 1596), 100K 2581 (fp8 2428); dev 500K 1762 (fp8 1619)
- [x] fp4 indexer with many streams at full context (a5dd419, bit-identical): the shared-tile kernel ran a tile's
      rows in one program once tiles >= 192 (1.30-1.51x the per-row kernel at 16 or 8 streams x 1 row, 16K-614K keys);
      now a program scores one stream's run of rows: 16x1 / 8x8 1.03-1.05x per-row, 16 rows of one stream 0.61-0.75x,
      4x4 0.70-0.77x, 32 rows of 8 streams 0.72-0.76x (`tools/dsv41_ik_bench.py`); decode-bench 4 x 128K 36.89 ms
      (HEAD 37.24, per-row 37.36), same tokens
- [x] C1 ~4% behind fp8 in fp4: not there with the current code (2026-10-06). Engine at 4K, --dspark 3: fp4 step
      26.2-26.6 ms vs fp8 26.8-27.0, verify R=2/3/4 30.4/35.8/41.1 vs 31.1/36.3/41.5, drafter 5.0-5.1 both, ~70
      fewer launches a step; server, 6 fixed prompts greedy: fp4 101.2 / fp8 98.2 tok/s; bench_decode C1 (temperature
      0.2: trials 80-104 on one server) fp4 103.3 / fp8 101.6 and 93.0
- [x] FP4 tensor-core indexer: closed (not bit-exact against the reference order; the exactness rule)
- [x] Dev loop (notes/dsv41/DEV.md): one-load suites and tiers (`tools/dsv41_suite2.sh quick`: 168 s vs 499 s as
      fresh runs), prepared weights in dev runs (22-24 s loads), OOM guard (torch cap + per-run memwatch, start
      refusal), `make hot` / `make cold` (119 s, no image build)
- [x] Slow first start after `make image`: not reproduced (2026-10-06) in first starts after a cache-hit image, a
      pip-layer rebuild and 40 GB of fresh writes, nor in 4 restarts (C1 99.6-103, PP 1583-1670, 100K 2449-2578; no
      compaction, reclaim or dirty pages, clocks and governor unchanged). The one slow run (PP 1092, 100K 1715, hard
      31) overlapped an outside client's request in the server log: check `/health` requests_running before benching.
      Every start compiled ~84 Triton kernels at boot and ~12 mid-serving (the cache was in the container layer): now
      in CACHE_DIR (efb3911)
- [ ] Review-flagged test gaps: mqa_fp4 R=1 masking ([x] indexer: many streams at full visibility, n_keys 4096 / 4100)
- [x] FP4 quality (188bcdf, notes/dsv41/DEV.md): teacher-forced 26 docs top-1 vs bf16 fp4 96.75% / fp8 97.69% (sig.),
      dNLL +0.0012; MMLU 2280 paired fp4 85.75 / fp8 86.23 (p 0.11; gap = 16-token budget overruns); tool-eval
      111 / 117 of 138 (4 scenarios, sign p 0.125)
- [ ] Tool-eval fp4 vs fp8: 2-3 repeats each to settle the 6-point gap
- [x] Dev runs: expandable segments on by default in `dsv41_run2.sh` (4c9b2a3; quick tier fingerprints unchanged)
- [ ] Draft PR #342: description predates FP4, the CUDA kernels, the dev loop and the doorbell
- [x] Serial decode >= 40 tok/s (45 tok/s, see Phase 3)
- [x] Copy drafts in concurrent rounds (on by default, 26570cc; Phase 9): written behind `TF_MULTI_COPY` (CPU tests in
      tests/test_dsv41_multi_copy.py); GPU A/B pending (--decoder-test, --jaybench edit,docs,... 0 vs 1)
- [x] RoCE two rails: closed (Phase 10, tools/rdma_bench.py): NCCL over both HCAs = over one (20 KB 20.0 vs 20.8 us,
      640 KB 199 vs 205 us): the two "rails" are halves of one port; our one-rail gather beats both at every size
- [x] Triton kernels: `TRITON_CACHE_DIR` / `CUDA_CACHE_PATH` in the cache volume, shared by prebuild and every
      container (efb3911): boot 87 -> 44 s, none compiled on the first start after `make image`; a changed kernel
      compiles once on the first start that runs it (prebuild does not run kernels)
- [x] Exact-chunk TTFT mode (Phase 10): prompt chunks of 4096 rows (`TF_DSV41_PROMPT_CHUNK`, exact: fingerprints
      equal) prefill 32K at 2214 / 2021 vs 2225 / 2151 tok/s with 2048: no gain. The TTFT lever was the per-call Engram
      read wait: `TF_DSV41_PREFILL_AHEAD` (on by default, uncached 8K TTFT -0.45 s)
- [x] More narrow decode widths: no gain at 2K (Phase 10) (each width's graphs cost ~40 MB a graph of driver memory)
- [x] L2 prefetch sweep: unpaced bulk sites alone / in pairs / together within 0.4 ms of off; paced at 150 GB/s
      (`l2pace.cu`, after jayleaton's G14; all sites, joined at the step's end) the default: 1 / 2 / 4-row windows
      25.2 / 29.9 / 37.9 -> 24.1 / 28.5 / 36.5 ms, single requests code +3.6%, prose +5%, structured +2.6%, c1 +2.3%
- [x] In-engine measurement mode (`dsv41_serial_run.py --jaybench`, DEV.md) with the round split (`TF_ROUND_PROF`):
      window graph ~89% of a round, drafter ~10%, host + sync ~0.5 ms
- [x] Draft policy reset a request (`TF_DSV41_DRAFT_RESET`), prior 0.6 -> 0.8, relax 0.02 toward it in undrafted
      rounds (a stream that stopped drafting never drafted again): prose 41.4 -> 42.3 (single runs, within ~2.5 tok/s noise), hard 38.9 -> 47.4 single
- [x] Deployed (6c46ef7, make image): jaybench code / prose / structured / greedy c1_hard 79.7 / 41.5 / 114.0 / 41.7 ->
      81.2 / 44.7 / 117.8 / 45.8 (jayleaton G19 84.7 / 46.8 / 121.3), same replies, identical round counts across reps;
      C1 100.2 -> 106.0, C2 139.3, C3 154.7 -> 164.2, C4 154.2 -> 152.9, hard 38.3 -> 49.4, PP8192 1658 -> 1635,
      100K prefill 2569 -> 2497 (decode 44.0), 16 clients 122.6 -> 124.4 (first start; PP / 100K prefill untouched)
- [x] (Phase 10: the measured cost curve gave no throughput change, ab4; 16 clients prior 0.6 / 0.8 / 0.9 and relax 0 /
      0.02 all 127.2-129.9 tok/s, out/perf/served17.txt: defaults kept) Draft policy: 6-row windows (48.5 ms) vs 4 rows (37.9): a per-k cost model with the drafter's 5 ms and the
      relax / prior under concurrent load (16 clients) not swept
- [x] (ours now 22.0; Phase 10: priorities ignored in graphs, SEND=one null, copy-out pred in the default set, the
      lean gather's remaining stage cost is the system fence) jayleaton's 1-row window 21.7 ms vs ours 24.1 with pacing: his remaining levers (BRANCHES side priority,
      ROCE_FAST, plan link) not ported. Landed off, A/B pending (DEV.md): the gather phase trace (`TF_RDMA_TRACE`,
      the go / no-go for a lean gather: >= 2 us kernel overhead at R=1 or >= 5 us copy-out at R=6) and the
      one-exchange ROUND send (`TF_DSV41_SEND=one`, his plan link's idea)
- [ ] Engine vs reference layer-diff (94.5% -> ~99% agreement)
- [ ] Eager == graph per width
- [ ] Mapped-table accounting in capacity
- [ ] Checkpoint map / per-rank weight budget (Phase 1)
- [ ] Container flags doc for the carveout (Phase 0)
- [ ] `kv_pool` for GLM-5 / Nemotron-H
- [ ] Housekeeping: agent worktrees (`attribution`, `item4-kvdisk`, `item5-serving` under `.claude/worktrees`),
      stray `uv.lock`, private path in `tools/dsv41_vllm_dump_patch.py`
- [ ] Watchdog: decide `WATCH_HEAL=1` (cluster heal tests in deploy README) and user linger
- [x] Rank 1 idled at ~95% GPU util spinning in the collective: CPU idle doorbell (2248e36, `TF_IDLE_DOORBELL`):
      0-1% / ~11 W idle, outputs and latency unchanged
- [x] Server MemAvailable after warm-up was 2.8-2.9 GiB: TF_DSV41_RESERVE_GIB 2.5 -> 3 (3.86 / 3.91 GiB; pool 6.52M -> 6.41M)
- [ ] Old `tf-dev-old-*` containers on both nodes can go (keep the `tf-dev-snapshot:*` images: tf-dev runs on them)
- [x] (TF_DSV41_IDX_BASE on by default, Phase 9) Global top-k indices rebuilt every compressed layer in the multi-stream path (`serial.py` attention's static
      branch: idx + the stream's entry base, 6 small ops a layer, ~0.4 ms a step): built once per (index source, kv
      source) with the round's ring / entry bases once a graph piece behind `TF_DSV41_IDX_BASE=1` (default 0 until
      the GPU A/B: quick-tier fingerprints, decode-bench R=1..6, jaybench sha). Passing the base into `K.mqa`
      (idea: vLLM #57659) dropped: ~0.05 ms more for an ext bump
- [x] (Phase 10: none left; per-call Engram read wait fixed by PREFILL_AHEAD) First-start PP8192 dip still seen with zero Triton compiles (1473-1523, then 1830-1856 a minute later): cause open
- [ ] Server MemAvailable 2.7-2.9 GiB under load (3.8 idle) with RESERVE 3: decide whether the 3 GiB floor applies
      under load
- [x] 16 clients fp4 122.6 vs fp8 128.0: now fp4 127.3-128.1 vs fp8 124.3-125.6 (Phase 10; stale baseline)
- [ ] Triton cache in the serve volume never evicts (98 entries, 42 MB) and is root-owned under ~/.cache
- [ ] Review test gaps: shared-tile many-stream test never mixes visible bounds within a run at capacities 4096/4100;
      suite2.sh baseline copy handles only `--suite-baseline PATH` under /tf/out

## Ops notes

- aiai / aiai2 are dev boxes until the port is done: `make down` / `up` / `restart` / `image` of the compose server
  (/home/docker/ai/vllm-serve/tensorfold-dsv41-TP2 on aiai; the worker gets the directory via `make sync`) is
  normal work. One GPU job at a time; `make down` before dev runs.
- GB10 memory: fragmented host memory makes loads and prefill slow (compact before start; watch `compact_stall`);
  each decode graph costs ~40 MB of driver memory outside torch's allocator (check MemAvailable, not reserved).
- vLLM service: a private vLLM recipe checkout (`make`/`recipe/start.sh stop|start`).
  OK to stop for GPU work; restart when done.
- Unified memory: a GB10 OOM once wedged both nodes (2026-09-11). Cap side jobs with
  `systemd-run --scope -p MemoryMax=…`; `--oom-score-adj 1000` on our containers.
