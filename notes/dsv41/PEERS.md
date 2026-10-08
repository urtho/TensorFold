# Peers: other DeepSeek-V4.1-Flash TP=2 recipes on two DGX Sparks

These are the GitHub repos serving DeepSeek-V4.1-Flash with EXL3 routed experts at TP=2 across two GB10s. The list was compiled on 2026-10-08.
`tools/dsv41_peers.py` reports what changed since its last run: new commits per watched branch, new branches, stars,
README edits, plus repos found by search that neither list names. The lists are in `peers.json`, and the state is in
`~/.local/state/tensorfold-peers/`.

    python3 tools/dsv41_peers.py [--dry-run] [--out digest.md]

All numbers below are the authors' own, on their own benchmarks; they are not comparable one to one. Ours on the same
hardware (2026-10-04, fp4 KV, 16 × 614K, 6.5M pool tokens): C1 98 tok/s, PP8192 1508, 100K prompt 2286 tok/s.

## TensorFold engines (watched)

| repo | engine source | weights | claims | notes |
|---|---|---|---|---|
| [jayleaton/deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark) | upstream TensorFold submodule (`vendor/TensorFold`) + its own family | Mia 2.9 bpw, dealignai uncensored 2.9 | 1.8-1.9× code, 2.5× multi-stream, 1.7-1.95× prefill vs the Mia vLLM kit | G14 (L2 pacing) and G19 releases; we took the bounded-tail idea, `l2pace.cu` and `dsv41_soak.py` from here (NOTICE) |
| [bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) | [bertholomus/TensorFold@deepseek-v41-tp2](https://github.com/bertholomus/TensorFold/tree/deepseek-v41-tp2) (base 0.6.6) | Mia 2.9 bpw | v0.4: code C1 101 tok/s, 4 streams 112.5; v0.5: 1M context | the most active; Capicua25x sends PRs (their fork is ignored); v0.5.1 fixes reasoning loops |
| [sfxnz/DeepSeek-V4.1-Flash-EXL3-TensorFold-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-TensorFold-2x-DGX-Spark) | [sfxnz/TensorFold](https://github.com/sfxnz/TensorFold), branches `dsv41-recipe-engine1..4` (one branch per engine drop) | own 2.0 bpw pack (sfxnz/DeepSeek-V4.1-Flash-EXL3) | c=1 prose 56, structured 98 tok/s; `--parallel 1..4` lane decoder (b86514a) | the upstream PRs #390/#391 were closed with ours; a strict benchmark harness (frozen `bench_decode.py`, gated boots) |
| [soumyarupsarkar/deepseek-v4.1-flash-2X-GB10-TensorFold](https://github.com/soumyarupsarkar/deepseek-v4.1-flash-2X-GB10-TensorFold) | built on Bertholomus's engine | Mia 2.9 + drowzeys abliteration | 8.6M resident KV tokens, 32 sessions; C1 code 94.5, C16/C32 210/272 tok/s aggregate | larger pool than our 6.5M: how? |
| [ZackO2o/DeepSeek-V4.1-Flash-TensorFold-DGXSpark-TP2](https://github.com/ZackO2o/DeepSeek-V4.1-Flash-TensorFold-DGXSpark-TP2) | jayleaton's recipe | Mia / dealignai 2.9 | code 81.8 tok/s C1, 4-way 150; an 855K-token request served | measured methodology (fit ladder, pool-vs-context failure mode), EN + 中文 |

## vLLM recipes, EXL3 (watched)

| repo | weights | claims | notes |
|---|---|---|---|
| [MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) | Mia-AiLab 2.9 bpw | code 39-43, C1 31-40 tok/s (two configs); 32K prefill 1138 | the reference kit (★258); our vLLM compose is built from it; quiet since 09-19 |
| [sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark](https://github.com/sfxnz/DeepSeek-V4.1-Flash-EXL3-vLLM-2x-DGX-Spark) | own 2.0 bpw | greedy C1 39.6, prose c=1 50.4 tok/s (round 36) | sibling of the TensorFold one; quiet since 09-27 |
| [coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark](https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark) | own 3 bpw MUL1 + DSpark drafter | 25-40 C1, 32K prefill ~1078 | TP2/DCP2, compact FP4 KV, vision, 1M-context index scoring; quiet since 09-28 |
| [0xSero/DeepSeek-V4.1-Flash-Two-Sparks](https://github.com/0xSero/DeepSeek-V4.1-Flash-Two-Sparks) | own 2.77 bpw | code C1 39-41, prefill 8K/32K ~2050 | 262K context, 2M KV, vision, MTP, bundled with Pi |
| [vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe](https://github.com/vcruz305/DeepSeek-V4.1-Flash-EXL3-DGX-Spark-recipe) | SAGE 1.59 / 3.30 / 4.75 bpw | TP1 prose 30, code 33 | TP2 path marked unverified on hardware (the author has one Spark); TP4 TP-MoE work |

## Seen, not watched (`ignore` in peers.json)

- Capicua25x/deepseek-v4.1-tensorfold-tp2-2xgb10 (a fork of the bertholomus recipe), jvr0x/… (a fork of vcruz305),
  drowzeys/TensorFold (GLM branches only).
- drowzeys/keys-…-Abliterated-Mia-2x-Spark-EXL3: a one-commit weight-overlay helper.
- MiaAI-Lab/DeepSeek-v4.1-Flash-DGX-Sparks: SGLang native weights on 3-4 Sparks (TP3/TP4), not TP2.
- Snail3D/dsv41-vision-2xspark, ajensenwaud/recursant-two-sparks-deepseekv4.1-flash: stale since mid-September.
- ivanusto/dsv41-flash-vllm030-2x-gb10: REAP-256E on stock vLLM 0.30, not EXL3.

## Leads to check against ours

Adopted 2026-10-08 (TODO.md Phase 8): the drafter Markov kernels, the RoCE gather's registered memory, the
prompt-chunk work list, candidate-only reindex, late-load reporting and the serving warm-up, the loop guard and
kept-prompt trimming. Still open: the verify-window levers (L2 discard, mHC split, merge / norm fusions).
The round glue (bd0024d rounds.py `_ix`, `TF_DS_ROUND_GLUE`) and the mHC split / defer / dots (`TF_DS_HC_SPLIT`,
`HC_DEFER`, `HC_DOTS`) are written here behind `TF_DSV41_IDX_BASE` and `TF_DSV41_HC_SPLIT` / `HC_SIDE` /
`HC_SIDE_PART`, all off until measured.

- TF32 (bertholomus 741a507, recipe #6): NGC PyTorch containers set `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, so fp32
  cuBLAS GEMMs (router logits, indexer weights, mHC mixes) run in TF32. Seeded replies then differ from the reference,
  and one reported long agentic turn looped to the 32K cap. Our serving container has the variable set and
  `allow_tf32` reads True (checked 2026-10-08). For us it is minor: the target forward has no torch fp32 GEMMs; only
  the DSpark drafter's attention einsums (`dspark.py`) run in TF32, which can change drafts, never replies; the
  prompt mHC kernel's TF32 dot is deliberate (as vLLM).
- Decode round cost (bertholomus bd0024d, +18% single stream): drafter 5.1 -> 3.3 ms (vocabulary split per rank, one
  argmax gather, cached Markov bias rows for the 256 most frequent tokens; ours recomputes
  `markov_embed[prev] @ markov_head.T` each draft token); 6-row verify 39.5 -> 36.8 ms (ours 48.5); cross-node
  gathers 2.9 -> 1.9 ms a round (rings and flags in cudaHostAlloc memory, one system fence a staging block).
- Lazy Triton loads mid-serving (bertholomus#1: a 13-row prefill-tail specialization first loaded after 15.5 h failed
  with CUDA 800 and took the lane down): warm every small prefill-tail row count.
- Prompt chunks: `group_members` (`serial.py`) syncs the host three times a layer (bincount, `int(max)`, nonzero);
  bertholomus v0.5 builds the expert work list on the device.
- Long-context decode: exact pruned top-k, candidate-only reindex, tile skip (decode after a 128K prompt 89 -> 115-129).
- Opt-in reasoning loop guard (bertholomus dfbe519, `TF_LOOP_GUARD`).
- soumyarupsarkar's 8.6M resident KV tokens vs our 6.5M at 16 × 614K.
- sfxnz's lane decoder (`--parallel 1..4`, top_k-off lanes drawn together) and their frozen-ruler benchmark gates.
