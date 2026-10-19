# DeepSeek-V4.1-Flash on 2× DGX Spark (TP=2): benchmark

Measured 2026-10-09 on branch `dsv41-cuda` (commit `11817b9`), served from the deploy image built from that tree.
All numbers below are taken over HTTP from the running server, the way any client sees it.

## Setup

| | |
|---|---|
| Hardware | 2× NVIDIA DGX Spark (GB10, 128 GB unified memory each) |
| Interconnect | one 200 GbE RoCE link (ConnectX-7, one port) between the two boxes |
| Parallelism | tensor parallel 2 (one rank a box) |
| Model | DeepSeek-V4.1-Flash, Mia-AI-Lab EXL3 2.9 bpw quant |
| Engine | this TensorFold fork, CUDA graphs, FP4 KV cache, DSpark speculative drafter (up to 5 drafts a round) |
| Server | `deploy/dsv41-tp2` compose: `--parallel 16 --context 614400 --reasoning-effort low`, every other switch at its default |
| KV cache | one shared FP4 pool of **5,150,720 tokens** for 16 streams of up to 614,400 tokens each (+ 8 GiB of kept prompt states on NVMe) |

Every speed change in this fork is exact: a drafted reply has the same tokens as plain greedy decode, and the
quick-tier fingerprint suite is unchanged across changes.

## TG: single-request decode (jaybench)

[`tools/dsv41_jaybench_http.py`](tools/dsv41_jaybench_http.py): jayleaton's m2bench single-stream prompts
([deepseek-v41-tensorfold-spark](https://github.com/jayleaton/deepseek-v41-tensorfold-spark), MIT), sent to the
server's `/v1/chat/completions`.

- One user message, no system prompt, thinking off, temperature 0, `max_tokens` 384, streaming.
- tok/s = (completion tokens − 1) / (last content chunk − first content chunk): decode speed, prefill excluded.
- One warm-up request a workload, then 3 measured requests, each after the server is idle; median reported.

| Workload | Prompt | tok/s (median of 3) | runs |
|---|---|---|---|
| code | `LRUCache` class with tests | **91.1** | 91.4 / 91.1 / 91.1 |
| prose | 400-word essay on lighthouses | **48.2** | 48.5 / 48.2 / 48.1 |
| structured | count from 1 to 200 | **132.2** | 132.3 / 132.0 / 132.2 |
| c1_hard | long essay on the history of computing | **51.3** | 51.0 / 51.3 / 51.3 |

Speculative decoding helps most where replies are predictable (structured, code) and least on free prose.

## PP: cold prefill

[`tools/prefill_cold.py`](tools/prefill_cold.py): chat prompts of fixed token lengths cut from the Python standard
library's source, each with a unique first line so no cached prefix is reused, then "Say in one sentence what the
code above does."

- Thinking off, temperature 0, `max_tokens` 2, streaming.
- tok/s = prompt tokens / time to first token (TTFT; includes the first decoded token).
- One 1,024-token warm-up, then 3 prompts a length; median reported.

| Prompt | TTFT (median) | tok/s |
|---|---|---|
| 8K (8,192) | 5.76 s | **1,423** |
| 32K (32,767) | 15.71 s | **2,086** |
| 64K (65,536) | 29.27 s | **2,239** |
| 128K (131,072) | 63.23 s | **2,073** |

## Concurrency

[`tools/dsv41_clients.py`](tools/dsv41_clients.py): 16 clients, each sending 256-token requests back to back for
120 s (varied prompts, thinking off), aggregate completion tokens over the wall time: **127–130 tok/s**
(8.5–8.7 tok/s per request).

## Reproduce

```bash
# on the head node, with the server up (deploy/dsv41-tp2: make up)
TF_API_KEY=... python3 tools/dsv41_jaybench_http.py --reps 3 > tg.json          # needs httpx
python3 tools/prefill_cold.py build /models/.../DeepSeek-V4.1-Flash-EXL3-2.9bpw pp.json   # needs tokenizers
PREFILL_LENGTHS=8192,32768,65536,131072 TF_API_KEY=... \
    python3 tools/prefill_cold.py run http://localhost:8888 DeepSeek-v4.1-Flash-EXL3 pp.json pp_out.json
TF_API_KEY=... python3 tools/dsv41_clients.py 16 120
```

## DeepSeek-V4.1-Flash-1017 (2026-10-19)

The deployment moved to DeepSeek-V4.1-Flash-1017 (LibertAIDAI EXL3 2.9 bpw, the same tensor widths as the release
above, with the same Keys `wo_b` overlay the old deployment served), same engine and settings. Same tools, the day of
the cutover; the old row was measured just before it:

| | earlier release | 1017 |
|---|---|---|
| TG code / prose / structured / c1_hard | 89.5 / 48.7 / 132.9 / 51.4 | 82.4 / 49.8 / 135.4 / 54.7 |
| PP 8K / 32K / 64K / 128K | 1,449 / 2,121 / 2,249 / 2,146 | 1,449 / 2,075 / 2,228 / 2,077 |
| 16 clients, aggregate | 126.9 | 126.0 |
| KV pool (FP4 tokens) | 5,150,720 | 5,820,416 |

Prefill and concurrency match within noise (the same engine). The code workload's drop is the drafter, not the step:
1017's reply takes 109 rounds at 3.52 tokens each against 95 at 4.04 (c1_hard moves the other way, 2.01 against 1.68).
Served on 1017: a 15-minute soak (318 requests, 0 errors, 0 sha splits) and the stress run were clean.

## Notes for comparing

- TG excludes prefill (first-to-last content chunk); PP includes the first token. Both are per request, one request
  at a time.
- The kept-prompt cache would turn repeated prompts into cache hits: the PP prompts are unique, and TG prompts are
  short enough that it doesn't matter.
- Run to run, TG varies by about ±1 tok/s across server restarts (the draft-cost calibration is timed at start).
- How the numbers moved and what each change did: [TODO.md](TODO.md), "Phase 9" and "Phase 10"; method details in
  [notes/dsv41/DEV.md](notes/dsv41/DEV.md).
