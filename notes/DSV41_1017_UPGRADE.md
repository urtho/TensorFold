# DeepSeek-V4.1-Flash-1017 upgrade plan

Status as of 2026-10-18. Resume from "Day 2".

## Why this is a weight swap

`deepseek-ai/DeepSeek-V4.1-Flash-1017` was released 2026-10-17 at rev `1bae0e10`.

**Unchanged** compared with the V4.1 we serve (09-10, through the Mia 2.9 bpw pack):
- `config.json` (byte-identical)
- `model.safetensors.index.json` (byte-identical)
- tokenizer and `encoding/`
- MTP `mtp.0-2`

**Changed:**
- All weights. Shard sizes are identical and the sha256 values differ.
- The Engram shards 47/48. They are new; never reuse the old ones.
- 1017 ships no `chat_template.jinja`. Copy ours from the current pack.

## Day 1 (done): staged on alab

`root@alab:/hdd/share/models/dsv41-1017/` is on ZFS `hdd/share`, which is exported with `sharenfs rw=@192.168.1.0/24`. aiai currently mounts only `/mnt2/share` (120 GiB free, too small).

| Path | Source | Size |
|---|---|---|
| `DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/` | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `c178a012` (whole repo) | 196 GiB |
| `DeepSeek-V4.1-Flash-1017-engram/` | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`: shards 47/48, `config.json`, index, tokenizer | 189 GiB |
| `abliteration/mrfakename-1017/` | `mrfakename/DeepSeek-V4.1-Flash-1017-abliterated`: `abliteration-edit.npz` (Q + per-layer scales, `W' = W - s*Q*Q^T*W` on `attn.wo_b` and `ffn.shared_experts.w2`), LoRA, README | 5 MiB |

The download runs in `tmux` session `dsv41-1017` on alab through `run.sh`, which uses aria2c with per-file sha256. To resume, rerun `./run.sh`; it continues where it stopped.

`DONE.txt` gets an "aria2c exit 0" line when the download finishes, and a "verify exit 0" line once `sha256sum -c SHA256SUMS` passes (output in `verify.txt`).

Check: `ssh root@alab 'cat /hdd/share/models/dsv41-1017/DONE.txt; tail -3 /hdd/share/models/dsv41-1017/run.out'`

## Day 2

1. **Check the staging.** `DONE.txt` must show `aria2c exit 0` and `verify exit 0`.
2. **Stop serving and free the old weights on both nodes** (aiai and aiai2, each 916 GiB with about 95 GiB free).
   - Run `make down` in `deploy/dsv41-tp2`.
   - Delete `~/models/dsv41/DeepSeek-V4.1-Flash-EXL3-2.9bpw` and `~/models/dsv41/DeepSeek-V4.1-Flash-engram`.
   - On aiai, also delete `~/.cache/huggingface/dsv41-exl3` and `dsv41-engram`. They are hardlinks to the same inodes, so the space only comes back when every link is gone.
   - Delete `~/.cache/tensorfold-prepared/DeepSeek-V4.1-Flash-EXL3-2.9bpw` (99 GiB of fastboot cache, rebuilt by `make prepare`).
   - **Keep** `~/models/dsv41/keys-dsv41-wo_b-overlay`. It holds the recipe and metadata.
   - **Keep** the refusal directions in `~/.cache/huggingface/hub/models--drowzeys--keys-DeepSeekV4-Flash-GA-0731-Dspark-Abliterated-Anchored-Tensors/snapshots/*/ablit/`. Copy them out first if the 157 GiB drowzeys cache gets deleted.
   - Optional: the old pack can be kept on alab `/hdd/share` for rollback (copy before deleting).
3. **Copy to both nodes:**
   - Mount it: `mount -t nfs4 192.168.1.210:/hdd/share /mnt/alab-hdd`, or rsync over ssh from alab.
   - Lay it out as `~/models/dsv41/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` and `~/models/dsv41/DeepSeek-V4.1-Flash-1017-engram`.
   - Measure the link first with iperf3. About 385 GiB per node is around 1 h per node at 1 GbE. Copy aiai2 from aiai over IB if that is faster.
   - Copy `chat_template.jinja` from the old pack, or from `/mnt/alab-share` if it was saved there.
4. **Engine checks:**
   - The LibertAI pack adds 43 `*.ffn.gate.bias_vl` (fp32 [384], image routing). Make sure the loader ignores them; `weights.py` only reads named tensors.
   - The `kvdisk.compat_ident` cache key uses (name, size) of the weight and Engram files, plus sha256 of `config.json` and `tokenizer.json`, all of which are unchanged. A stale prompt-KV disk cache could therefore be reused. Clear the dsv41 kvdisk cache, or better, add the index or first-shard sha or the `ABLIT_META` sha to the key.
5. **Settings** (`.env` on both nodes):
   - Set `MODEL=dsv41/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` and `ENGRAM=dsv41/DeepSeek-V4.1-Flash-1017-engram`.
   - **Keep `--alias deepseek-v4-flash-0731`**: the user's apps use it.
   - Optionally add `--alias deepseek-v4.1-flash-1017` (the models.dev id).
   - `SERVED_NAME` can stay as is.
6. **Bring it up and check it:** run `make prebuild`, `make prepare` (fastboot) and `make up`, then `make smoke`. Compare against the BENCHMARK.md numbers with `tools/dsv41_jaybench_http.py`.
7. **Uncensoring.** The current served pack carries our own `wo_b` overlay: layers 10–35, K=5 EXL3 mul1, λ 3.5, one direction (`ABLIT_META.json`, `apply_mia_exl3_wob.py`). It does not carry over, because the weights changed. Options, best first:
   - **(a)** Rerun our recipe on 1017.
     - Fetch only `layers.10-35.attn.wo_b.{weight,scale}`, FP8, about 40 MB each, one per shard in shards 13–38 of `deepseek-ai/DeepSeek-V4.1-Flash-1017`. Use HTTP range requests from the safetensors header offsets instead of whole 5 GiB shards.
     - Edit them with the refusal direction, re-encode to EXL3 K=5 mul1, and splice them in with `apply_mia_exl3_wob.py`.
     - Optionally recapture the direction on 1017.
   - **(b)** Use mrfakename's `abliteration-edit.npz`: KL 0.067, refusals 97→4. It also edits `shared_experts.w2`, so those tensors need re-encoding too.
   - **(c)** `dealignai/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw-UNCENSORED`. Avoid it: its BF16 `wo_b`/`w2` in layers 10–36 need a BF16 linear path in our engine, and it scores −4.2 points on MMLU.

## Separately: syncing with upstream

Upstream 1.x is Zig with no DeepSeek family, so the only sync target is v0.6.6. A trial merge conflicted in 7 files:
- `cli.py`, `cli_args.py`
- `cuda/capacity.py`, `cuda/comm.py`, `cuda/http.py`, `cuda/server.py`
- `server/request_options.py`

For API keys, keep upstream's `--api-key` options and keep `TF_API_KEY` as an alias. The carveout PR #487 needs no porting back, because our Python carveout is the original.
