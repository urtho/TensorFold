#!/usr/bin/env python3
"""Reasoning-length probe: replay chat requests against an OpenAI-compatible server with controlled variations
(temperature, seed, max_tokens, thinking_budget, reasoning_effort, loop_guard) and measure how each reply's reasoning
ends: finish reason, reasoning tokens, and two loop signals over the reasoning text, standard library only.

- ``dry``: the loop guard's own signal (``call_gate.ThinkLoop``: 1,024-token windows, the share of 8-grams no earlier
  window held), computed from the reasoning's token ids (the tokenizer named by --tokenizer, via ``tokenizers`` if
  installed, else whitespace words); ``max_dry`` = the longest run of windows under 2% new.
- ``repeat``: the most frequent line of the reasoning (stripped, >= 12 chars) and its count — rumination repeats
  questions and verdicts as lines ("AlgoSwap? No.") long before exact 8-gram windows repeat.

  TF_API_KEY=... python3 tools/dsv41_reasoning_probe.py --base http://127.0.0.1:8888 --model M \\
      --request a.request.json [--request ...] --temperature 0.1,1.0 --seeds 1,2,3 --max-tokens 16000 \\
      [--set thinking_budget=3000] --concurrency 2 --out probe.jsonl

The key is read from TF_API_KEY only. Every reply's reasoning and content go to --out (JSONL, one row a request).
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import itertools
import json
import os
import sys
import time
import urllib.request
from pathlib import Path


def tokenizer(path: str | None):
    if path:
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(path)
            return lambda text: tok.encode(text, add_special_tokens=False).ids
        except ImportError:
            pass
    return lambda text: text.split()


def dry_windows(ids: list, width: int = 1024, n: int = 8, least: float = 0.02) -> tuple[list[float], int]:
    """ThinkLoop's per-window novelty and the longest run of windows under ``least``."""

    seen: set = set()
    tail: list = []
    out, run, best = [], 0, 0
    for w0 in range(0, len(ids) - width + 1, width):
        win = ids[w0:w0 + width]
        seq = [*tail, *win]
        grams = [tuple(seq[j:j + n]) for j in range(len(seq) - n + 1)]
        nov = sum(1 for g in grams if g not in seen) / max(1, len(grams))
        seen.update(grams)
        tail = seq[-(n - 1):]
        out.append(round(nov, 4))
        run = run + 1 if nov < least else 0
        best = max(best, run)
    return out, best


def repeat(text: str) -> tuple[str, int]:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) >= 12]
    if not lines:
        return "", 0
    line, count = collections.Counter(lines).most_common(1)[0]
    return line[:120], count


def post(base: str, key: str, body: dict, timeout: float) -> dict:
    req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        k, _, v = item.partition("=")
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True, help="the served model name (replaces the request's)")
    ap.add_argument("--request", action="append", required=True, help="a chat request body (JSON file); repeatable")
    ap.add_argument("--temperature", default="", help="comma list (empty: the request's own)")
    ap.add_argument("--seeds", default="", help="comma list of seeds (empty: none sent)")
    ap.add_argument("--max-tokens", type=int, default=0, help="override max_tokens (0: the request's)")
    ap.add_argument("--set", action="append", default=[], help="extra body field key=json (repeatable)")
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--tokenizer", default=str(Path(__file__).resolve().parent.parent / "notes/dsv41/tokenizer.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    key = os.environ.get("TF_API_KEY", "")
    tok = tokenizer(a.tokenizer)
    extra = parse_set(a.set)
    temps = [float(t) for t in a.temperature.split(",") if t] or [None]
    seeds = [int(s) for s in a.seeds.split(",") if s] or [None]
    jobs = []
    for path, temp, seed in itertools.product(a.request, temps, seeds):
        body = json.loads(Path(path).read_text())
        body["model"] = a.model
        body.pop("stream", None)
        if temp is not None:
            body["temperature"] = temp
        if seed is not None:
            body["seed"] = seed
        if a.max_tokens:
            body["max_tokens"] = a.max_tokens
        body.update(extra)
        jobs.append((Path(path).name, temp, seed, body))

    def run(job):
        name, temp, seed, body = job
        t0 = time.perf_counter()
        try:
            r = post(a.base, key, body, a.timeout)
        except Exception as exc:  # noqa: BLE001 - recorded per row
            return {"request": name, "temperature": temp, "seed": seed, "error": repr(exc),
                    "elapsed_s": round(time.perf_counter() - t0, 1)}
        c = r["choices"][0]
        m = c["message"]
        reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
        nov, max_dry = dry_windows(tok(reasoning))
        line, count = repeat(reasoning)
        u = r.get("usage") or {}
        return {"request": name, "temperature": temp, "seed": seed, "extra": extra,
                "elapsed_s": round(time.perf_counter() - t0, 1), "finish": c.get("finish_reason"),
                "completion_tokens": u.get("completion_tokens"),
                "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                "content_chars": len(m.get("content") or ""), "novelty": nov, "max_dry": max_dry,
                "repeat_line": line, "repeat_count": count, "loop_guard": (r.get("tensorfold") or {}).get("loop_guard"),
                "reasoning": reasoning, "content": m.get("content")}

    with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.concurrency) as ex:
        for row in ex.map(run, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            short = {k: row.get(k) for k in ("request", "temperature", "seed", "elapsed_s", "finish",
                                             "completion_tokens", "reasoning_tokens", "max_dry", "repeat_count",
                                             "error")}
            print(json.dumps(short), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
