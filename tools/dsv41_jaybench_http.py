#!/usr/bin/env python3
"""jayleaton's m2bench single-stream cells (code / prose / structured) over HTTP against our server.

Prompts from jayleaton/deepseek-v41-tensorfold-spark m2bench.py (MIT, Copyright (c) 2026 Jay Leaton), see prompts.json.
Settings: one user message, no system prompt, thinking off, temperature 0, max_tokens 384, streaming.
tok/s = (completion_tokens - 1) / (t_last_content_chunk - t_first_content_chunk), median of REPS.
Before each rep: wait for /health requests_running == 0. Per request: /health counter deltas (rounds, drafted,
accepted) and the server's "done <id>" log line (matched by the response id, fetched by the caller).
The server's key from TF_API_KEY.

  TF_API_KEY=... python3 tools/dsv41_jaybench_http.py [--base http://localhost:8888] [--reps 3] [--only code,prose] > out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

import httpx

BASE = "http://localhost:8888"
MODEL = "DeepSeek-v4.1-Flash-EXL3"

CODE = """Write a Python class `LRUCache` with `get(key)` and `put(key, value)` in O(1) using a dict and a doubly linked
list, with docstrings and type hints, then three unit tests with pytest."""
PROSE = """Write a 400-word essay on why lighthouses were built where they were, how their keepers lived, and what
replaced them. Plain prose, no lists or headings."""
STRUCTURED = "Count from 1 to 200, separated by commas, nothing else."
# our bench_decode C1hard (nonce fixed here; bench_decode runs it at T 0.2 / top_p 0.95 with 2048 tokens)
HARD = ("Benchmark jaybench0: write a very long, detailed essay on the history of "
        "computing. Keep going until you are cut off.")
WORKLOADS = {"code": CODE, "prose": PROSE, "structured": STRUCTURED, "c1_hard": HARD}


def api_key() -> str:
    import os
    return os.environ.get("TF_API_KEY", "")


def health(c: httpx.Client) -> dict:
    return c.get(BASE + "/health").json()


def wait_idle(c: httpx.Client, timeout: float = 1800) -> int:
    t0, waited = time.time(), 0
    while True:
        h = health(c)
        if h.get("requests_running", 0) == 0 and not h.get("busy"):
            # two consecutive idle reads 0.5 s apart
            time.sleep(0.5)
            h = health(c)
            if h.get("requests_running", 0) == 0 and not h.get("busy"):
                return waited
        waited += 1
        if time.time() - t0 > timeout:
            raise RuntimeError("server never went idle")
        time.sleep(2)


def one(c: httpx.Client, hdr: dict, prompt: str, max_tokens: int, ignore_eos: bool) -> dict:
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0,
            "max_tokens": max_tokens, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    if ignore_eos:
        body["ignore_eos"] = True
    t_first = t_last = None
    completion = None
    rid = None
    chunks = 0
    text = []
    finish = None
    t0 = time.perf_counter()
    with c.stream("POST", BASE + "/v1/chat/completions", headers=hdr, json=body, timeout=1800) as r:
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.read().decode()[:300]}")
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            now = time.perf_counter()
            rid = rid or obj.get("id")
            ch = obj.get("choices") or []
            if ch:
                d = ch[0].get("delta") or {}
                piece = d.get("content") or ""
                if piece:
                    chunks += 1
                    text.append(piece)
                    if t_first is None:
                        t_first = now
                    t_last = now
                finish = ch[0].get("finish_reason") or finish
            u = obj.get("usage")
            if u and u.get("completion_tokens"):
                completion = u["completion_tokens"]
    tok_s = (completion - 1) / (t_last - t_first) if completion and t_last and t_last > t_first else None
    return {"id": rid, "completion": completion, "chunks": chunks, "finish": finish,
            "ttft_s": round(t_first - t0, 3) if t_first else None, "window_s": round(t_last - t_first, 4),
            "tok_s": round(tok_s, 2) if tok_s else None, "text": "".join(text)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=384)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--only", default="code,prose,structured,c1_hard")
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    globals()["BASE"] = a.base.rstrip("/")
    k = api_key()
    hdr = {"Authorization": f"Bearer {k}"} if k else {}
    out = {"settings": vars(a), "workloads": {}}
    with httpx.Client(timeout=60) as c:
        for name in a.only.split(","):
            prompt = WORKLOADS[name]
            for _ in range(a.warmup):
                wait_idle(c)
                one(c, hdr, prompt, a.max_tokens, a.ignore_eos)
            runs = []
            for i in range(a.reps):
                waited = wait_idle(c)
                h0 = health(c)
                r = one(c, hdr, prompt, a.max_tokens, a.ignore_eos)
                time.sleep(0.3)
                h1 = health(c)
                d = {key: h1.get(key, 0) - h0.get(key, 0)
                     for key in ("requests_total", "rounds_total", "drafted_total", "accepted_total",
                                 "completion_tokens_total")}
                r["health_delta"] = d
                r["health_clean"] = d["requests_total"] == 1
                if d["rounds_total"]:
                    r["tokens_per_round_health"] = round(d["completion_tokens_total"] / d["rounds_total"], 3)
                r["idle_wait_polls"] = waited
                runs.append(r)
                print(f"[jaybench] {name} rep {i + 1}: {r['tok_s']} tok/s, {r['completion']} tok, "
                      f"finish={r['finish']}, health {d}", file=sys.stderr, flush=True)
            vals = [r["tok_s"] for r in runs if r["tok_s"]]
            out["workloads"][name] = {"median_tok_s": round(statistics.median(vals), 2) if vals else None,
                                      "runs": runs}
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
