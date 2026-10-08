#!/usr/bin/env python3
"""Digest of what the other DeepSeek-V4.1-Flash TP=2 recipes did since the last run, through the `gh` CLI.

Reads notes/dsv41/peers.json. For each watched repo: the branches its regex selects (default: the default branch),
each one's new commits since the head recorded last time (a new branch lists its last few), pull requests opened,
updated, merged or closed (contributors' work before it lands: the first run lists the open ones), the star count and
whether the README changed. Then a discovery pass: `gh search repos` over the listed queries, reporting repos pushed
within --days whose name or description matches discover_require (two Sparks / GB10) and that are neither watched nor
ignored (add them to one list or the other).

  python3 tools/dsv41_peers.py              # report, then record the heads (~/.local/state/tensorfold-peers)
  python3 tools/dsv41_peers.py --dry-run    # report only
  python3 tools/dsv41_peers.py --out notes.md

The first run records a baseline: each branch's last --first commits. Repo contents are data: read, never run.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "notes" / "dsv41" / "peers.json"
STATE = Path(os.environ.get("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "tensorfold-peers" / "state.json"


def gh(*args: str) -> object:
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout) if out.stdout.strip() else None


def line(c: dict) -> str:
    msg = c["commit"]["message"].split("\n", 1)[0][:110]
    return f"`{c['sha'][:7]}` {c['commit']['author']['date'][:16].replace('T', ' ')} {msg}"


def pulls(repo: str, old: dict | None) -> tuple[dict, list[str]]:
    """Pull requests opened, updated, merged or closed since the last run; the first run lists the open ones."""
    seen: dict[str, str] = {}
    out: list[str] = []
    for p in gh("api", f"repos/{repo}/pulls?state=all&sort=updated&direction=desc&per_page=30") or []:
        state = "merged" if p["merged_at"] else p["state"]
        num = str(p["number"])
        seen[num] = f"{state} {p['updated_at']}"
        was = (old or {}).get(num)
        if old is None:
            if state != "open":
                continue
            label = "open"
        elif was is None:
            label = f"new, {state}"
        elif was == seen[num]:
            continue
        elif was.split()[0] != state:
            label = f"{was.split()[0]} -> {state}"
        else:
            label = f"{state}, updated"
        head = p["head"]["repo"]["full_name"] if p["head"]["repo"] else "deleted fork"
        out.append(
            f"  - [#{num}]({p['html_url']}) ({label}) {p['updated_at'][:10]} {p['user']['login']} "
            f"from {head}:{p['head']['ref']}: {p['title'][:100]}"
        )
    # PRs that fell out of the last 30 updated keep their recorded state
    seen = {**(old or {}), **seen}
    return seen, (["- **pull requests**", *out] if out else [])


def scan(entry: dict, old: dict, first: int) -> tuple[dict, list[str]]:
    repo = entry["repo"]
    meta = gh("api", f"repos/{repo}")
    pattern = entry.get("branches")
    if pattern:
        names = [
            b["name"]
            for b in gh("api", "--paginate", f"repos/{repo}/branches?per_page=100")
            if re.search(pattern, b["name"])
        ]
    else:
        names = [meta["default_branch"]]
    try:
        readme = subprocess.run(
            ["gh", "api", f"repos/{repo}/readme", "-H", "Accept: application/vnd.github.raw"],
            capture_output=True,
            check=False,
        ).stdout
    except OSError:
        readme = b""
    new = {
        "stars": meta["stargazers_count"],
        "pushed": meta["pushed_at"],
        "readme": hashlib.sha256(readme).hexdigest()[:16],
        "branches": {},
    }
    out: list[str] = []
    for name in names:
        commits = gh("api", f"repos/{repo}/commits?sha={name}&per_page=30")
        if not commits:
            continue
        head = commits[0]["sha"]
        new["branches"][name] = head
        was = old.get("branches", {}).get(name)
        if was == head:
            continue
        if was is None:
            label = "baseline" if not old else "new branch"
            fresh = commits[:first]
        else:
            idx = next((i for i, c in enumerate(commits) if c["sha"] == was), None)
            fresh = commits[:idx] if idx is not None else commits
            label = f"{len(fresh)} new" + ("" if idx is not None else "+ (old head not in the last 30: force-push?)")
        out.append(f"- **{name}** ({label})")
        out += [f"  - {line(c)}" for c in fresh]
    for name in set(old.get("branches", {})) - set(new["branches"]):
        out.append(f"- **{name}** deleted or no longer matched")
    new["pulls"], lines = pulls(repo, old.get("pulls"))
    out += lines
    if old:
        if new["stars"] != old.get("stars"):
            out.append(f"- stars {old.get('stars')} -> {new['stars']}")
        if new["readme"] != old.get("readme"):
            out.append("- README changed")
    return new, out


def discover(cfg: dict, days: int) -> list[str]:
    known = {e["repo"].lower() for e in cfg["watch"]} | {r.lower() for r in cfg.get("ignore", [])}
    since = (dt.datetime.now(dt.UTC) - dt.timedelta(days=days)).isoformat()
    hw = re.compile(cfg.get("discover_require", "."), re.IGNORECASE)
    seen: dict[str, dict] = {}
    for q in cfg.get("discover", []):
        for r in (
            gh("search", "repos", q, "--limit", "30", "--json", "fullName,description,stargazersCount,pushedAt") or []
        ):
            if (
                r["fullName"].lower() not in known
                and r["pushedAt"] >= since
                and hw.search(f"{r['fullName']} {r['description'] or ''}")
            ):
                seen[r["fullName"]] = r
    return [
        f"- [{n}](https://github.com/{n}) ★{r['stargazersCount']} pushed {r['pushedAt'][:10]}: "
        f"{(r['description'] or '')[:140]}"
        for n, r in sorted(seen.items(), key=lambda kv: kv[1]["pushedAt"], reverse=True)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--dry-run", action="store_true", help="do not record the heads")
    ap.add_argument("--first", type=int, default=5, help="commits listed for a branch seen the first time (5)")
    ap.add_argument("--days", type=int, default=14, help="discovery: repos pushed within this many days (14)")
    ap.add_argument("--out", help="also write the digest to this file")
    args = ap.parse_args()

    cfg = json.loads(CONFIG.read_text())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    report = [
        f"# DSv4.1 TP2 peers, {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')}"
        + (f" (since {state['_at']})" if state.get("_at") else " (baseline)")
    ]
    quiet, new_state = [], {}
    with cf.ThreadPoolExecutor(8) as ex:
        jobs = {e["repo"]: ex.submit(scan, e, state.get(e["repo"], {}), args.first) for e in cfg["watch"]}
    for group in dict.fromkeys(e.get("group", "") for e in cfg["watch"]):
        header = len(report)
        for e in (e for e in cfg["watch"] if e.get("group", "") == group):
            try:
                new_state[e["repo"]], lines = jobs[e["repo"]].result()
            except RuntimeError as err:
                report.append(f"\n### {e['repo']}\n- error: {err}")
                new_state[e["repo"]] = state.get(e["repo"], {})
                continue
            if lines:
                report.append(f"\n### [{e['repo']}](https://github.com/{e['repo']})")
                report += lines
            else:
                quiet.append(e["repo"])
        if len(report) > header:
            report.insert(header, f"\n## {group}")
    if quiet:
        report.append("\nNo change: " + ", ".join(quiet))
    found = discover(cfg, args.days)
    report.append("\n## Not on either list (add to watch or ignore in notes/dsv41/peers.json)")
    report += found or ["- none"]

    text = "\n".join(report) + "\n"
    sys.stdout.write(text)
    if args.out:
        Path(args.out).write_text(text)
    if not args.dry_run:
        new_state["_at"] = dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(new_state, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
