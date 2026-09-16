#!/usr/bin/env python3
"""Turn the step-trial logs into the funnel, per family and per code.

The tally alone would not be worth much. What makes these trials diagnostic is
the pair of columns beside it: how many calls a working side cost, and how many
of its shell calls ended in a traceback. The campaign's stuck cells spend 38 %
of their run_bash calls on tracebacks and about 26 write-run-error-rewrite
cycles each; if a trial exports with none, the API knowledge was never the
obstacle for that code and the difficulty lies in the orchestration around it.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ERR = re.compile(r"\b([A-Za-z_][A-Za-z_.]*(?:Error|Exception)): ([^\n\"]{0,70})")


def log_stats(log: Path) -> dict:
    if not log.is_file():
        return {"shell": 0, "tracebacks": 0, "errors": Counter()}
    text = log.read_text(errors="replace")
    shell = len(re.findall(r"^  → run_bash", text, re.M))
    errs = Counter(f"{k}: {v.strip()[:48]}" for k, v in ERR.findall(text))
    return {"shell": shell, "tracebacks": sum(errs.values()), "errors": errs}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True)
    args = ap.parse_args()

    rows = json.loads(args.json.read_text())
    print(f"{'fam':5s} {'side':4s} {'code':8s} {'role':10s} {'exported':>8s} "
          f"{'pts':>5s} {'calls':>5s} {'know':>4s} {'errs':>5s}")
    both = Counter()
    allerr = Counter()
    for r in rows:
        st = log_stats(args.logs / f"{r['problem']}_{r['side']}.log")
        allerr.update(st["errors"])
        both[r["problem"]] += 1 if r["exported"] else 0
        print(f"{r['problem']:5s} {r['side']:4s} {r['code']:8s} {r['role']:10s} "
              f"{('YES' if r['exported'] else 'no'):>8s} {r['points']:5d} "
              f"{r['calls']:5d} {r['knowledge_calls']:4d} {st['tracebacks']:5d}")

    done = [r for r in rows if r["exported"]]
    fams = sorted({r["problem"] for r in rows}, key=lambda s: int(s[1:]))
    print(f"\n{len(done)} of {len(rows)} sides exported")
    print(f"families with BOTH sides exporting: "
          f"{sum(1 for f in fams if both[f] == 2)} of {len(fams)}")
    if done:
        calls = sorted(r["calls"] for r in done)
        print(f"calls when it worked: median {calls[len(calls)//2]}, "
              f"min {calls[0]}, max {calls[-1]}")
    per_code: dict = {}
    for r in rows:
        per_code.setdefault(r["code"], [0, 0])
        per_code[r["code"]][1] += 1
        per_code[r["code"]][0] += 1 if r["exported"] else 0
    print("\nby code:")
    for code, (ok, n) in sorted(per_code.items(), key=lambda kv: -kv[1][0] / kv[1][1]):
        print(f"  {code:9s} {ok}/{n}")
    if allerr:
        print("\nmost widespread errors across all trials:")
        for k, n in allerr.most_common(8):
            print(f"  {n:4d}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
