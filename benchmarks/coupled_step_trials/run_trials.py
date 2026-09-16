#!/usr/bin/env python3
"""Run one-side step trials across the coupled families and report the funnel.

WHAT THIS ANSWERS. The campaign's coupled record is carried by two families:
of 579 graded cells, 34 are CORRECT and 30 of those are C2 and C3. C10 is 0 of
51. Four families have three attempts each and four have never been run at all.
So "how many of these can be solved" is genuinely unknown for eight of fourteen,
and a full graded round is the wrong instrument to find out -- it costs an hour
per cell and mostly re-measures a failure that is already understood.

The gate everything sits behind is smaller and cheaper to test: can each CODE be
driven to write a participant that RUNS and EXPORTS. Measured over the record,
of 217 cells with files only 99 got both sides to three levels, and the silent
side is B in 16 of 17 stuck cells.

WHAT IS RECORDED, and why the budget column matters. A side that exports in 8
calls is a different answer from one that exports in 30: CORRECT cells need at
least 42 tool calls in total, and a family whose per-call latency buys fewer has
never scored once. So export/no-export alone would hide the result.

Nothing here allocates a seed, opens a sealed key, or writes to the campaign.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from make_task import build, code_of, spec                      # noqa: E402

FAMILIES = [f"C{i}" for i in range(1, 15)]


def exported(workdir: Path) -> tuple[bool, int, float]:
    """Did a real exports.json appear, how many points, and is there a FIELD?

    THE CONDITION, NOT A PROXY -- AND "A FILE EXISTS" IS ITSELF A PROXY.
    This first returned (found, points), and on the first nine trials that read
    7 of 9 succeeded. Two of those seven carry a field that is identically zero
    in both values and fluxes: C1 side A wrote 132 points of 0.0 and C5 side A
    wrote 88. On a driven problem with a source term that is not an answer, and
    counting it as one made the per-family table wrong in the direction that
    flatters the tool.

    It is the same failure as the full C9 run that reached three levels and an
    interface residual of 6.8e-11 with nothing in it -- and it says how a
    coupled pair gets there: if one side can write zeros this easily ALONE,
    with no partner and nothing to orchestrate, then two sides doing it agree
    perfectly and every self-consistency measure reads as success.

    So the peak magnitude travels with the count, and the caller decides.
    """
    best = (False, 0, 0.0)
    for path in sorted(workdir.rglob("exports.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:                                        # noqa: BLE001
            continue
        if not isinstance(data, dict):
            continue
        # A FILE IN THE WRONG ENVELOPE IS NOT A MISSING FILE, and reporting it
        # as one loses the whole diagnosis. `imports.json` is keyed by partner
        # -- {"A": {...}} -- while `exports.json` is FLAT. One trial mirrored
        # the imports envelope onto its exports, so 44 correct points read as
        # "no file" until this looked inside.
        #
        # Worth a caution before blaming the model for it: this harness's own
        # task text describes the imports envelope explicitly, which the
        # campaign's task text does not, and across 801 exports.json files in
        # the graded record NOT ONE is wrapped. The mistake may be the prompt's.
        if "coordinates" not in data and len(data) == 1:
            inner = next(iter(data.values()))
            if isinstance(inner, dict) and "coordinates" in inner:
                data = dict(inner)
                data["_wrapped_envelope"] = True
        for key in ("values", "value", "field"):
            v = data.get(key)
            if not (isinstance(v, list) and v):
                continue
            flat: list = []
            for entry in v:
                flat.extend(entry if isinstance(entry, (list, tuple)) else [entry])
            nums = [abs(float(x)) for x in flat if isinstance(x, (int, float))]
            peak = max(nums) if nums else 0.0
            if peak > best[2] or not best[0]:
                best = (True, len(v), peak)
                if data.get("_wrapped_envelope"):
                    print("      note: exports.json was wrapped in a side key; "
                          "the driver reads it flat and would reject it",
                          flush=True)
    return best


def run_one(problem: str, side: str, model: str, step_limit: int,
            keep_going: int, timeout_s: int, out_root: Path) -> dict:
    work = out_root / f"{problem}_{side}"
    work.mkdir(parents=True, exist_ok=True)
    task = build(problem, side)
    (work / "_task.txt").write_text(task)
    log = out_root / f"{problem}_{side}.log"
    started = time.time()
    cmd = [str(REPO / ".venv/bin/python"), str(REPO / "run_agent.py"),
           "--model", model, "--step-limit", str(step_limit),
           "--keep-going", str(keep_going), "--workdir", str(work), task]
    with log.open("w") as fh:
        try:
            subprocess.run(cmd, cwd=REPO, stdout=fh, stderr=subprocess.STDOUT,
                           timeout=timeout_s)
        except subprocess.TimeoutExpired:
            fh.write("\n[trial timed out]\n")
    text = log.read_text(errors="replace")
    calls = re.findall(r"^  → (\w+)", text, re.M)
    ok, npts, peak = exported(work)
    return {"problem": problem, "side": side, "code": code_of(spec(problem), side),
            "role": (spec(problem).get("roles") or {}).get(side, ""),
            "exported": ok, "points": npts, "peak": peak,
            "carries_field": bool(ok and peak >= 1e-8), "calls": len(calls),
            "knowledge_calls": calls.count("knowledge"),
            "wall_s": round(time.time() - started, 1),
            "tools": dict(sorted({t: calls.count(t) for t in set(calls)}.items(),
                                 key=lambda kv: -kv[1]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--problem", action="append", help="limit to these (repeatable)")
    ap.add_argument("--side", action="append", choices=["A", "B"])
    ap.add_argument("--model", default="qwen/qwen3.5-27b")
    ap.add_argument("--step-limit", type=int, default=45)
    ap.add_argument("--keep-going", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY is not set", file=sys.stderr)
        return 2
    problems = args.problem or FAMILIES
    sides = args.side or ["A", "B"]
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for problem in problems:
        for side in sides:
            row = run_one(problem, side, args.model, args.step_limit,
                          args.keep_going, args.timeout, args.out)
            rows.append(row)
            mark = ("OK  " if row["carries_field"]
                    else "ZERO" if row["exported"] else "no  ")
            print(f"{mark} {problem:4s} {side}  {row['code']:8s} {row['role']:10s} "
                  f"calls={row['calls']:3d} knowledge={row['knowledge_calls']} "
                  f"points={row['points']:4d} peak={row['peak']:.2e} "
                  f"{row['wall_s']:7.1f}s", flush=True)
            if args.json:
                args.json.write_text(json.dumps(rows, indent=2) + "\n")

    wrote = [r for r in rows if r["exported"]]
    done = [r for r in rows if r.get("carries_field")]
    print(f"\n{len(wrote)} of {len(rows)} sides wrote exports.json; "
          f"{len(done)} of those carry a field (the rest are identically zero)")
    print(f"median calls when it worked: "
          f"{sorted(r['calls'] for r in done)[len(done)//2] if done else '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
