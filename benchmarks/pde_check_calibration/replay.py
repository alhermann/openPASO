#!/usr/bin/env python3
"""Does the equation check actually separate a right answer from a wrong one?

WHY THIS EXISTS. The evaluation campaign is about to make verify_pde_consistency
run automatically inside couple(), because two wordings of the invitation to
call it were measured and both failed: 2 calls against 14 asks, then 0 against
7. The argument for it is that only the field against its own equation can tell
these cells apart -- three coupled runs that are internally self-consistent,
converge cleanly, and give different answers, one right and two wrong.

The argument has never been measured. This project's own rule is that a gate
nobody has watched fire is not a gate, so this replays the graded cells that
already exist -- their exported fields are on disk, their outcomes are recorded
-- through the real check, and reports whether the verdicts line up.

It uses ONLY public data: the exported CSVs the agent itself wrote, and
source_public / coefficients / domain / equation from spec_public.json, which is
what the task handed the agent. No sealed key is opened, no seed is allocated,
no model is called. It reads the campaign tree and never writes to it.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

DEFAULT_CAMPAIGN = Path.home() / "Schreibtisch" / "ofa-v2" / "campaign3_blind"


def load_check():
    """The real tool, registered the way the server registers it."""
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools

    load_all_backends()
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    register_consolidated_tools(_Recorder())
    return captured["verify_pde_consistency"]


def graded_outcome(campaign: Path, seed: str) -> str | None:
    for path in (campaign / "honest_rounds_20260910" / "grades").glob(f"*seed{seed}.json"):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            if "outcome" in data:
                return data["outcome"]
            for value in data.values():
                if isinstance(value, dict) and "outcome" in value:
                    return value["outcome"]
    return None


def cells(campaign: Path):
    """Every graded coupled run directory, as (problem, seed, workdir)."""
    for run in sorted((campaign / "runs").glob("C*_27b_MCP_seed*")):
        match = re.match(r"(C\d+)_27b_MCP_seed(\d+)$", run.name)
        if not match:
            continue
        work = run / "work"
        if work.is_dir():
            yield match.group(1), match.group(2), work


def solution_files(work: Path) -> list[str]:
    """The agent's own solution exports, finest level last."""
    found = sorted(work.glob("solution_level*.csv"))
    return [str(p) for p in found]


def spec_of(campaign: Path, problem: str) -> dict:
    path = campaign / "problems" / problem / "spec_public.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def source_for(spec: dict, side: str = "A") -> str:
    src = spec.get("source_public")
    if isinstance(src, dict):
        return str(src.get(side) or next(iter(src.values()), ""))
    return str(src or "")


def coefficient_for(spec: dict, side: str) -> str:
    """The side's own k, as a number.

    THE PUBLIC SPEC STATES IT IN PROSE and the check wants a number:
    "thermal conductivity k = 1 in subdomain A; thermal conductivity
    k = 1000 in subdomain B". An agent reads the same sentence in its task and
    has to do this conversion itself -- which is friction on the path to a
    check nobody calls, and worth recording as such.
    """
    text = str(spec.get("coefficients", ""))
    want = "subdomain " + side.upper()
    for clause in re.split(r"[;\n]", text):
        if want.lower() in clause.lower():
            found = re.search(r"=\s*([0-9.eE+-]+)", clause)
            if found:
                return found.group(1)
    found = re.search(r"=\s*([0-9.eE+-]+)", text)
    return found.group(1) if found else "1.0"


def domain_for(spec: dict, side: str) -> str:
    """That side's own rectangle, which the spec gives machine-readable."""
    extent = spec.get(f"extent_{side.lower()}")
    return json.dumps(extent) if extent else str(spec.get("domain", "[[0,1],[0,1]]"))


def side_files(work: Path, side: str) -> list[str]:
    """One side's solution exports, level order preserved."""
    return [str(p) for p in sorted(work.glob(f"solution_level*_{side.upper()}.csv"))]


def verdict_of(reply: str) -> str:
    head = reply.lstrip()
    if head.startswith("REFUSED"):
        return "REFUSED"
    import json as _j
    try:
        return str(_j.loads(reply.split("\n\n")[0]).get("verdict", "UNPARSED"))
    except Exception:                                # noqa: BLE001
        pass
    for token in ("NOT_APPLICABLE", "INCONSISTENT", "CONSISTENT", "UNDECIDED"):
        if token in reply[:3000]:
            return token
    return "UNPARSED"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", type=Path, default=DEFAULT_CAMPAIGN)
    ap.add_argument("--problem", action="append",
                    help="limit to these problems (repeatable), e.g. --problem C8")
    ap.add_argument("--seed", action="append", help="limit to these seeds")
    ap.add_argument("--json", type=Path, help="also write the rows here")
    args = ap.parse_args()

    if not args.campaign.is_dir():
        print(f"campaign not found: {args.campaign}", file=sys.stderr)
        return 2

    check = load_check()
    rows = []
    for problem, seed, work in cells(args.campaign):
        if args.problem and problem not in args.problem:
            continue
        if args.seed and seed not in args.seed:
            continue
        outcome = graded_outcome(args.campaign, seed)
        if outcome is None:
            continue
        spec = spec_of(args.campaign, problem)
        # PER SIDE, because the problem is per side: each subdomain has its own
        # coefficient, its own source and its own rectangle. Handing both sides
        # to one call asks the check about a field that solves two different
        # equations, which is not a question it can answer.
        verdicts = {}
        n_files = 0
        for side in ("A", "B"):
            files = side_files(work, side)
            n_files += len(files)
            if not files:
                verdicts[side] = "NO_FILES"
                continue
            reply = check(solution_files=",".join(files),
                          source_term=source_for(spec, side),
                          coefficient=coefficient_for(spec, side),
                          domain=domain_for(spec, side),
                          equation=str(spec.get("equation", "")))
            verdicts[side] = verdict_of(reply)
        if not n_files:
            rows.append(dict(problem=problem, seed=seed, outcome=outcome,
                             verdict="NO_SOLUTION_FILES", files=0))
            continue
        combined = ("REFUSED" if "REFUSED" in verdicts.values()
                    else "INCONSISTENT" if "INCONSISTENT" in verdicts.values()
                    else "CONSISTENT" if set(verdicts.values()) == {"CONSISTENT"}
                    else "/".join(f"{s}:{v}" for s, v in verdicts.items()))
        rows.append(dict(problem=problem, seed=seed, outcome=outcome,
                         verdict=combined, files=n_files,
                         per_side=verdicts))

    print(f"{'problem':8s} {'seed':6s} {'graded outcome':22s} {'check says':18s} files")
    for r in sorted(rows, key=lambda r: (r["problem"], r["seed"])):
        print(f"{r['problem']:8s} {r['seed']:6s} {r['outcome']:22s} "
              f"{r['verdict']:18s} {r['files']}")

    print(f"\n{len(rows)} cell(s)")
    from collections import Counter
    pair = Counter((r["outcome"], r["verdict"]) for r in rows)
    print("\noutcome x verdict:")
    for (o, v), n in sorted(pair.items()):
        print(f"  {n:4d}  {o:22s} -> {v}")

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nrows written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
