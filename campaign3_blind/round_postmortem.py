"""Where a round's cells actually stopped, from their own trajectories and workspaces.

Usage: python round_postmortem.py <seed> [seed ...]

Round 49 needed this and it was done by hand: the answer ("side A ran in seven of nine cells, side B in
two, and only one cell reached couple()") is what turned a 0/9 into a diagnosis. Campaign-side by
design; it reads only what the cells wrote.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"


def cell_dir(seed: int):
    hits = sorted(RUNS.glob(f"*_27b_MCP_seed{seed}"))
    return hits[0] if hits else None


def look(seed: int) -> dict:
    d = cell_dir(seed)
    if d is None:
        return {"seed": seed, "missing": True}
    w = d / "work"
    traj = (w / "trajectory_live.txt")
    s = traj.read_text(errors="ignore") if traj.is_file() else ""
    exports = {p.parent.name for p in w.rglob("exports.json")}
    side_a = any(e.endswith("A") or "_A" in e for e in exports)
    side_b = any(e.endswith("B") or "_B" in e for e in exports)
    ledger = {}
    lp = d / "ledger.json"
    if lp.is_file():
        try:
            ledger = json.loads(lp.read_text())
        except Exception:                                  # noqa: BLE001
            ledger = {}
    calls = ledger.get("tool_calls") or len(re.findall(r"TOOL_CALL ", s))
    wall = ledger.get("wall_s") or 0
    result = (w / "RESULT.txt")
    head = result.read_text(errors="ignore").splitlines()[0].strip() if result.is_file() else ""
    gates = {"write": s.count("[write check]"), "run": s.count("[run check]"),
             "deck": s.count("4C DECK")}
    return {"seed": seed, "problem": d.name.split("_")[0], "calls": calls, "wall": wall,
            "s_per_call": (wall / calls) if calls else 0,
            "couple": len(re.findall(r"TOOL_CALL couple", s)),
            "sideA": side_a, "sideB": side_b,
            "csv": len(list(w.rglob("*.csv"))), "logs": len(list(w.rglob("run_level*.log"))),
            "result": head[:40], "gates": gates,
            "physics": sorted(set(re.findall(r"'physics': '([^']+)'", s))),
            "stopped": ("both sides up" if side_a and side_b else
                        "second side never ran" if side_a or side_b else "no side ran")}


def main(seeds) -> int:
    rows = [look(int(x)) for x in seeds]
    print(f"{'cell':14} {'calls':>5} {'s/call':>7} {'A':>2} {'B':>2} {'cpl':>4} {'csv':>4} "
          f"{'logs':>5}  gates(w/r/d)  where it stopped / RESULT")
    for r in rows:
        if r.get("missing"):
            print(f"seed {r['seed']}: no run directory")
            continue
        g = r["gates"]
        print(f"{r['problem']+' '+str(r['seed']):14} {r['calls']:5} {r['s_per_call']:7.1f} "
              f"{'y' if r['sideA'] else '-':>2} {'y' if r['sideB'] else '-':>2} {r['couple']:4} "
              f"{r['csv']:4} {r['logs']:5}  {g['write']:>2}/{g['run']:>2}/{g['deck']:>2}      "
              f"{r['stopped']}{' | ' + r['result'] if r['result'] else ''}")
    live = [r for r in rows if not r.get("missing")]
    if live:
        both = sum(1 for r in live if r["sideA"] and r["sideB"])
        cpl = sum(1 for r in live if r["couple"])
        sec = [r["s_per_call"] for r in live if r["s_per_call"]]
        print(f"\n{len(live)} cells: both sides ran in {both}, couple() reached in {cpl}; "
              f"per-call cost {min(sec):.0f}-{max(sec):.0f} s" if sec else "")
        phys = sorted({p for r in live for p in r["physics"]})
        print(f"physics words asked: {phys or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or []))
