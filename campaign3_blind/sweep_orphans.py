"""Find solvers that outlived the run that started them.

Twice now a run has finished, written its ledger, and left a solver behind at
100% of a core: three found 23-24 h later costing 71 CPU-hours (fa3cf276), and
one found 7 h 10 min after C10_27b_BARE_seed3 finished. The mechanism is the
same both times — the agent's shell hits its timeout, the shell dies, and the
process it started does not.

An orphan cannot corrupt a result. The ledger is already written and the grader
reads files, not processes. What it does is quieter: it takes a core from
whatever runs next, so wall-clock and timeout counts drift between rounds for a
reason that has nothing to do with the models. That is the kind of drift that
makes two rounds uncomparable without anyone noticing.

SAFETY. This never kills a process belonging to a live run. A candidate must
satisfy all of:
  * its cwd is inside campaign3_blind/runs/<cell>/,
  * that cell's ledger.json exists, so its run is over,
  * no live run_blind.py is working that same cell,
  * it is not this script and not an ancestor of it.

Default is a dry run. Pass --kill to act.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from pathlib import Path

# Overridable so the firing test below can run against a scratch tree
# instead of the live campaign — a sweep that has never been shown to
# FIRE is worth nothing, and proving it fires must not mean planting a
# fake process in the real runs/ directory while a round is going.
RUNS = Path(os.environ.get("OPENPASO_RUNS_DIR",
                           Path(__file__).resolve().parent / "runs"))


def live_cells() -> set[str]:
    """Cells a run_blind.py process is working on RIGHT NOW."""
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    cells = set()
    for line in out.stdout.splitlines():
        if "run_blind.py" not in line or "--model" not in line:
            continue
        prob = re.search(r"--problems\s+(\S+)", line)
        seed = re.search(r"--seed\s+(\d+)", line)
        model = re.search(r"--model\s+(\S+)", line)
        cond = re.search(r"--conditions\s+(\S+)", line)
        if prob and seed and model and cond:
            cells.add(f"{prob.group(1)}_{model.group(1)}_"
                      f"{cond.group(1)}_seed{seed.group(1)}")
    return cells


def candidates(busy: set[str]):
    """Processes whose cwd sits inside a FINISHED run directory."""
    me = os.getpid()
    ancestors = set()
    p = me
    while p and p != 1:
        ancestors.add(p)
        try:
            p = int(Path(f"/proc/{p}/stat").read_text().split()[3])
        except Exception:
            break
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid in ancestors:
            continue
        try:
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if str(RUNS) not in cwd:
            continue
        rel = Path(cwd).relative_to(RUNS)
        cell = rel.parts[0] if rel.parts else ""
        if not cell or cell in busy:
            continue
        if not (RUNS / cell / "ledger.json").is_file():
            continue          # run still in progress: leave it alone
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
            cmd = cmd.replace(b"\0", b" ").decode(errors="replace").strip()
            stat = Path(f"/proc/{pid}/stat").read_text().split()
            ticks = int(stat[13]) + int(stat[14])
            cpu_s = ticks / os.sysconf("SC_CLK_TCK")
        except Exception:
            continue
        found.append({"pid": pid, "cell": cell, "cpu_s": cpu_s,
                      "cmd": cmd[:100]})
    return found


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kill", action="store_true",
                    help="actually terminate them (default reports only)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    busy = live_cells()
    found = candidates(busy)
    if a.json:
        print(json.dumps({"live_cells": sorted(busy), "orphans": found},
                         indent=2))
    else:
        print(f"live run_blind cells: {len(busy)}")
        if not found:
            print("no orphans: every process under runs/ belongs to a live run")
        for o in found:
            print(f"  pid {o['pid']:>8}  {o['cell']:<28} "
                  f"{o['cpu_s']/3600:6.2f} CPU-h  {o['cmd']}")
        total = sum(o["cpu_s"] for o in found) / 3600
        if found:
            print(f"  total wasted: {total:.2f} CPU-hours")

    if a.kill:
        for o in found:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.kill(o["pid"], sig)
                except ProcessLookupError:
                    break
        print(f"killed {len(found)} orphan(s)")
    elif found:
        print("dry run: pass --kill to terminate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
