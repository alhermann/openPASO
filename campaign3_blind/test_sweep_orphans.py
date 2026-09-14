"""Does the orphan sweep actually fire, and does it spare what it must?

Three cases against a scratch runs/ tree, so nothing here can touch round 4:
  1. a process in a FINISHED cell (ledger present)   -> must be flagged
  2. a process in an UNFINISHED cell (no ledger)     -> must be spared
  3. a process in a cell a live run_blind is on      -> must be spared

Case 3 is simulated by starting a fake `run_blind.py --model ...` process whose
command line names that cell, since the sweep decides "live" by reading ps.
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "sweep_runs"
SWEEP = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/sweep_orphans.py"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"

shutil.rmtree(RUNS, ignore_errors=True)
cells = {
    "AA1_27b_BARE_seed9": True,    # finished: ledger present
    "BB1_27b_BARE_seed9": False,   # still running: no ledger
    "CC1_27b_BARE_seed9": True,    # finished, but a live run_blind names it
}
for cell, finished in cells.items():
    (RUNS / cell / "work").mkdir(parents=True)
    if finished:
        (RUNS / cell / "ledger.json").write_text('{"problem": "x"}')

procs = []
for cell in cells:
    procs.append(subprocess.Popen(["sleep", "600"], cwd=RUNS / cell / "work"))

# a fake live runner for CC1 — the sweep reads ps to decide what is in flight
fake = HERE / "fake_run_blind.py"
fake.write_text("import time\ntime.sleep(600)\n")
live = subprocess.Popen(
    [PY, str(fake), "run_blind.py", "--model", "27b", "--conditions", "BARE",
     "--problems", "CC1", "--seed", "9"], cwd=HERE)
time.sleep(2)

env = dict(os.environ, OPENPASO_RUNS_DIR=str(RUNS))
out = subprocess.run([PY, SWEEP], env=env, capture_output=True, text=True)
print(out.stdout.rstrip())
flagged = {c for c in cells if c in out.stdout}

expected = {"AA1_27b_BARE_seed9"}
print()
print(f"  flagged : {sorted(flagged) or 'none'}")
print(f"  expected: {sorted(expected)}")
ok = flagged == expected
print(f"  {'PASS — fires on the orphan, spares the other two' if ok else 'FAIL'}")

# and does --kill actually kill it?
if ok:
    before = procs[0].poll()
    subprocess.run([PY, SWEEP, "--kill"], env=env, capture_output=True,
                   text=True)
    time.sleep(2)
    after = procs[0].poll()
    print(f"  --kill: orphan alive before={before is None} "
          f"after={after is None}  "
          f"{'PASS' if before is None and after is not None else 'FAIL'}")
    still = procs[1].poll()
    print(f"  spared process still alive after --kill: {still is None}  "
          f"{'PASS' if still is None else 'FAIL'}")

for p in procs + [live]:
    try:
        p.send_signal(signal.SIGKILL)
    except Exception:
        pass
shutil.rmtree(RUNS, ignore_errors=True)
fake.unlink(missing_ok=True)
sys.exit(0 if ok else 1)
