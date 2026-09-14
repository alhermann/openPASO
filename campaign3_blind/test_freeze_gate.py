"""Does the widened freeze gate actually catch what it now claims to catch?

The freeze exists because knowledge could otherwise be changed in response to
the problems being graded. So the test that matters is: freeze, then tamper,
then verify — and the verify must FAIL. A gate that passes after tampering is
worse than no gate, because it certifies something it never checked.

Cases:
  1. freeze, verify immediately            -> must pass the tree check
  2. edit a served knowledge file          -> must fail
  3. edit the grader                       -> must fail
  4. change the output-token cap           -> must fail the pin check
Every edit is reverted afterwards, byte for byte.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/alexander/Schreibtisch/ofa-v2")
FREEZE = ROOT / "campaign3_blind" / "freeze.py"
MARKER = ROOT / "campaign3_blind" / "FROZEN.json"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
ENV = dict(os.environ,
           OPENPASO_BLIND_KEYS="/home/alexander/Schreibtisch/qwen_uplift_test/"
                            "campaign3_blind/keys")

TARGETS = {
    "served knowledge": ROOT / "src" / "tools" / "coupling_knowledge.py",
    "the grader": ROOT / "campaign3_blind" / "grade_blind.py",
    "the output cap": ROOT / "campaign3_blind" / "run_blind.py",
}


def verify():
    r = subprocess.run([PY, str(FREEZE), "--verify"], cwd=ROOT, env=ENV,
                       capture_output=True, text=True)
    return r.returncode, r.stdout


had_marker = MARKER.is_file()
backup = MARKER.read_bytes() if had_marker else None
originals = {k: p.read_bytes() for k, p in TARGETS.items() if p.is_file()}
failures = []
try:
    r = subprocess.run([PY, str(FREEZE), "--draw-seed", "999", "--force"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True)
    if not MARKER.is_file():
        print("could not write a marker:", (r.stderr or r.stdout)[-300:])
        sys.exit(1)
    m = json.loads(MARKER.read_text())
    print(f"  marker written; trees recorded: "
          f"{sorted(k for k in (m.get('trees') or {}) if k != 'combined')}")
    print(f"  max_output_tokens pinned: "
          f"{m.get('pinning', {}).get('max_output_tokens')}")
    print(f"  endpoint pinned: {m.get('pinning', {}).get('endpoint')}")

    rc, out = verify()
    base_ok = "OK    served knowledge, participants and grader unchanged" in out
    print(f"\n  1. verify straight after freezing -> "
          f"{'PASS' if base_ok else 'FAIL'}")
    if not base_ok:
        failures.append("clean verify did not pass the tree check")

    for label, path in TARGETS.items():
        if label not in originals:
            continue
        path.write_bytes(originals[label] + b"\n# tamper\n")
        rc, out = verify()
        caught = rc != 0
        which = ("pin unchanged: max_output_tokens" if label == "the output cap"
                 else "served knowledge, participants and grader unchanged")
        named = any(line.startswith("  FAIL") and which in line
                    for line in out.splitlines())
        path.write_bytes(originals[label])
        print(f"  2. tamper with {label:<18} -> "
              f"{'CAUGHT' if caught else 'MISSED'}"
              f"{'' if named or not caught else ' (but not by the expected check)'}")
        if not caught:
            failures.append(f"tampering with {label} was not caught")

    rc, out = verify()
    print(f"\n  3. verify after reverting everything -> "
          f"{'clean' if rc == 0 else 'STILL FAILING'}")
    if rc != 0:
        failures.append("verify did not return to clean after revert")
        print("\n".join(out.splitlines()[-6:]))
finally:
    for k, p in TARGETS.items():
        if k in originals:
            p.write_bytes(originals[k])
    if had_marker:
        MARKER.write_bytes(backup)
    elif MARKER.is_file():
        MARKER.unlink()

print()
print("  ALL CHECKS PASSED" if not failures else f"  FAILURES: {failures}")
sys.exit(1 if failures else 0)
