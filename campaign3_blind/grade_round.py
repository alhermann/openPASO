"""Grade every run of a round, inside one sealed window, reproducibly.

Rounds 1-3 were graded ad hoc from the shell. That is how a grading pass
becomes unrepeatable: nobody can tell later which runs were included, whether
the keys were sealed before and after, or whether a cell was graded twice with
different code. This script is the procedure, so the next round is graded the
same way as this one.

CUSTODY. The answer keys are chmod-000 while agents run and must be sealed
again the moment grading ends. This script:
  * refuses to start if the keys are not SEALED (if they were already open, the
    campaign's custody claim is already broken and unsealing further hides it),
  * unseals, grades every run, and reseals in a `finally` block so a crash or
    a KeyboardInterrupt cannot leave them readable,
  * verifies the seal state afterwards and reports it, non-zero on failure.

The passphrase is read once from the terminal and passed in memory. It is
never written to disk, never placed in argv where ps would show it, and never
echoed.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

CELLS = ["FE1", "FE2", "DL1", "DL2", "NG1", "NG2", "SK1", "SK2",
         "KR1", "KR2", "DU1", "DU2", "FB1", "FB2", "FC1", "FC2",
         "SP1", "SP2"] + [f"C{i}" for i in range(1, 15)]


def keys_dir() -> Path:
    d = os.environ.get("OPENPASO_BLIND_KEYS")
    if not d:
        sys.exit("OPENPASO_BLIND_KEYS is unset: refusing to guess where the "
                 "answer keys live")
    return Path(d)


def seal_state(d: Path) -> str:
    return subprocess.run(["stat", "-c", "%A", str(d)], capture_output=True,
                          text=True).stdout.strip()


def shield(action: str) -> str:
    script = HERE / "shield_keys.sh"
    if not script.is_file():
        sys.exit(f"no shield script at {script}")
    r = subprocess.run(["bash", str(script), action], capture_output=True,
                       text=True)
    output = (r.stdout + r.stderr).strip()
    if r.returncode != 0:
        raise RuntimeError(
            f"shield_keys.sh {action} failed ({r.returncode}): {output}")
    return output


def arm_reseal_on_signals() -> None:
    """Reseal on SIGTERM/SIGINT/SIGHUP, not only on the way out of `try`.

    The module docstring promised that "a crash or a KeyboardInterrupt cannot
    leave them readable", and the mechanism was a `finally` block. A finally
    block does not run when the process is killed: Python's default SIGTERM
    disposition terminates immediately. That is the same shape as the four
    other defects this campaign has found -- a mechanism exists, is
    instrumented, and does not reach the case it was built for.

    It reached that case on 2026-08-30. A regrade of seeds 2-11 was launched
    under a two-minute harness limit, the limit killed it with SIGTERM
    mid-grade, the finally block never ran, and the vault was found
    drwxr-xr-x. Nothing could read it -- the round had finished and no agent
    process existed -- but the custody claim was false for about two minutes,
    and it was false without anyone being told.

    Handlers reseal and then re-raise the default disposition, so the exit
    status still reports a killed process rather than a clean one.
    """
    import signal

    def _handler(signum, _frame):
        try:
            shield("seal")
            state = seal_state(keys_dir())
            print(f"\n  signal {signum}: {shield_note(state)}", flush=True)
        finally:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass          # not on the main thread, or not supported here


def shield_note(state: str) -> str:
    return (f"RESEALED {state}" if state == "d---------"
            else f"FAILED TO RESEAL, keys are {state} — seal them by hand")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--model", default="27b")
    ap.add_argument("--out", required=True,
                    help="grades json, one file per seed: <out>_seed<N>.json")
    ap.add_argument("--with-key", action="store_true",
                    help="keys on disk are encrypted; prompt for the phrase")
    a = ap.parse_args()

    kd = keys_dir()
    before = seal_state(kd)
    if before != "d---------":
        print(f"REFUSING: keys are not sealed ({before}). If they were left "
              f"open, custody is already in question and that must be "
              f"recorded, not papered over by unsealing again.")
        return 2
    try:
        shield("status")
    except RuntimeError as exc:
        print(f"REFUSING: a sibling key store is open or the repository "
              f"shield failed: {exc}")
        return 2
    print(f"  keys before: {before}  SEALED")

    # Read from a terminal when there is one. When there is not — grading
    # driven from an automation context — read ONE line from stdin instead.
    # Never from argv (ps would show it), never from a file, never from the
    # environment: the vault's whole claim is that the phrase is on no disk
    # while runs happen.
    pw = None
    if a.with_key:
        if sys.stdin.isatty():
            pw = getpass.getpass("key passphrase: ")
        else:
            pw = sys.stdin.readline().rstrip("\n")
            if not pw:
                print("no passphrase on stdin"); return 2

    arm_reseal_on_signals()

    import grade_blind_v2 as G
    runs = HERE / "runs"
    written = {}
    try:
        print(" ", shield("unseal"))
        for seed in a.seeds:
            grades, missing = {}, []
            for cell in CELLS:
                for arm in ("BARE", "MCP"):
                    d = runs / f"{cell}_{a.model}_{arm}_seed{seed}"
                    if not (d / "ledger.json").is_file():
                        missing.append(d.name)
                        continue
                    try:
                        grades[f"{cell}_{arm}"] = G.grade_run(
                            d, cell, passphrase=pw)
                    except Exception as e:               # noqa: BLE001
                        # A grader exception is recorded, never silently
                        # dropped: a missing verdict would otherwise read as
                        # a failed run.
                        grades[f"{cell}_{arm}"] = {
                            "problem": cell,
                            "outcome": "GRADER_EXCEPTION",
                            "error": f"{type(e).__name__}: {e}"[:300]}
            out = Path(f"{a.out}_seed{seed}.json")
            out.write_text(json.dumps(grades, indent=2, default=str))
            written[seed] = (out, len(grades), missing)
            print(f"  seed {seed}: graded {len(grades)} cells -> {out.name}"
                  + (f"   MISSING {len(missing)}: {missing[:4]}"
                     if missing else ""))
    finally:
        print(" ", shield("seal"))
        after = seal_state(kd)
        print(f"  keys after: {after}"
              f"  {'SEALED' if after == 'd---------' else 'NOT SEALED'}")
        if after != "d---------":
            print("  FAILURE: keys did not return to sealed")
            return 3

    exc = sum(1 for _, (o, _, _) in written.items()
              for v in json.loads(o.read_text()).values()
              if v.get("outcome") == "GRADER_EXCEPTION")
    if exc:
        print(f"  WARNING: {exc} grader exception(s) recorded — these are OUR "
              f"failures, not the model's, and must be resolved before the "
              f"numbers are quoted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
