#!/usr/bin/env python3
"""Write the freeze marker — the moment development stops and evaluation may begin.

WHY THIS IS A SCRIPT AND NOT A FILE SOMEBODY WRITES

`phase.assert_evaluation_is_clean` refuses to grade an evaluation without a
marker at campaign3_blind/FROZEN.json, because without one "held out" means
nothing: the knowledge could have been changed in response to the very
problems being graded. But nothing in the repository ever WROTE that marker.
It was going to be hand-authored, and the gate only checked that it existed
and carried a non-null draw seed — so a marker asserting a commit that was
never built, model pins that no run used, or a commitment that had since
drifted would have passed.

This script records what the claim actually depends on, measured at the moment
of freezing:

  * the git commit, and whether the tree was dirty (a dirty tree means the
    frozen state is not the committed state, which is worth knowing later)
  * the sha256 of data/blind_campaign_commitment.json, so a later reader can
    tell whether the pre-registration is the one the freeze covered
  * the model ids and temperatures actually pinned in the runner and agent
  * the evaluation draw seed
  * the development instances that are spent at this moment, by id and by
    task-text fingerprint, so an evaluation draw cannot launder one by rename
  * the seal state of the key vault

Usage:
    python campaign3_blind/freeze.py --draw-seed 12345
    python campaign3_blind/freeze.py --verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MARKER = HERE / "FROZEN.json"
COMMITMENT = ROOT / "data" / "blind_campaign_commitment.json"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else ""


def _pins() -> dict:
    run = (HERE / "run_blind.py").read_text()
    models = sorted(set(re.findall(r'"(qwen/[a-z0-9.\-]+)"', run)))
    temps = sorted(set(re.findall(r"temperature\s*=\s*([0-9.]+)",
                                  (ROOT / "langgraph_eval" / "agent.py").read_text())))
    timeout = re.search(r'"--timeout".*?default=(\d+)', run, re.S)
    recursion = re.search(r"RECURSION_LIMIT\s*=\s*(\d+)", run)
    # The output cap is part of the budget, not a detail: every request
    # reserves it from the same context window the history lives in, so
    # changing it changes how many turns an agent gets before the provider
    # refuses. It was unpinned (the provider's own 65536) until 2026-08-18.
    max_out = re.search(r"_MAX_OUT\s*=\s*(\d+)", run)
    # Which endpoint served the request. The model id is not sufficient: the
    # same id behind a different upstream provider is a different measurement,
    # and one provider's context accounting is what killed FC2 BARE seed3.
    base = re.search(r'base_url="([^"]+)"', run)
    return {"models": models, "temperatures": temps,
            "timeout_default_s": int(timeout.group(1)) if timeout else None,
            "recursion_limit": int(recursion.group(1)) if recursion else None,
            "max_output_tokens": int(max_out.group(1)) if max_out else None,
            "endpoint": base.group(1) if base else None}


def _tree_digest(*rels: str) -> dict:
    """sha256 over every file under the given paths, plus a combined digest.

    THE KNOWLEDGE IS THE SYSTEM UNDER TEST. The whole reason a freeze exists,
    in this file's own words, is that "the knowledge could have been changed
    in response to the very problems being graded" — and until now nothing
    here hashed it. git_commit only helps if the change was committed; an
    uncommitted edit to a served payload between the freeze and the
    evaluation would have passed every check.

    The grader is included for the same reason with more force: it decides
    every outcome, so a grader edited after the freeze can move the numbers
    without touching a single run.
    """
    out, h = {}, hashlib.sha256()
    for rel in rels:
        base = ROOT / rel
        if base.is_file():
            files = [base]
        elif base.is_dir():
            files = sorted(f for f in base.rglob("*")
                           if f.is_file() and "__pycache__" not in f.parts)
        else:
            out[rel] = "MISSING"
            continue
        d = hashlib.sha256()
        for f in files:
            d.update(f.relative_to(ROOT).as_posix().encode())
            d.update(f.read_bytes())
        out[rel] = d.hexdigest()
        h.update(out[rel].encode())
    out["combined"] = h.hexdigest()
    return out


# What must not move between the freeze and the evaluation. Served knowledge,
# the participant scripts that knowledge hands out, and the grader.
_FROZEN_TREES = ("src/tools", "src/backends", "src/core",
                 "data/coupling_participants",
                 "campaign3_blind/grade_blind.py",
                 "campaign3_blind/grade_blind_v2.py",
                 "campaign3_blind/run_blind.py")


def _vault_state() -> str:
    try:
        from blind_eval import keyvault as kv
        import os
        d = os.environ.get("OPENPASO_BLIND_KEYS")
        return kv.seal_state(Path(d)) if d else "OPENPASO_BLIND_KEYS unset"
    except Exception as e:                                    # noqa: BLE001
        return f"unavailable: {type(e).__name__}"


def build(draw_seed: int) -> dict:
    import phase
    return {
        "schema": 1,
        "frozen_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        # Tracked modifications only. `git status --porcelain` also lists
        # UNTRACKED files, and this tree always has some — a driver written
        # for the round in flight, a quarantine directory, solver output. On
        # the old test every freeze was therefore "dirty" and the signal meant
        # nothing. What invalidates a freeze is a tracked file differing from
        # the commit the marker names; untracked paths are recorded instead of
        # judged, so a reader can see whether one of them looks like a served
        # payload rather than scratch.
        "git_dirty": bool([ln for ln in _git("status", "--porcelain").splitlines()
                           if not ln.startswith("??")]),
        "untracked": [ln[3:] for ln in _git("status", "--porcelain").splitlines()
                      if ln.startswith("??")],
        "commitment_sha256": _sha(COMMITMENT),
        "commitment_entries": len(json.loads(COMMITMENT.read_text())["entries"])
        if COMMITMENT.is_file() else 0,
        "pinning": _pins(),
        "draw_seed": draw_seed,
        "key_vault": _vault_state(),
        "development_spent": sorted(phase.DEVELOPMENT),
        "development_fingerprints": phase.development_fingerprints(),
        "trees": _tree_digest(*_FROZEN_TREES),
    }


def verify() -> int:
    if not MARKER.is_file():
        print(f"  no marker at {MARKER.name} — not frozen")
        return 1
    m = json.loads(MARKER.read_text())
    bad = 0

    def check(label: str, ok: bool, detail: str = "") -> None:
        nonlocal bad
        print(f"  {'OK  ' if ok else 'FAIL'}  {label}{': ' + detail if detail else ''}")
        bad += (not ok)

    check("marker carries a draw seed", m.get("draw_seed") is not None)
    check("commitment unchanged since the freeze",
          m.get("commitment_sha256") == _sha(COMMITMENT),
          "regenerating the pre-registration after freezing invalidates it")
    now = _pins()
    for k in ("models", "temperatures", "recursion_limit", "timeout_default_s"):
        check(f"pin unchanged: {k}", m.get("pinning", {}).get(k) == now.get(k),
              f"frozen {m.get('pinning', {}).get(k)} vs now {now.get(k)}")
    import phase
    check("spent list unchanged",
          sorted(m.get("development_spent", [])) == sorted(phase.DEVELOPMENT),
          "an instance added or removed after the freeze")
    fp_now = phase.development_fingerprints()
    drifted = [k for k, v in m.get("development_fingerprints", {}).items()
               if fp_now.get(k) != v]
    check("development task texts unchanged", not drifted,
          f"{len(drifted)} drifted: {drifted[:5]}")
    check("the freeze was taken on a clean tree", not m.get("git_dirty"),
          "frozen from a dirty working tree")
    # The one that matters most: knowledge, participants and grader unchanged.
    frozen_trees = m.get("trees") or {}
    if not frozen_trees:
        check("marker records tree digests", False,
              "marker predates tree hashing — re-freeze before evaluating")
    else:
        now_trees = _tree_digest(*_FROZEN_TREES)
        moved = [k for k, v in frozen_trees.items()
                 if k != "combined" and now_trees.get(k) != v]
        check("served knowledge, participants and grader unchanged",
              not moved, f"{len(moved)} changed: {moved}")
    print(f"  frozen at {m.get('frozen_utc')} on {str(m.get('git_commit'))[:12]}"
          f", {bad} problem(s)")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--draw-seed", type=int, default=None)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing marker (re-freezing means the "
                         "previous freeze claim was wrong; say so out loud)")
    a = ap.parse_args()

    if a.verify:
        return verify()
    if a.draw_seed is None:
        print("refusing to freeze without --draw-seed: the evaluation draw "
              "seed is part of what the freeze commits to")
        return 2
    if MARKER.is_file() and not a.force:
        print(f"refusing: {MARKER.name} already exists. A campaign freezes "
              f"once; re-freezing means the earlier claim was wrong. Pass "
              f"--force if that is genuinely the case.")
        return 2
    doc = build(a.draw_seed)
    if not doc["git_commit"]:
        print("refusing: cannot read the git commit, so the marker could not "
              "say what was frozen")
        return 2
    MARKER.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"  FROZEN at {doc['frozen_utc']}")
    print(f"    commit          {doc['git_commit'][:12]}"
          f"{'  (DIRTY TREE)' if doc['git_dirty'] else ''}")
    print(f"    commitment      {doc['commitment_sha256'][:12]}"
          f" over {doc['commitment_entries']} files")
    print(f"    models          {doc['pinning']['models']}")
    print(f"    draw seed       {doc['draw_seed']}")
    print(f"    spent instances {len(doc['development_spent'])}")
    print(f"    key vault       {doc['key_vault']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
