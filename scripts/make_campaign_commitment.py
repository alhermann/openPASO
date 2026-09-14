#!/usr/bin/env python3
"""Re-make the campaign pre-registration commitment, completely, by script.

WHY A SCRIPT
------------
The previous commitment (2026-08-07) was assembled by hand and went stale
without anyone noticing: 10 of 35 entries drifted, 3 went missing, and all
seven B task texts changed CONTENT at identical byte length — the case only a
hash catches. A pre-registration that does not match the tree is worse than
none, because it testifies falsely. Regenerating by script makes re-making it
one command and makes "what is frozen" a decision recorded here rather than a
memory.

WHAT IS COMMITTED (the audit's freeze list, in full)
----------------------------------------------------
 * the design and every amendment
 * the RUNNER, the grader (v2 package), the PHASE GATE, both BUILDERS
 * every task text and every public spec, all 47 problems
 * the evidence-thresholds tree src/blind_eval/ — the audit found grading
   executed code from a checkout the freeze never touched; it is THIS tree now
   and the commitment proves it
 * the key COMMITMENT: sha256 of every sealed key file (hash of ciphertext or
   plaintext as found — the point is immutability, not secrecy of the hash)
 * model pinning: the exact model ids, temperatures, provider routing, token
   and wall-clock budgets, read from the runner and agent source so the
   commitment cannot disagree with the code
 * the draw seed for the evaluation phase, passed in and recorded
 * the git commit of this tree at commitment time

Verification is the same command with --verify.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "blind_campaign_commitment.json"

# NOT COMMITTED, DELIBERATELY: campaign3_blind/CONVERGENCE.md and
# DEV_FINDINGS.md. They are running logs — every defect closed appends to
# them — so hashing them makes --verify report drift after ordinary
# note-taking. An integrity check that cries wolf on every edit is one nobody
# reads, and then real drift goes past. The commitment covers what can change
# a RESULT: tasks, keys, grader, runner, agent, served knowledge.
FILES = [
    "campaign3_blind/DESIGN.md",
    "campaign3_blind/run_blind.py",
    "campaign3_blind/grade_blind_v2.py",
    "campaign3_blind/grade_blind.py",
    "campaign3_blind/phase.py",
    "campaign3_blind/build_coupled_v2.py",
    "campaign3_blind/build_balanced.py",
    "campaign3_blind/build_single_v2.py",
    "campaign3_blind/build_offpool.py",
    "campaign3_blind/path_readiness.json",
    "campaign3_blind/dev_driver.sh",
    "scripts/blind_grade.py",
    # THE AGENT DEFINES BOTH ARMS — the bare tool set, the MCP tool set, both
    # system prompts, the sandbox confinement. It was NOT committed, and it
    # changed one minute after a commitment was generated. Without it, the
    # treatment itself could be edited after the freeze and --verify would
    # still report clean.
    "langgraph_eval/agent.py",
    "src/server.py",
]
GLOBS = [
    "campaign3_blind/grading/*.py",
    "campaign3_blind/problems/*/task.txt",
    "campaign3_blind/problems/*/spec_public.json",
    "src/blind_eval/*.py",
    "src/blind_eval/**/*.py",
    # THE MCP ARM'S ENTIRE TREATMENT. The knowledge served to the openPASO arm is
    # the independent variable of this experiment; none of it was hashed, so
    # "which knowledge produced these numbers" had no answer.
    "src/tools/*.py",
    "src/backends/**/*.py",
    "src/backends/**/*.json",
    "src/core/*.py",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _entry(p: Path) -> dict:
    return {"path": str(p.relative_to(REPO)), "bytes": p.stat().st_size,
            "sha256": _sha(p)}


def _pinning() -> dict:
    """Read the pins out of the code, so this file cannot disagree with it."""
    run = (REPO / "campaign3_blind" / "run_blind.py").read_text()
    models = sorted(set(re.findall(r'"(qwen/[a-z0-9.\-]+)"', run)))
    agent_paths = [REPO / "langgraph_eval" / "agent.py",
                   Path("/home/alexander/Schreibtisch/open-fem-agent/"
                        "langgraph_eval/agent.py")]
    temps = []
    for ap in agent_paths:
        if ap.is_file():
            temps = sorted(set(re.findall(r"temperature\s*=\s*([0-9.]+)",
                                          ap.read_text())))
            agent_file = str(ap)
            break
    else:
        agent_file, temps = "NOT FOUND", []
    timeout = re.search(r'"--timeout".*?default=(\d+)', run, re.S)
    return {
        "models": models,
        "agent_file": agent_file,
        "temperatures": temps,
        "timeout_default_s": int(timeout.group(1)) if timeout else None,
        "endpoint": "openrouter.ai/api/v1",
        "note": ("provider/quantization routing is NOT pinned by OpenRouter "
                 "by default; the campaign driver must set provider "
                 "preferences per request, and the per-run ledger records "
                 "the provider actually used for every call"),
    }


def collect(keys_dir: Path | None, seed: int | None) -> dict:
    files = [REPO / f for f in FILES]
    for g in GLOBS:
        files += sorted(REPO.glob(g))
    seen, entries = set(), []
    for p in files:
        if p.is_file() and p not in seen:
            seen.add(p)
            entries.append(_entry(p))
    keys = []
    if keys_dir and keys_dir.is_dir():
        for kf in sorted(keys_dir.rglob("*")):
            if kf.is_file():
                keys.append({"path": str(kf.relative_to(keys_dir)),
                             "bytes": kf.stat().st_size, "sha256": _sha(kf)})
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()
    return {
        "schema": 2,
        "campaign": "campaign3_blind",
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "git_commit": head,
        "draw_seed": seed,
        "pinning": _pinning(),
        "entries": entries,
        "key_commitment": keys,
        "note": ("Regenerate ONLY via scripts/make_campaign_commitment.py, "
                 "and only before the freeze marker exists. After the freeze "
                 "any change here invalidates the campaign."),
    }


def verify(doc: dict, keys_dir: Path | None) -> int:
    bad = 0
    for e in doc["entries"]:
        p = REPO / e["path"]
        if not p.is_file():
            print(f"  MISSING  {e['path']}"); bad += 1
        elif _sha(p) != e["sha256"]:
            print(f"  DRIFTED  {e['path']}"); bad += 1
    if keys_dir and keys_dir.is_dir():
        for e in doc.get("key_commitment", []):
            p = keys_dir / e["path"]
            if not p.is_file():
                print(f"  KEY MISSING  {e['path']}"); bad += 1
            elif _sha(p) != e["sha256"]:
                print(f"  KEY DRIFTED  {e['path']}"); bad += 1
    print(f"  entries {len(doc['entries'])}, keys "
          f"{len(doc.get('key_commitment', []))}, problems {bad}")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys", type=Path,
                    default=Path(os.environ.get("OPENPASO_BLIND_KEYS", "")))
    ap.add_argument("--seed", type=int, default=None,
                    help="evaluation draw seed to record (required to write)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    keys = a.keys if str(a.keys) else None
    if a.verify:
        return verify(json.loads(OUT.read_text()), keys)
    if a.seed is None:
        print("refusing to write without --seed: the draw seed is part of "
              "the pre-registration")
        return 2
    doc = collect(keys, a.seed)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"  written {OUT.relative_to(REPO)}: {len(doc['entries'])} files, "
          f"{len(doc['key_commitment'])} keys, commit "
          f"{doc['git_commit'][:12]}, seed {a.seed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
