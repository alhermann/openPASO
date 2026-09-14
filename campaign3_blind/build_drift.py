"""Which runs of a round were served a DIFFERENT build from the others?

WHY THIS EXISTS. On 2026-09-01 three commits to `src/` landed while round 9 was
in flight. Each `run_blind.py` invocation is a fresh process that imports the
openPASO tools from the working tree, so runs that started after a commit were
served knowledge the earlier runs never saw: 16 of the round's 64 MCP runs
finished before 14:58:23 and got the pre-change core, the rest got rule 7 (the
halvings-not-cells refinement rule) and the mesh-size audit finding.

That makes the round's openPASO number a MIXTURE of two builds. The bare arm is
unaffected — it calls no openPASO tool — so the damage is one-sided, which is the
worse kind: it moves the uplift without moving the control.

Current ledgers carry the SHA-256 of the immutable source snapshot actually
mounted into the run. This script groups on that field. Older ledgers carry no
such proof and are reported as ``LEGACY_UNPINNED``; their build is never guessed
from a file mtime. A timestamp says when a ledger was written, not which bytes a
long-lived process imported.

    python build_drift.py 96 97

prints, per seed, the exact build groups and any runs whose provenance cannot be
established.
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
REPO = D.parent

# What an AGENT is actually served. `campaign3_blind/grading/` and the reading
# tools are excluded on purpose: the grader runs offline, after the fact, so
# changing it does not change what any run received.
AGENT_FACING = [
    "src",
    "data",
    "langgraph_eval/agent.py",
    "campaign3_blind/host_hygiene.py",
    "campaign3_blind/phase.py",
    "campaign3_blind/run_blind.py",
    "scripts/blind_keys.py",
]


def newest_agent_facing_commit() -> tuple[int, str]:
    out = subprocess.run(
        ["git", "log", "-1", "--format=%ct %h %s", "--"] + AGENT_FACING,
        cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
    ts, rest = out.split(" ", 1)
    return int(ts), rest


def main(seeds) -> int:
    _cut, desc = newest_agent_facing_commit()
    print(f"current agent-facing commit: {desc}\n")
    all_builds = defaultdict(list)
    total_unpinned = 0
    for seed in seeds:
        builds = defaultdict(list)
        unpinned, unfinished = [], []
        for d in sorted((D / "runs").glob(f"*_seed{seed}")):
            led = d / "ledger.json"
            if not led.is_file():
                unfinished.append(d.name)
                continue
            try:
                record = json.loads(led.read_text())
            except (OSError, json.JSONDecodeError):
                unpinned.append(d.name + " (invalid ledger)")
                continue
            source_sha = record.get("source_sha256")
            commit = record.get("source_git_commit")
            if not source_sha:
                unpinned.append(d.name)
                continue
            label = f"{source_sha[:16]} ({str(commit)[:12] or 'no commit'})"
            builds[label].append(d.name)
            all_builds[label].append(d.name)
        total_unpinned += len(unpinned)
        print(f"seed {seed}: {sum(map(len, builds.values()))} pinned, "
              f"{len(unpinned)} LEGACY_UNPINNED, {len(unfinished)} unfinished")
        for label, names in sorted(builds.items()):
            print(f"   build {label}: {len(names)} run(s)")
        if unpinned:
            print("   no source hash; provenance cannot be reconstructed:")
            for name in unpinned:
                print(f"     {name}")
        if unfinished:
            print("   unfinished:")
            for name in unfinished:
                print(f"     {name}")

    if total_unpinned:
        print(f"\n{total_unpinned} completed run(s) predate source hashing. "
              "Their build cannot be proven from mtimes; do not quote them as "
              "a single-build round.")
    if len(all_builds) > 1:
        print(f"\nMIXED BUILD: {len(all_builds)} source hashes occur across "
              "the selected seeds. Re-run until one hash remains.")
        return 1
    if total_unpinned:
        return 1
    if all_builds:
        print("\nno drift: every completed run carries the same source hash")
    else:
        print("\nno completed pinned runs found")
    return 0


if __name__ == "__main__":
    sys.exit(main([int(a) for a in sys.argv[1:]] or [96, 97]))
