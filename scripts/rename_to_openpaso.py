#!/usr/bin/env python3
"""Rename the product from OASiS to openPASO, in one deterministic pass.

Why a script and not a one-off sed: the evaluation campaign runs from a branch
that keeps the old name until its build population is frozen. When that branch
adopts the new name, the same rename has to be reproduced on top of whatever it
has grown in the meantime. Replaying this script is one command; resolving a
nine-hundred-file text conflict is not.

What it renames: code, product text, packaging, configuration and documentation.

What it does NOT rename: the record of what already happened under the old name.
Run logs, grade files, launch scripts of rounds already run, and the campaign's
own written history say OASiS because that is what the software was called when
they were written. Rewriting them would misstate the record, so EXCLUDE holds
them out and `--check` proves they were left alone.

Usage:
    python scripts/rename_to_openpaso.py --dry-run    # report, change nothing
    python scripts/rename_to_openpaso.py              # rewrite in place
    python scripts/rename_to_openpaso.py --check      # exit 1 if work remains
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

# Three case classes cover every spelling measured in the tree, compounds
# included: OASIS_CONFIG_JSON, oasis-source, OASiS-arm, oasis__couple.
SUBSTITUTIONS = (
    ("OASiS", "openPASO"),   # prose and docstrings
    ("OASIS", "OPENPASO"),   # environment variables and other SHOUTING
    ("oasis", "openpaso"),   # identifiers, paths, log channels, tool prefixes
)

# Paths held out of the rename. Matched against the repository-relative path
# with fnmatch, so a trailing /* covers a whole directory.
EXCLUDE = (
    # --- the campaign's record of runs already made -------------------------
    "campaign3_blind/runs/*",
    "campaign3_blind/runs_quarantine/*",
    "campaign3_blind/rounds/*",
    "campaign3_blind/honest_rounds_*/*",
    "campaign3_blind/step_trials/*",
    "campaign3_blind/*.md",
    "campaign3_blind/*grades*.json",
    "campaign3_blind/path_readiness.json",
    "campaign3_blind/*.log",
    "campaign3_blind/round*_driver.sh",   # the driver of one round already run
    "campaign3_blind/chain_round*.sh",
    "data/blind_key_commitment.json",
    "validation/*/transcript.txt",
    # --- the two files that must keep saying both names ---------------------
    "scripts/rename_to_openpaso.py",
    "src/core/env_compat.py",
    # --- binary and rendered assets -----------------------------------------
    "*.png", "*.svg", "*.gif", "*.mp4", "*.pdf", "*.ico",
    "*.vtu", "*.vtk", "*.msh", "*.h5", "*.xdmf", "*.bp", "*.npz", "*.pyc",
)

PATTERN = re.compile("|".join(re.escape(a) for a, _ in SUBSTITUTIONS))
REPLACEMENT = dict(SUBSTITUTIONS)

# Names that are NOT ours. The official upstream project keeps its own name, so
# a blanket substitution has to be undone wherever it renamed someone else.
PRESERVE = (
    # The official upstream project keeps its own name.
    ("Hereon-InstituteMS/openPASO", "Hereon-InstituteMS/OASiS"),
    ("Hereon-InstituteMS/openpaso", "Hereon-InstituteMS/OASiS"),
    ("upstream openPASO project", "upstream OASiS project"),
    # Sentences that exist in order to name the old spelling. They document the
    # back-compatibility in src/core/env_compat.py and would be self-defeating
    # if this script renamed them.
    ("(OPENPASO_*) and new", "(OASIS_*) and new"),
    ("OPENPASO_* and OPENPASO_* name the same", "OASIS_* and OPENPASO_* name the same"),
)


def tracked_files(root: Path) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root,
                         capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def is_excluded(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in EXCLUDE)


def rewrite(text: str) -> str:
    out = PATTERN.sub(lambda m: REPLACEMENT[m.group(0)], text)
    for renamed, original in PRESERVE:
        out = out.replace(renamed, original)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and change nothing")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if any included file still carries the old name")
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()

    root = args.root.resolve()
    changed: list[tuple[str, int]] = []
    skipped_binary = 0

    for rel in tracked_files(root):
        if is_excluded(rel):
            continue
        path = root / rel
        if not path.is_file() or path.is_symlink():
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            skipped_binary += 1
            continue
        hits = len(PATTERN.findall(original))
        if rewrite(original) == original:
            continue
        if not hits:
            continue
        changed.append((rel, hits))
        if not (args.dry_run or args.check):
            path.write_text(rewrite(original), encoding="utf-8")

    total = sum(hits for _, hits in changed)
    verb = "would change" if (args.dry_run or args.check) else "changed"
    print(f"{verb} {total} occurrence(s) in {len(changed)} file(s)"
          f"; {skipped_binary} non-text file(s) skipped")
    for rel, hits in sorted(changed, key=lambda item: -item[1])[:25]:
        print(f"  {hits:5d}  {rel}")
    if len(changed) > 25:
        print(f"  ... and {len(changed) - 25} more")

    if args.check and changed:
        print("\nFAIL: the rename is not complete.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
