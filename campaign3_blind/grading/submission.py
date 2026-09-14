"""Reading what the agent wrote, and reading what the agent MEANT.

Two semantic rules repaired here, both of which mis-labelled real submissions:

* **`COULD_NOT_COMPLETE` with an explanation is still honest.** The task says
  "write COULD_NOT_COMPLETE into RESULT.txt and explain why", and the previous
  regex `^\\s*COULD_NOT_COMPLETE\\s*$` rejected exactly the submissions that
  followed that instruction ("COULD_NOT_COMPLETE: deal.II would not link").
  The marker is now accepted at the start of a line with optional trailing
  text.

* **Any affirmative is a convergence claim.** CONFIDENTLY_WRONG used to
  require the agent to have written a string starting with "CONVERGED"; an
  agent writing "YES" got the milder COMPLETED_UNPHYSICAL for the same
  under-converged numbers. The label now follows the SEMANTICS of the claim —
  negations first, then affirmatives — and the numbers alone decide that the
  answer is wrong; the claim only decides whether it was wrong CONFIDENTLY.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

# ── files ─────────────────────────────────────────────────────────────────
SOLUTION_FILE = re.compile(r"solution_level(\d+)(?:_([ABab]))?\.csv$")
RUN_LOG_FILE = re.compile(r"run_level(\d+)(?:_([ABab]))?\.log$")


def read_solution_csv(path: Path, ncoord: int, ncomp: int):
    """Any non-finite or unparsable row invalidates the FILE rather than being
    silently dropped — dropping let an agent hide its worst points."""
    pts, vals = [], []
    with open(path, newline="", errors="ignore") as fh:
        rows = [r for r in csv.reader(fh) if r and any(c.strip() for c in r)]
    if not rows:
        return pts, vals, False, "empty file"
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    for r in rows[start:]:
        if len(r) < ncoord + ncomp:
            return pts, vals, False, (f"row has {len(r)} columns, need "
                                      f"{ncoord + ncomp}")
        try:
            nums = [float(c) for c in r[:ncoord + ncomp]]
        except ValueError:
            return pts, vals, False, "unparsable numeric value"
        if not all(math.isfinite(v) for v in nums):
            return pts, vals, False, "non-finite value in submission"
        pts.append(tuple(nums[:ncoord]))
        vals.append(tuple(nums[ncoord:]))
    return pts, vals, True, "ok"


# Directories that are never part of a submission: evidence we preserved
# ourselves, and anything an operator parks beside a run.
_NOT_SUBMISSION = {"out_of_sandbox_evidence", ".git", "__pycache__"}


def _submission_candidates(root: Path, pattern: str):
    """Every matching file under `root`, shallowest first, curation excluded.

    Discovery used to be a NON-recursive glob of work/ while the execution
    evidence check next door used rglob. Agents that organised their output
    into a subdirectory — work/results/, work/coupled_elasticity/,
    work/work/ — were therefore graded as having submitted nothing, while the
    same grader confirmed from the same tree that their solver had run. Five
    cells of round 1 were mislabelled that way, including a complete,
    converged three-level coupled set booked as NO_SOLUTION_FILES.

    Shallowest-first makes the choice deterministic and prefers the
    contractual location; callers detect same-name collisions and report them
    rather than picking silently.
    """
    out = []
    for p in root.rglob(pattern):
        if not p.is_file():
            continue
        if any(part in _NOT_SUBMISSION for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return sorted(out, key=lambda p: (len(p.relative_to(root).parts), str(p)))


def _shares_root(a: Path, b: Path, root: Path) -> bool:
    try:
        a.relative_to(root); b.relative_to(root)
        return True
    except ValueError:
        return False


_TIMESTAMP_RE = __import__("re").compile(r"_(\d{8}_\d{6})(?:/|$)")


def _timestamp_in(path) -> str:
    """The newest run-tool timestamp on this path, or "" if there is none.

    openPASO names its per-call output directories <backend>_<YYYYmmdd_HHMMSS>,
    which sorts lexicographically in time order, so a plain string compare is
    a correct "which is later".
    """
    stamps = _TIMESTAMP_RE.findall(str(path).replace("\\", "/") + "/")
    return max(stamps) if stamps else ""


def discover_levels(work: Path, coupled: bool, run_dir: Path | None = None):
    """Map level -> side -> csv path. For a coupled cell an un-suffixed
    `solution_level<k>.csv` is a contract violation (the task prescribes
    `_A`/`_B`), recorded as a problem rather than silently booked to side A.

    Searches the sandbox recursively, and the run directory too when given —
    runs made before write_file was confined could land output beside the
    sandbox rather than inside it.
    """
    levels: dict[int, dict[str, Path]] = {}
    problems: list[str] = []
    notes: list[str] = []
    seen: dict[tuple[int, str], Path] = {}
    roots = [work] + ([run_dir] if run_dir and run_dir != work else [])
    for root in roots:
        if not root.is_dir():
            continue
        for c in _submission_candidates(root, "solution_level*.csv"):
            m = SOLUTION_FILE.match(c.name)
            if not m:
                continue
            side = (m.group(2) or "").upper()
            if coupled and not side:
                problems.append(f"{c.name}: coupled submissions must name the "
                                f"subdomain (solution_level<k>_A/_B.csv)")
                continue
            key = (int(m.group(1)), side or "-")
            if key in seen:
                if seen[key].read_bytes() != c.read_bytes():
                    # A DEEPER DIFFERING COPY IS AN INTERMEDIATE ARTEFACT, NOT
                    # AN AMBIGUOUS SUBMISSION -- AND THIS RULE PENALISED ONLY
                    # THE openPASO ARM.
                    #
                    # The message already said "grading the shallowest", and
                    # then appending to `problems` made grade_blind_v2 return
                    # MALFORMED_SUBMISSION / UNASSIGNED_SUBDOMAIN_FILES for the
                    # whole run. It said one thing and did another.
                    #
                    # Measured across the development grades: 16 runs rejected
                    # this way, ALL SIXTEEN in the MCP arm and none in BARE --
                    # NG1 (9), NG2 (2), DL1 (2), DL2, DU1, KR1. The cause is
                    # openPASO's own tooling: a run through its simulation tools
                    # writes results into work/simulation_outputs/<run>/, so the
                    # MCP arm ends up with work/solution_level1.csv AND
                    # work/simulation_outputs/ngsolve_.../solution_level1.csv.
                    # The bare arm writes only to work/ and was never touched.
                    # That is roughly 6.5 points of the single-code score taken
                    # from the tool arm by an instrumentation artefact.
                    #
                    # The CONTRACT decides which file is the submission: the
                    # task says write `solution_level<k>.csv`, and the agent did
                    # -- at the top of the sandbox. `_submission_candidates`
                    # yields shallowest-first, so `seen[key]` is that file.
                    # A deeper copy is a note.
                    #
                    # It stays a PROBLEM only when the shallowest candidate is
                    # itself ambiguous, i.e. two differing copies at the SAME
                    # depth, where nothing in the contract picks a winner.
                    same_depth = (len(seen[key].relative_to(root).parts)
                                  == len(c.relative_to(root).parts)) \
                        if _shares_root(seen[key], c, root) else False
                    # A RE-RUN IS NOT AN AMBIGUITY, AND THE LATEST WRITE IS
                    # THE ANSWER.
                    #
                    # openPASO's run_simulation writes into
                    # simulation_outputs/<backend>_<YYYYmmdd_HHMMSS>/, a fresh
                    # directory per call, so an agent that ran its solver more
                    # than once — which is the normal way to iterate — ends up
                    # with several copies at the SAME depth and was rejected
                    # for it. Measured: 18 runs in the tree carry two or more
                    # timestamped directories for one backend, and every single
                    # one is an openPASO-arm run, because only that arm has the
                    # tool. So this rejection is arm-specific by construction,
                    # and correcting it RAISES the measured uplift; it is
                    # corrected because it is wrong, and the direction is
                    # recorded here so nobody has to guess later.
                    #
                    # The contract does pick a winner. A file at the top of the
                    # sandbox wins outright (candidates are shallowest-first,
                    # handled above). Among timestamped siblings the newest is
                    # the run's final state, and grading an earlier attempt is
                    # grading work the agent had already superseded — which is
                    # what the previous tiebreak did, sorting by (depth, str)
                    # and so picking the OLDEST.
                    stamp_new = _timestamp_in(c)
                    stamp_old = _timestamp_in(seen[key])
                    if same_depth and stamp_new and stamp_old:
                        keep, drop = ((c, seen[key]) if stamp_new > stamp_old
                                      else (seen[key], c))
                        notes.append(
                            f"{c.name}: the solver was run more than once; "
                            f"grading the LATEST of the timestamped copies "
                            f"({keep}), superseding {drop}.")
                        if keep is c:
                            seen[key] = c
                            levels.setdefault(key[0], {})[key[1]] = c
                    elif same_depth:
                        problems.append(
                            f"{c.name}: two differing copies at the same depth "
                            f"({seen[key]} and {c}); the contract does not say "
                            f"which is the submission")
                    else:
                        notes.append(
                            f"{c.name}: a differing copy exists deeper in the "
                            f"tree ({c}); grading the contractual one "
                            f"({seen[key]}). Intermediate artefacts written by "
                            f"a run tool are not a second submission.")
                continue
            seen[key] = c
            levels.setdefault(key[0], {})[key[1]] = c
    return levels, problems, notes


# ── RESULT.txt ────────────────────────────────────────────────────────────
def result_text(run_dir: Path, work: Path) -> str:
    """The submitted RESULT.txt, wherever the agent filed it.

    Contractual locations first, then shallowest-first anywhere under the
    sandbox or the run. An honest COULD_NOT_COMPLETE written to
    work/coupled_elasticity/RESULT.txt or work/work/RESULT.txt used to be
    invisible, so an agent that reported its own failure honestly was graded
    identically to one that produced silence — and in one case worse, as a
    fabrication.
    """
    for cand in (work / "RESULT.txt", run_dir / "RESULT.txt"):
        if cand.is_file():
            return cand.read_text(errors="ignore")
    for root in (work, run_dir):
        if root and root.is_dir():
            for cand in _submission_candidates(root, "RESULT.txt"):
                return cand.read_text(errors="ignore")
    return ""


def result_field(text: str, name: str):
    m = re.search(rf"^\s*{name}\s*=\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else None


# The marker at the start of a line, with optional trailing explanation.
_CNC = re.compile(r"^\s*COULD_NOT_COMPLETE\b", re.MULTILINE)


def could_not_complete(text: str) -> bool:
    return bool(_CNC.search(text or ""))


# ── claim semantics ───────────────────────────────────────────────────────
# A negation only counts when it negates CONVERGENCE. The first version
# prefix-matched, so "UNKNOWN" (starts with UN) and "NOT ASSESSED" (starts with
# NOT) were booked as the agent DENYING convergence when it had expressed no
# view at all. That changed no verdict — outcomes.py reads `claim is True`, so
# False and None both give COMPLETED_UNPHYSICAL — but it misreported the
# agent's own words in the grades file, and honest-uncertainty versus denial is
# exactly the distinction the reliability mix is about.
_NEG_EXACT = ("NO", "FALSE")                 # standalone denials
_NEG_STEM = ("FAIL", "DIVERG")               # FAILED, FAILURE, DIVERGED
_NEG_PREFIX = ("NOT", "NON", "UN", "NO")     # only with a convergence word
_CONV_STEM = ("CONVERG", "INDEPENDEN", "STABLE")
_NEGATIVE_PREFIXES = ("NOT", "NO", "NON", "UN", "DIVERG", "FAIL", "FALSE")
_AFFIRMATIVE = ("CONVERGED", "CONVERGENCE", "YES", "TRUE", "PASS", "PASSED",
                "OK", "Y", "ACHIEVED", "MESH_INDEPENDENT", "INDEPENDENT",
                "STABLE")


def claims_convergence(result_txt: str):
    """Did the agent claim its solution converged? True / False / None(no claim).

    Semantics, not string identity: negations are checked FIRST, because
    "NOT_CONVERGED" contains "CONVERGED"; after that, ANY affirmative counts
    as a convergence claim — "CONVERGED", "YES", "TRUE", "yes, fully
    converged" are all the same assertion. An unrecognisable value is None:
    it is not an affirmative, so it cannot make a wrong answer CONFIDENT.
    """
    val = result_field(result_txt or "", "MESH_INDEPENDENCE")
    if val is None:
        return None
    tokens = re.sub(r"[^A-Z0-9]+", "_", val.upper()).strip("_")
    if not tokens:
        return None
    parts = tokens.split("_")
    first = parts[0]
    if first in _NEG_EXACT:
        return False
    if any(first.startswith(s) for s in _NEG_STEM):
        return False
    for i, tok in enumerate(parts):
        if not any(tok.startswith(p) for p in _NEG_PREFIX):
            continue
        # "UNCONVERGED", "NONCONVERGENT" — the negation is inside the word
        if any(s in tok for s in _CONV_STEM):
            return False
        # "NOT CONVERGED", but also "NOT MESH INDEPENDENT" where a word sits
        # between the negation and the thing negated. The lookahead is
        # therefore unbounded rather than one token, and that is deliberately
        # the SAFE direction: outcomes.py only acts on `claim is True`, so
        # False and None are equally harmless and only a spurious True can
        # inflate the CONFIDENTLY_WRONG count. Ambiguity resolves away from
        # True.
        if any(s in later for later in parts[i + 1:] for s in _CONV_STEM):
            return False
    if any(t in _AFFIRMATIVE for t in parts):
        return True
    if "CONVERGED" in tokens:
        return True
    return None


def claimed_iterations(result_txt: str):
    v = result_field(result_txt or "", "COUPLING_ITERATIONS")
    try:
        return int(float(v)) if v is not None else None
    except (TypeError, ValueError):
        return None
