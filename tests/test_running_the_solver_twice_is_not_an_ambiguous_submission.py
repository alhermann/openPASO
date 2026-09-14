"""Re-running the solver got the submission rejected, in one arm only.

openPASO's run_simulation writes into
`simulation_outputs/<backend>_<YYYYmmdd_HHMMSS>/`, a fresh directory per call.
An agent that runs its solver more than once — the normal way to iterate — ends
up with several copies of solution_level<k>.csv at the SAME depth, and
discover_levels called that an ambiguity the contract could not resolve and
raised a problem.

MEASURED: 18 runs in the tree carry two or more timestamped directories for one
backend, and every one is an openPASO-arm run, because only that arm has the tool.
DU1_27b_MCP_seed7 has 26 of them. The rejection is therefore arm-specific by
construction, and correcting it RAISES the measured uplift — it is corrected
because it is wrong, and the direction is recorded so nobody has to guess.

The contract does pick a winner: a file at the top of the sandbox wins outright,
and among timestamped siblings the newest is the run's final state. The previous
tiebreak sorted by (depth, str) and so picked the OLDEST — grading an attempt
the agent had already superseded.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign3_blind"))

from grading.submission import _timestamp_in, discover_levels  # noqa: E402


def _mk(work: Path, rel: str, body: str) -> Path:
    p = work / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(body)
    return p


OLD = "x,y,u\n0.5,0.5,1.0\n"
NEW = "x,y,u\n0.5,0.5,2.0\n"


def test_the_timestamp_helper_orders_in_time():
    a = _timestamp_in("w/simulation_outputs/ngsolve_20260821_110608/s.csv")
    b = _timestamp_in("w/simulation_outputs/ngsolve_20260821_113326/s.csv")
    assert a and b and b > a
    assert _timestamp_in("w/solution_level1.csv") == ""


def test_two_runs_of_the_solver_are_not_a_problem(tmp_path):
    work = tmp_path / "work"
    _mk(work, "simulation_outputs/ngsolve_20260821_110608/solution_level1.csv", OLD)
    _mk(work, "simulation_outputs/ngsolve_20260821_113326/solution_level1.csv", NEW)
    levels, problems, notes = discover_levels(work, False)
    assert not any("same depth" in p for p in problems), (
        f"re-running the solver was still rejected: {problems}"
    )
    assert any("LATEST" in n for n in notes)


def test_the_latest_copy_is_the_one_graded(tmp_path):
    work = tmp_path / "work"
    _mk(work, "simulation_outputs/ngsolve_20260821_110608/solution_level1.csv", OLD)
    newest = _mk(work, "simulation_outputs/ngsolve_20260821_113326/solution_level1.csv", NEW)
    levels, _, _ = discover_levels(work, False)
    picked = [p for lvl in levels.values() for p in lvl.values()]
    assert newest in picked, (
        f"an earlier attempt was graded instead of the agent's final one: {picked}"
    )


def test_a_top_level_file_still_wins_outright(tmp_path):
    """The contractual location beats any tool directory, newest or not."""
    work = tmp_path / "work"
    top = _mk(work, "solution_level1.csv", OLD)
    _mk(work, "simulation_outputs/ngsolve_20260821_113326/solution_level1.csv", NEW)
    levels, problems, _ = discover_levels(work, False)
    picked = [p for lvl in levels.values() for p in lvl.values()]
    assert top in picked
    assert not any("same depth" in p for p in problems)


def test_two_untimestamped_copies_are_still_ambiguous(tmp_path):
    """Without a timestamp nothing picks a winner, and that must still say so."""
    work = tmp_path / "work"
    _mk(work, "attempt_a/solution_level1.csv", OLD)
    _mk(work, "attempt_b/solution_level1.csv", NEW)
    _, problems, _ = discover_levels(work, False)
    assert any("same depth" in p for p in problems), (
        "the ambiguity rule must survive for copies the contract really cannot "
        "choose between"
    )
