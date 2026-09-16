"""Fixing this at level 1 costs one solve; at level 3 it costs the whole ladder.

THE FAILURE. `SK1_27b_MCP` builds a Stokes solve with 592,387 degrees of
freedom against a 1936-point probe grid — two orders of magnitude larger than
the coarsest mesh the task prescribes. Level 1 completes, level 2 cannot finish
inside the clock, and the run dies with one of three levels delivered. It does
this at seeds 14, 40 AND 96 with the identical DOF count: three seeds, one
cause, and nothing in the loop said a word.

THE CHECK NEEDS NOTHING BUT THE AGENT'S OWN TWO FILES: the NDOF its run log
printed at the coarsest level, against the number of rows in that level's
solution file. No key, no spec, no backend knowledge, and it is answerable at
level 1.

MEASURED over the 336 single-code runs on disk that wrote both files:

    highest ratio among runs graded CORRECT     0.50
    threshold NDOF/rows > 2 fires on 8 runs     0 of them CORRECT
                                                8 of 8 timed out or fell short

and executed through the real `contract_findings` over all 981 run directories
it fires on 10 with zero false alarms on a CORRECT run. Two of the ten are
coupled cells that submitted their INTERFACE point count as the field file
(C7: 44 rows, C10: 10 rows) — the same ratio caught from below.

These tests build the directory an agent leaves behind and ask what the audit
tells it, because a finding that is computed and not delivered is not a finding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.result_audit import contract_findings          # noqa: E402


def _run(tmp: Path, ndof: int, rows: int, levels=(1, 2, 3), side="") -> Path:
    w = tmp / "work"
    w.mkdir(parents=True, exist_ok=True)
    for k in levels:
        (w / f"run_level{k}{side}.log").write_text(
            f"solver ok\nNDOF = {ndof * 4 ** (k - 1)}\n")
        body = "x,y,u\n" + "".join(
            f"{0.5 + i},{0.25},{1.0 + i + k}\n" for i in range(rows))
        (w / f"solution_level{k}{side}.csv").write_text(body)
    return w


def _fired(w: Path) -> list:
    return [f for f in contract_findings(w)
            if f["sequence"] == "discretisation size"]


def test_a_mesh_two_orders_too_large_is_named_at_level_one():
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _run(Path(d), ndof=592387, rows=1936)
        hits = _fired(w)
        assert hits, "SK1's actual numbers must produce a finding"
        txt = hits[0]["finding"]
        assert "592387" in txt and "1936" in txt
        assert "306" in txt, "the ratio itself must be in the message"
        assert "coarsest" in txt.lower()
        # it must name BOTH causes, because they need opposite fixes
        assert "far larger than the coarsest level" in txt
        assert "far fewer rows" in txt


def test_a_normal_run_is_left_alone():
    """The measured ceiling among CORRECT runs is 0.50 — leave that alone."""
    import tempfile
    for ndof, rows in ((81, 1936), (289, 1936), (968, 1936), (3600, 1936)):
        with tempfile.TemporaryDirectory() as d:
            w = _run(Path(d), ndof=ndof, rows=rows)
            assert not _fired(w), (
                f"NDOF={ndof} against {rows} rows is ratio "
                f"{ndof / rows:.2f}; every CORRECT run in the tree sits below "
                f"0.51 and none of these may be flagged")


def test_a_two_row_solution_file_trips_the_same_ratio_from_below():
    """FE2 seed 4's real shape: the deliverable is short, not the mesh large."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _run(Path(d), ndof=659, rows=2)
        assert _fired(w), "a 2-row solution file must not pass unremarked"


def test_the_coupled_per_side_shape_is_covered():
    """C7 and C10 submitted their interface count as the field file."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = _run(Path(d), ndof=216, rows=44, levels=(1, 2), side="_A")
        assert _fired(w), (
            "the per-side deliverable must be checked too — two real coupled "
            "runs submitted 44 and 10 rows where the field grid is far bigger")


def test_the_threshold_is_exactly_where_the_measurement_put_it():
    """Not a round number chosen by taste: 2x, with CORRECT topping out at 0.5."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        assert not _fired(_run(Path(d) / "a", ndof=200, rows=100))   # 2.0
    with tempfile.TemporaryDirectory() as d:
        assert _fired(_run(Path(d) / "b", ndof=201, rows=100))       # just over


def test_a_missing_run_log_does_not_make_this_check_shout():
    """No NDOF line means the OTHER finding fires; this one must stay silent."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        w = Path(d) / "work"
        w.mkdir(parents=True)
        (w / "solution_level1.csv").write_text("x,y,u\n0.5,0.5,1.0\n")
        assert not _fired(w)
        assert any("NDOF" in f["finding"] for f in contract_findings(w)), (
            "the run-log contract finding is the one that belongs here")
