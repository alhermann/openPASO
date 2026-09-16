"""The ladder is delegated, so the write-time hooks never see it.

The served ladder asks for one sub-agent per step. A worker that writes the
deliverables takes every write-time finding with it when it exits, and the
parent gets a success report. Measured on a live cell: C8 seed 8583 built a
real three-level ladder (its own consoles read 60/212/795 against 72/255/957
on a constant 1936-point probe grid), had a worker write all eighteen
deliverables, and the worker copied level 3's console into all six run logs.
Four findings existed, each naming the file and the fix. None reached anyone
who could act, and the cell graded MALFORMED on that one reason.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.workspace_advisor import deliverable_findings_after_worker  # noqa: E402

LIVE = ROOT / "campaign3_blind/runs/C8_27b_MCP_seed8583/work"


def _ladder(tmp_path, *, copy_log_from=None):
    """Three coupled levels with honest consoles; optionally a copied run log."""
    ndof = {1: {"A": 60, "B": 72}, 2: {"A": 212, "B": 255}, 3: {"A": 795, "B": 957}}
    pts = [(i / 6.0, j / 6.0) for i in range(7) for j in range(7)]
    for side in ("A", "B"):
        d = tmp_path / f"side_{side}"
        d.mkdir(exist_ok=True)
        for k in (1, 2, 3):
            (d / f"participant_output_level{k}.log").write_text(
                f"solver banner\nNDOF = {ndof[k][side]}\niteration 1 res 1e-3\n")
    for k in (1, 2, 3):
        (tmp_path / f"residual_level{k}.csv").write_text(
            "iteration,interface_residual\n2,1e-1\n3,4e-2\n4,1e-2\n")
        for side in ("A", "B"):
            (tmp_path / f"solution_level{k}_{side}.csv").write_text(
                "x,y,u\n" + "".join(f"{x},{y},{x + y}\n" for x, y in pts))
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "y,u,q\n0.0,1.0,-2.0\n0.5,1.5,-2.0\n")
            src = copy_log_from if copy_log_from else k
            (tmp_path / f"run_level{k}_{side}.log").write_text(
                f"solver banner\nNDOF = {ndof[src][side]}\niteration 1 res 1e-3\n")
    return tmp_path


def test_an_honest_ladder_is_left_alone(tmp_path):
    assert deliverable_findings_after_worker(_ladder(tmp_path)) == ""


def test_the_finest_console_copied_over_every_log_is_named(tmp_path):
    out = deliverable_findings_after_worker(_ladder(tmp_path, copy_log_from=3))
    assert "RUN LOG FROM THE WRONG LEVEL" in out
    assert "run_level1_A.log" in out
    # THIS USED TO ASSERT THE PHRASE "budget left now", AND THAT WAS WRONG OF
    # IT. The point of the hook is that it fires on the worker's return instead
    # of at hand-in, and the way to pin that is where it is called from, not a
    # sentence about the clock -- which openPASO cannot see and which was false for
    # roughly half the runs that read it. The sentence is gone; what the finding
    # says about the DEFECT is what the hook is for, so that is what is pinned.
    # See tests/test_openpaso_does_not_claim_to_know_the_clock.py.
    assert "whoever judges the result" in out, (
        "the finding must say what the defect costs, not merely name it")


def test_it_names_the_file_to_copy_not_just_the_defect(tmp_path):
    out = deliverable_findings_after_worker(_ladder(tmp_path, copy_log_from=3))
    assert "participant_output_level1.log" in out


def test_a_near_miss_dof_count_is_not_called_a_copied_log(tmp_path):
    """969 against 957 is not another level's console; it must not fire.

    Measured: the underlying body reports any mismatch, and on the graded
    record that spoke on 3 of the 32 correct cells, two of them differing by
    1% and 6% with no other level to match.
    """
    w = _ladder(tmp_path)
    (w / "run_level3_B.log").write_text("solver banner\nNDOF = 969\niteration 1 res 1e-3\n")
    assert deliverable_findings_after_worker(w) == ""


def test_the_mesh_ladder_check_is_not_carried_here(tmp_path):
    """It speaks on a quarter of the cells that go on to be correct."""
    out = deliverable_findings_after_worker(_ladder(tmp_path, copy_log_from=3))
    assert "WAS NOT HALVED" not in out


def test_an_empty_directory_says_nothing(tmp_path):
    assert deliverable_findings_after_worker(tmp_path) == ""


@pytest.mark.skipif(not LIVE.is_dir(), reason="the recorded cell is not present")
def test_the_recorded_cell_that_motivated_this_is_named():
    out = deliverable_findings_after_worker(LIVE)
    assert "RUN LOG FROM THE WRONG LEVEL" in out
    assert "it is level 3's console" in out
