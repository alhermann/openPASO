"""The proof a level ran, asked for while there is still budget to produce it.

The checks themselves are not new. They run at hand-in, and by then the median
recorded cell has two tool calls left and one in five has none, while the work
they name costs five to fifteen actions. 34 cells reached three coupled levels
with both codes proven, the coupling proven and the interface satisfied, and
scored nothing on this.

What is new is WHEN they are asked, and the discipline that makes an early ask
affordable: never mention a level the agent has not reached.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.result_audit import (  # noqa: E402
    completeness_findings, contract_findings, deliverable_proof_due)
from tools.workspace_advisor import _level_proof_gap  # noqa: E402


def _coupled_level(tmp_path, k, *, sides=("A", "B"), with_log=True, rows=4):
    """One coupled level's files: a history, a field per side, and its logs."""
    hist = ["iteration,interface_residual"] + [
        f"{i + 2},{0.1 * 0.4 ** i:.6e}" for i in range(rows)]
    (tmp_path / f"residual_level{k}.csv").write_text("\n".join(hist) + "\n")
    # A REALISTIC PROBE GRID. An earlier fixture wrote three rows against an
    # NDOF of 40 and drew the discretisation-size finding, which was the check
    # doing its job on a toy: no result set verified correct anywhere has a
    # ratio like that.
    pts = [(i / 6.0, j / 6.0) for i in range(7) for j in range(7)]
    for s in sides:
        (tmp_path / f"solution_level{k}_{s}.csv").write_text(
            "x,y,u\n" + "".join(f"{x},{y},{x + y}\n" for x, y in pts))
        (tmp_path / f"interface_level{k}_{s}.csv").write_text(
            "y,u,q\n0.0,1.0,-2.0\n0.5,1.5,-2.0\n")
        if with_log:
            (tmp_path / f"run_level{k}_{s}.log").write_text(
                f"solver banner\nNDOF = {40 * 4 ** (k - 1) + (7 if s == 'B' else 0)}\ndone\n")
    return tmp_path


def test_a_coupled_level_with_no_dof_line_is_named(tmp_path):
    _coupled_level(tmp_path, 1, with_log=False)
    found = deliverable_proof_due(tmp_path, [1])
    assert any("NO DOF-COUNT LINE" in f["finding"] for f in found), found


def test_a_coupled_level_that_has_its_logs_is_left_alone(tmp_path):
    _coupled_level(tmp_path, 1, with_log=True)
    assert deliverable_proof_due(tmp_path, [1]) == []


def test_nothing_is_asked_about_a_level_not_yet_reached(tmp_path):
    """The whole point of the early ask: it must not demand unfinished work."""
    _coupled_level(tmp_path, 1, with_log=True)
    found = deliverable_proof_due(tmp_path, [1])
    text = " ".join(f["finding"] for f in found)
    assert "level2" not in text and "level3" not in text, text
    assert "LEVEL(S) DELIVERED" not in text, (
        "the level-count finding is true at hand-in and noise mid-ladder")


def test_the_level_count_finding_is_not_carried_early(tmp_path):
    """contract_findings says it; the due-now selector must drop it."""
    _coupled_level(tmp_path, 1, with_log=True)
    full = " ".join(f["finding"] for f in contract_findings(tmp_path))
    assert "LEVEL(S) DELIVERED" in full, "fixture no longer reproduces the noise"
    early = " ".join(f["finding"] for f in deliverable_proof_due(tmp_path, [1]))
    assert "LEVEL(S) DELIVERED" not in early


def test_an_empty_level_list_asks_for_nothing(tmp_path):
    _coupled_level(tmp_path, 1, with_log=False)
    assert deliverable_proof_due(tmp_path, []) == []
    assert deliverable_proof_due(tmp_path, None) == []


def test_a_missing_side_at_a_coupled_level_is_named(tmp_path):
    """Both sides must be established elsewhere in the family, as on a real run.

    completeness_findings infers the expected set from the agent's own files,
    so a family whose only surviving member is side A expects only side A --
    which is the correct inference and not a gap.
    """
    _coupled_level(tmp_path, 1, with_log=True)
    _coupled_level(tmp_path, 2, with_log=True)
    (tmp_path / "solution_level2_B.csv").unlink()
    found = deliverable_proof_due(tmp_path, [1, 2])
    assert any("DELIVERABLE SET IS INCOMPLETE" in f["finding"] for f in found), found
    assert any("solution_level2_B.csv" in f["finding"] for f in found), found


def test_the_level_restriction_is_opt_in_for_both_bodies(tmp_path):
    """Called with no restriction the bodies must behave exactly as before."""
    _coupled_level(tmp_path, 1, with_log=True)
    _coupled_level(tmp_path, 2, with_log=True)
    assert contract_findings(tmp_path) == contract_findings(tmp_path, only_levels=None)
    assert completeness_findings(tmp_path) == completeness_findings(tmp_path, only_levels=None)
    one = contract_findings(tmp_path, only_levels={1})
    assert not any("'2" in f["finding"] for f in one), one


def test_the_write_time_gap_fires_only_for_a_level_that_coupled(tmp_path):
    _coupled_level(tmp_path, 1, with_log=False)
    assert "NO DOF-COUNT LINE" in _level_proof_gap(tmp_path, "1")
    # level 2 has a field file but no history: it was never coupled
    (tmp_path / "solution_level2_A.csv").write_text("x,y,u\n0,0,1\n")
    assert _level_proof_gap(tmp_path, "2") == ""


def test_the_write_time_gap_is_silent_when_the_log_is_there(tmp_path):
    _coupled_level(tmp_path, 1, with_log=True)
    assert _level_proof_gap(tmp_path, "1") == ""


@pytest.mark.parametrize("bad", ["", "x", None])
def test_the_write_time_gap_survives_a_bad_level(tmp_path, bad):
    _coupled_level(tmp_path, 1, with_log=False)
    assert _level_proof_gap(tmp_path, bad) == ""


# ── the answer's quality, asked at the same moment ─────────────────────────

def test_a_solution_field_that_peaks_at_nothing_is_named(tmp_path):
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    for s in ("A", "B"):
        (tmp_path / f"solution_level1_{s}.csv").write_text(
            "x,y,u\n" + "".join(f"{i/6.0},0.0,0.0\n" for i in range(7)))
    assert any("NEAR-ZERO FIELD" in f["finding"]
               for f in field_quality_due(tmp_path, [1])), "a zero solve must be named"


def test_an_interface_trace_of_zero_is_not_called_a_dead_solve(tmp_path):
    """A seam held at zero is ordinary; two cells that graded CORRECT carry one."""
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    for s in ("A", "B"):
        (tmp_path / f"interface_level1_{s}.csv").write_text("y,u,q\n0.0,0.0,0.0\n0.5,0.0,0.0\n")
    assert not any("NEAR-ZERO FIELD" in f["finding"]
                   for f in field_quality_due(tmp_path, [1]))


def test_a_ladder_going_backwards_is_named(tmp_path):
    from tools.result_audit import field_quality_due
    for k, scale in ((1, 1.0), (2, 5.0), (3, 25.0)):
        _coupled_level(tmp_path, k, with_log=True)
        for s in ("A", "B"):
            pts = [(i / 6.0, j / 6.0) for i in range(7) for j in range(7)]
            (tmp_path / f"solution_level{k}_{s}.csv").write_text(
                "x,y,u\n" + "".join(f"{x},{y},{scale*(x+y)}\n" for x, y in pts))
    out = field_quality_due(tmp_path, [1, 2, 3])
    assert any("GOING BACKWARDS" in f["finding"] for f in out), out


def test_nothing_is_said_about_levels_not_yet_coupled(tmp_path):
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    assert field_quality_due(tmp_path, []) == []
    assert field_quality_due(tmp_path, None) == []


# ── an interface that transmitted nothing ──────────────────────────────────

def _two_sided_interface(tmp_path, k, tx, ty):
    """One level's interface files for both sides, tractions as given."""
    for side, sgn in (("A", 1.0), ("B", -1.0)):
        rows = ["x,y,ux,uy,tx,ty"]
        for i in range(5):
            y = i * 0.25
            rows.append(f"0.625,{y},{3.8e-07},{9.1e-08},{sgn*tx},{sgn*ty}")
        (tmp_path / f"interface_level{k}_{side}.csv").write_text("\n".join(rows) + "\n")


def test_a_traction_dead_on_both_sides_is_named(tmp_path):
    """Neither participant's own self-check can see this: each fires only when
    its OWN recovery is ~0 against a NONZERO partner, and here both are zero."""
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    _two_sided_interface(tmp_path, 1, tx=0.0, ty=0.0)
    out = field_quality_due(tmp_path, [1])
    assert any("EXCHANGED NOTHING" in f["finding"] for f in out), out
    assert any("cannot disagree" in f["finding"] for f in out), (
        "it must say WHY the iteration converges anyway")


def test_a_live_traction_is_not_named(tmp_path):
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    _two_sided_interface(tmp_path, 1, tx=5.5e-04, ty=3.4e-04)
    assert not any("EXCHANGED NOTHING" in f["finding"]
                   for f in field_quality_due(tmp_path, [1]))


def test_a_seam_zero_on_ONE_side_only_is_not_named(tmp_path):
    """A trace legitimately at zero on one side is ordinary; two CORRECT cells
    on record carry one."""
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    _two_sided_interface(tmp_path, 1, tx=5.5e-04, ty=3.4e-04)
    rows = ["x,y,ux,uy,tx,ty"] + [f"0.625,{i*0.25},{3.8e-07},{9.1e-08},0.0,0.0" for i in range(5)]
    (tmp_path / "interface_level1_A.csv").write_text("\n".join(rows) + "\n")
    out = [f for f in field_quality_due(tmp_path, [1]) if "EXCHANGED NOTHING" in f["finding"]]
    assert out == [], "one dead side is not the same defect as a dead exchange"


def test_an_all_zero_interface_is_not_reported_as_an_exchange_failure(tmp_path):
    """With no scale anywhere there is nothing to compare against."""
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    for side in ("A", "B"):
        rows = ["x,y,ux,uy,tx,ty"] + ["0.0,0.0,0.0,0.0,0.0,0.0"] * 3
        (tmp_path / f"interface_level1_{side}.csv").write_text("\n".join(rows) + "\n")
    assert not any("EXCHANGED NOTHING" in f["finding"]
                   for f in field_quality_due(tmp_path, [1]))


# ── the trace a side was told to hold ──────────────────────────────────────

def _side_pair(tmp_path, side, imported, exported):
    """A participant dir with an imports.json and an exports.json."""
    import json
    d = tmp_path / f"side_{side}"
    d.mkdir(exist_ok=True)
    coords = [[0.625, i * 0.25] for i in range(len(imported))]
    (d / "imports.json").write_text(json.dumps(
        {"P": {"field_name": "displacement", "coordinates": coords, "values": imported}}))
    (d / "exports.json").write_text(json.dumps(
        {"field_name": "displacement", "coordinates": coords, "values": exported}))
    return d


def test_a_side_that_exports_what_it_imported_is_silent(tmp_path):
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    v = [[1.0e-6, 2.0e-6], [1.5e-6, 2.5e-6], [2.0e-6, 3.0e-6]]
    _side_pair(tmp_path, "B", v, [r[:] for r in v])
    assert not any("DIFFERENT TRACE" in f["finding"]
                   for f in field_quality_due(tmp_path, [1]))


def test_a_side_that_moved_the_trace_it_was_given_is_named(tmp_path):
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    v = [[1.0e-6, 2.0e-6], [1.5e-6, 2.5e-6], [2.0e-6, 3.0e-6]]
    moved = [[2.3e-6, 2.0e-6], [1.5e-6, 2.5e-6], [2.0e-6, 3.0e-6]]
    _side_pair(tmp_path, "B", v, moved)
    out = [f for f in field_quality_due(tmp_path, [1]) if "DIFFERENT TRACE" in f["finding"]]
    assert out, "a side that overwrote the imposed trace must be named"
    assert "converges anyway" in out[0]["finding"], (
        "it must say why no other check sees this")


def test_the_flux_side_is_not_judged(tmp_path):
    """A side importing fluxes and exporting a trace has nothing to compare."""
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    _side_pair(tmp_path, "A", [[1.0], [2.0], [3.0]],
               [[1.0e-6, 2.0e-6], [1.5e-6, 2.5e-6], [2.0e-6, 3.0e-6]])
    assert not any("DIFFERENT TRACE" in f["finding"]
                   for f in field_quality_due(tmp_path, [1]))


def test_an_all_zero_import_is_not_judged(tmp_path):
    """Iteration one hands over zeros; there is no scale to compare against."""
    from tools.result_audit import field_quality_due
    _coupled_level(tmp_path, 1, with_log=True)
    _side_pair(tmp_path, "B", [[0.0, 0.0]] * 3, [[1e-6, 2e-6]] * 3)
    assert not any("DIFFERENT TRACE" in f["finding"]
                   for f in field_quality_due(tmp_path, [1]))
