"""A side whose exports are identically zero handed its partner nothing.

THE UPSTREAM SUPPLY OF THE DEAD EXCHANGE. `_interface_transmitted_nothing`
catches the pair once BOTH sides are silent and the per-level interface files
exist. By then the iteration has converged on nothing and the ladder is built
on it. This reads one side's own exports.json, which exists the moment a
participant runs once -- before any coupling, before any deliverable.

THE SERVED SELF-CHECK CANNOT SEE IT. Its zero guard fires only for a Neumann
side whose recovered flux is ~0 against a NONZERO imported flux
(coupling_knowledge.py, EXPORT SELF-CHECK). A side handed zeros -- the first
coupling iteration, or a partner that is itself silent -- skips that branch and
exports nothing without a word. Measured in single-side trials with the
contract in hand and nothing to orchestrate: 2 of 7 successful-looking exports
were identically zero, both values and fluxes.

ADMISSION, over 588 graded cells: 41 flagged (33 honest-incomplete, 5 failed,
3 malformed) and **0 of 35 CORRECT**, 0 of 20 unphysical, 0 of 11 confidently
wrong. It fires on nothing that ever produced a gradeable number.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.result_audit import _side_exported_nothing, field_quality_due  # noqa: E402


def _side(tmp_path, name, values, fluxes):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "exports.json").write_text(json.dumps(
        {"field_name": "u", "coordinates": [[0.5, y / 10.0] for y in range(len(fluxes) or len(values))],
         "values": values, "normal_fluxes": fluxes,
         "n_points": len(fluxes) or len(values)}))
    return d


def test_a_side_of_exact_zeros_is_named(tmp_path):
    _side(tmp_path, "side_A", [0.0] * 8, [0.0] * 8)
    out = _side_exported_nothing(tmp_path)
    assert out, "a side that exported only zeros must be named"
    assert "IDENTICALLY ZERO" in out[0]["finding"]
    assert "side_A" in out[0]["finding"]


def test_a_side_carrying_a_field_is_left_alone(tmp_path):
    _side(tmp_path, "side_A", [1e-7] * 8, [0.0] * 8)
    assert _side_exported_nothing(tmp_path) == [], (
        "a field that is small is not a field that is absent")


def test_a_zero_trace_with_a_real_flux_is_left_alone(tmp_path):
    # A Dirichlet seam held at zero is physically ordinary and two cells that
    # graded CORRECT carry exactly that. What it still exports is a FLUX
    # recovered from its own system, and that is not zero.
    _side(tmp_path, "side_B", [0.0] * 8, [-2.5] * 8)
    assert _side_exported_nothing(tmp_path) == [], (
        "a legitimately zero trace must not be confused with a dead side")


def test_a_side_that_exported_nothing_at_all_is_a_different_defect(tmp_path):
    # Empty arrays mean the participant never got as far as writing a field;
    # that has its own finding and must not be reported as a zero export.
    _side(tmp_path, "side_A", [], [])
    assert _side_exported_nothing(tmp_path) == []


def test_it_reaches_the_agent_through_the_per_level_reply(tmp_path):
    # field_quality_due is what couple() serves at every converged level, so
    # this is the path the finding actually travels.
    _side(tmp_path, "side_A", [0.0] * 8, [0.0] * 8)
    found = field_quality_due(tmp_path, [1])
    assert any("IDENTICALLY ZERO" in str(f.get("finding", "")) for f in found), (
        "the finding must reach the agent from couple(), not only at hand-in")
