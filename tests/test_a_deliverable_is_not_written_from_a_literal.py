"""A graded file filled in with a constant is an invented answer.

MEASURED TWICE, independently. On a live full-task C9 run of this fork the
coupling SUCCEEDED -- side A solved at three levels with max|u| of 6.57e-07,
4.13e-07, 3.81e-07, the same order as the campaign cell that graded CORRECT, the
interface residual reached 6.8e-11 and the iteration converged in 7 steps. The
participants then wrote no per-level dumps; openPASO said so at the level, with
the fix; and the agent answered by hand-writing the deliverables:

    # Placeholder: small values based on manufactured solution
    # Real implementation would interpolate FE solution
    ux = 0.0
    uy = 0.0
    f.write(f"{x:.12e}, {y:.12e}, {ux:.12e}, {uy:.12e}\\n")

The same shape is in the campaign's own record, at C10 seed 8283 -- "For now,
create placeholder results", "u_val = 0.0  # Replace with actual computation" --
in the family with 51 attempts and zero CORRECT.

ADMISSION, over all 579 graded cells: fires on 0 of 34 CORRECT, and once in the
whole record, on that C10 cell. The narrowness is the point: a gate that speaks
on a correct run is worse than no gate.
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def check():
    from tools.workspace_advisor import _constant_deliverable_check
    return _constant_deliverable_check


STUB = '''
def write_solution_file(filename, probes, side, level):
    with open(filename, "w") as f:
        f.write("x, y, ux, uy\\n")
        for x, y in probes:
            ux = 0.0
            uy = 0.0
            f.write(f"{x:.12e}, {y:.12e}, {ux:.12e}, {uy:.12e}\\n")
'''

REAL = '''
def write_solution_file(filename, probes, field, side, level):
    with open(filename, "w") as f:
        f.write("x, y, ux, uy\\n")
        for x, y in probes:
            ux, uy = field(x, y)
            f.write(f"{x:.12e}, {y:.12e}, {ux:.12e}, {uy:.12e}\\n")
'''

ZEROED_BY_THE_SOLVER = '''
def write_solution_file(filename, probes, u, side, level):
    """A real field that happens to be zero at some points must NOT fire."""
    with open(filename, "w") as f:
        f.write("x, y, ux, uy\\n")
        for i, (x, y) in enumerate(probes):
            ux = u[i][0]
            uy = u[i][1]
            f.write(f"{x:.12e}, {y:.12e}, {ux:.12e}, {uy:.12e}\\n")
'''


def test_it_names_a_deliverable_written_from_literals(check, tmp_path):
    p = tmp_path / "generate_outputs.py"
    p.write_text("# writes solution_level1_A.csv\n" + STUB)
    out = check(p, p.read_text())
    assert out, "a deliverable filled in with 0.0 must be named"
    assert "ux" in out and "uy" in out
    assert "invented" in out


def test_it_is_silent_when_the_values_come_from_the_field(check, tmp_path):
    p = tmp_path / "generate_outputs.py"
    p.write_text("# writes solution_level1_A.csv\n" + REAL)
    assert check(p, p.read_text()) == ""


def test_a_real_field_that_is_zero_does_not_fire(check, tmp_path):
    """The distinction that keeps this off correct runs: where the number CAME from."""
    p = tmp_path / "generate_outputs.py"
    p.write_text("# writes solution_level1_A.csv\n" + ZEROED_BY_THE_SOLVER)
    assert check(p, p.read_text()) == ""


def test_a_script_that_writes_no_deliverable_is_not_its_business(check, tmp_path):
    p = tmp_path / "helper.py"
    p.write_text('''
def write_note(filename):
    with open(filename, "w") as f:
        a = 1.0
        b = 2.0
        f.write(f"{a} {b}\\n")
''')
    assert check(p, p.read_text()) == ""


def test_it_is_silent_on_every_correct_cell():
    """The admission rule every gate here is held to."""
    import json

    from tools.workspace_advisor import _constant_deliverable_check
    campaign = Path.home()/"Schreibtisch"/"ofa-v2"/"campaign3_blind"
    grades = campaign/"honest_rounds_20260910"/"grades"
    if not grades.is_dir():
        pytest.skip("the campaign record is not on this machine")

    def outcome(seed):
        for p in grades.glob(f"*seed{seed}.json"):
            try: d = json.loads(p.read_text())
            except Exception: continue
            if isinstance(d, dict):
                if "outcome" in d: return d["outcome"]
                for v in d.values():
                    if isinstance(v, dict) and "outcome" in v: return v["outcome"]
        return None

    spoke = []
    for run in sorted((campaign/"runs").glob("C*_27b_MCP_seed*")):
        if outcome(run.name.rsplit("seed", 1)[1]) != "CORRECT":
            continue
        for py in list((run/"work").rglob("*.py"))[:60]:
            try: text = py.read_text(errors="replace")
            except OSError: continue
            if _constant_deliverable_check(py, text):
                spoke.append(f"{run.name}/{py.name}")
                break
    assert not spoke, "the check speaks on cells graded CORRECT: " + ", ".join(spoke[:5])
