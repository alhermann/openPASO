"""The check that separates a right answer from a converged wrong one, run for you.

Two wordings of the invitation to call verify_pde_consistency were measured and
both failed: 2 calls against 14 asks, then 0 against 7 after the ask was moved
to the front of the reply and given its evidence. So couple() now takes the
four strings the agent already has and runs the check itself on every converged
level.

Nothing here reads a task file. The agent hands over its own strings and names
its own files; openPASO guesses no filename and stores nothing.
"""
import csv
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "tools"))

from pde_consistency import check_levels  # noqa: E402

CELL = ROOT / "campaign3_blind/runs/C8_27b_MCP_seed8453/work"
# subdomain A of the recorded two-material problem, as its task states them
SRC_A = ("-18*x**3*y/5 + 2*x**3/5 + 5063*x**2*y/20000 - 95021*x**2/60000 "
         "- 18*x*y**3/5 + 6*x*y**2/5 + 14607*x*y/4000 + 1669*x/2000 "
         "+ 5063*y**3/60000 - 95021*y**2/60000 + 14993*y/10000")
BOX_A = [[0.0, 0.625], [0.0, 1.0]]


def _load(p):
    rows = []
    with open(p) as fh:
        for r in csv.reader(fh):
            try:
                rows.append(tuple(float(c) for c in r))
            except ValueError:
                continue
    return [r[:3] for r in rows if len(r) >= 3]


def test_the_tool_and_couple_share_one_body():
    """couple() must run the SAME check, not a second implementation of it."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "def _pde_consistency_body(" in src
    assert src.count("_pde_consistency_body(") >= 3, (
        "the body must be called by the tool AND by couple()")


def test_couple_accepts_the_four_strings_and_the_files():
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "pde_check: str = \"\"" in src
    for key in ("solution_files", "equation", "source", "coefficient", "domain"):
        assert f'_one.get("{key}"' in src or f'"{key}"' in src, key


def test_it_names_no_deliverable_and_opens_no_task_file():
    """Option B: openPASO checks what it is handed and guesses nothing."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    # anchor on the branch itself: the initialisation moved to function scope
    # when it turned out a non-converged call read it unbound.
    i = src.index("            if pde_check:")
    block = src[i:i + 4000]
    assert "solution_files" in block, "the agent must name its own files"
    for forbidden in ("task.txt", "spec_public", "problems/", "solution_level"):
        assert forbidden not in block, forbidden


def test_an_inconsistent_verdict_leads_the_reply():
    """It leads, and after the test function was fixed it leads as a directive.

    The earlier softening was right against 1 false accusation in 14. With
    v = prod sin^2 both boundary terms of the weak identity vanish, and the
    re-measured rate over both sides of every graded C2/C8/C10 cell is 0 false
    alarms of 28, 28 wrong sides named, 118 of 118 answered.
    """
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "YOUR FIELD DOES NOT SATISFY THE EQUATION YOU GAVE" in src
    i = src.index("YOUR FIELD DOES NOT SATISFY THE EQUATION YOU GAVE")
    assert "_lead = (" in src[i - 400:i], "it must LEAD, not trail"
    assert "on NONE that are right" in src[i:i + 1400], (
        "the directive is earned by 0 false alarms of 28 correct-cell sides, "
        "and the reply must carry that number where it fires")


def test_without_the_strings_the_reply_asks_for_them_and_says_why():
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "NO LEVEL HAS BEEN CHECKED AGAINST ITS OWN EQUATION" in src
    i = src.index("NO LEVEL HAS BEEN CHECKED")
    ask = src[i:i + 1800]
    assert "converged cleanly to the" in ask, "it must say what the check is FOR"
    assert "reference answer" in ask, "it must say the check needs no answer key"


@pytest.mark.skipif(not CELL.is_dir(), reason="the recorded cell is not present")
def test_the_check_confirms_a_verified_correct_field():
    """The one CORRECT coupled cell on record must come back CONSISTENT."""
    fl = sorted((CELL).glob("solution_level*_A.csv"),
                key=lambda q: int(q.name.split("level")[1][0]))
    assert len(fl) >= 2, "fixture needs at least two levels"
    r = check_levels({i: _load(q) for i, q in enumerate(fl)}, SRC_A, 1.0, BOX_A).as_dict()
    assert r["verdict"] == "CONSISTENT", r


@pytest.mark.skipif(not CELL.is_dir(), reason="the recorded cell is not present")
def test_mesh_nodes_are_refused_rather_than_answered_wrongly():
    """The probe grid is required; on the participant's own scatter it says nothing.

    This is why couple() asks for solution_files instead of reaching for the
    per-level dumps beside each exports.json.
    """
    fl = sorted((CELL / "side_A").glob("field_level*.csv"),
                key=lambda q: int(q.stem.split("level")[1]))
    if len(fl) < 2:
        pytest.skip("no per-level dumps in this cell")
    r = check_levels({i: _load(q) for i, q in enumerate(fl)}, SRC_A, 1.0, BOX_A).as_dict()
    assert r["verdict"] == "NOT_APPLICABLE"
    assert "uniform" in r["explanation"] or "midpoint" in r["explanation"]
