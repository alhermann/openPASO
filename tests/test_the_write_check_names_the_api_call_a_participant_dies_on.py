"""A participant script is judged when it is written, not after the run that would have taught it.

Round 49 -- the first honest round on problems other than C1-C3 -- was lost to wall clock rather than to
knowledge: nine cells, 34.8 to 73.1 seconds per action, 37 to 75 actions each, and exactly one reached
couple(). The sink is the write-run-error-rewrite loop on the participant script. All 58 saved
step-trial fills were then re-executed and graded the way the campaign's tasks grade: 46 never wrote an
export, and every one of those died on an invented API call rather than on the physics.

Two numbers pin this gate, and both are measured here rather than asserted: it fires on NONE of the
served participants, and it names a trap in 23 of those 46 dead fills."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PARTICIPANTS = ROOT / "data" / "coupling_participants"
FILLS = ROOT / "campaign3_blind" / "step_trials"


def test_it_names_the_call_the_error_and_the_fix():
    from tools.participant_lint import participant_findings
    script = """
from skfem import Basis, MeshTri, ElementTriP1
from skfem.models.poisson import laplace
# reads imports.json, writes exports.json
m = MeshTri.init_rect(0, 1, 0, 1)
b = Basis(m, ElementTriP1(), doforder=0)
A = laplace.assemble(basis=b)
d = b.dofs(facets=[1])
"""
    f = participant_findings(script)
    assert len(f) >= 4
    joined = "\n".join(f)
    for phrase in ("'Dofs' object is not callable", "get_dofs", "'ubasis'", "init_tensor",
                   "doforder", "Measured on this install"):
        assert phrase in joined, phrase


@pytest.mark.parametrize("name", sorted(p.name for p in PARTICIPANTS.glob("participant_*.py")))
def test_it_fires_on_no_served_participant(name):
    """A gate that names an absent defect costs the agent an action for nothing."""
    from tools.participant_lint import participant_findings
    assert participant_findings((PARTICIPANTS / name).read_text()) == []


def test_a_docstring_that_quotes_a_wrong_call_is_not_a_defect():
    from tools.participant_lint import participant_findings
    script = '''
from ngsolve import Mesh
"""The geometry has no AddVertex and no AddRect; mesh.Faces() is an AttributeError."""
# CoefficientFunction(lambda x: x) is a TypeError -- do not write it
import json  # imports.json exports.json
'''
    assert participant_findings(script) == []


def test_it_would_have_named_half_the_fills_that_died_before_exporting():
    from tools.participant_lint import participant_findings
    rows = json.loads((FILLS / "regrade_fills.json").read_text())
    dead = [n for n, verdict, _ in rows if verdict == "NO EXPORTS"]
    assert len(dead) >= 40, "the re-graded fill record is missing"
    named = sum(1 for n in dead if participant_findings((FILLS / n).read_text()))
    assert named >= 20, f"the lint names only {named} of {len(dead)} dead fills; it used to name 23"


def test_the_harness_surfaces_it_on_a_write():
    """Plumbing only: the harness calls the check, the check lives in OASiS."""
    agent = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert "_participant_write_check(p, content)" in agent
    assert "_participant_write_check)" in agent or "_participant_write_check," in agent
    advisor = (ROOT / "src" / "tools" / "workspace_advisor.py").read_text()
    assert "def _participant_write_check" in advisor
    assert "participant_findings" in advisor


def test_the_check_is_quiet_on_a_file_that_is_not_python():
    sys.path.insert(0, str(ROOT / "src"))
    from tools.workspace_advisor import _participant_write_check
    assert _participant_write_check(Path("deck.yaml"), "SOME: yaml\n") == ""
