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
    assert named >= 24, f"the lint names only {named} of {len(dead)} dead fills; it used to name 26"


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


def test_the_formatted_write_check_reads_as_one_actionable_block():
    from tools.workspace_advisor import _participant_write_check
    script = """
from ngsolve import Mesh, H1
import json
# imports.json / exports.json
geo.AddVertex((0, 0))
for f in mesh.Faces():
    pass
r = f_vol.vec.vec
"""
    out = _participant_write_check(Path("participant_A.py"), script)
    assert out.startswith("\n[write check] participant_A.py:")
    assert "known to stop the run" in out
    assert "AddVertex" in out and "AddRectangle" in out
    assert "mesh.vertices" in out                      # the lowercase iterators
    assert "BaseVector" in out
    assert out.count("\n  - ") >= 3                    # one bullet per trap


def test_a_module_used_without_its_import_is_named():
    """Measured: a FEniCSx worker followed both the task and the served facts, wrote
    dolfinx.log.set_log_level(...) and died with NameError, because `from dolfinx import fem` binds no
    module. The served contracts bind it now; a script written from scratch still may not."""
    from tools.participant_lint import participant_findings
    bad = """
from dolfinx import fem
import json
# imports.json exports.json
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
"""
    f = participant_findings(bad)
    assert any("never imports the module" in x and "dolfinx" in x for x in f), f
    good = bad.replace("from dolfinx import fem", "from dolfinx import fem\nimport dolfinx")
    assert participant_findings(good) == []


def test_an_import_line_is_not_a_use():
    """`from dolfinx.fem.petsc import LinearProblem` contains 'dolfinx.' and binds nothing by that
    name; reading it as a use flagged four correct served contracts."""
    from tools.participant_lint import participant_findings
    script = """
from dolfinx.fem.petsc import LinearProblem
from dolfinx import fem
import json
# imports.json exports.json
a = 1
L = 2
p = LinearProblem(a, L, bcs=[], petsc_options_prefix="run")
"""
    assert participant_findings(script) == []


def test_it_stays_silent_on_every_fill_that_actually_exported():
    """The other half of the measurement: of the twelve fills that DID write exports, none is flagged.
    A gate that fires on working code costs an action and teaches the agent to ignore it."""
    from tools.participant_lint import participant_findings
    rows = json.loads((FILLS / "regrade_fills.json").read_text())
    worked = [n for n, verdict, _ in rows if verdict != "NO EXPORTS"]
    assert len(worked) >= 10
    noisy = [n for n in worked if participant_findings((FILLS / n).read_text())]
    assert not noisy, f"flagged scripts that ran: {noisy}"


def test_a_name_that_is_never_defined_is_named_before_the_run():
    """The leave-behind contract, checked. Measured on the DUNE 3-D trial and on five NGSolve fills:
    scripts that used CF, InnerProduct or outer_dofs without ever defining them, each dying at that
    line after the run had already been paid for."""
    from tools.participant_lint import undefined_names
    script = """
from ngsolve import Mesh, H1
import json
# imports.json exports.json
gf = CoefficientFunction(1.0)
val = InnerProduct(a, b)
"""
    f = undefined_names(script)
    assert any("CoefficientFunction" in x for x in f) and any("InnerProduct" in x for x in f), f
    assert all("never defined in this file" in x for x in f)


def test_a_lambda_parameter_is_not_an_undefined_name():
    """Six served contracts were flagged by a first version that did not read lambda arguments."""
    from tools.participant_lint import undefined_names
    assert undefined_names("f = lambda X: X[0] + X[1]\ny = f((1, 2))\n") == []


def test_a_failed_run_names_its_own_fix():
    """A failed run has already bought the diagnosis; a second run to learn it is the loop that ate
    round 49 (12-36 shell calls per cell at 35-73 seconds each)."""
    from tools.participant_lint import findings_from_output
    from tools.workspace_advisor import _participant_run_check
    console = ("Traceback (most recent call last):\n"
               "  File \"participant_A.py\", line 94, in <module>\n"
               "TypeError: 'Dofs' object is not callable\n")
    f = findings_from_output(console)
    assert f and "get_dofs" in f[0] and "Measured on this install" in f[0]
    block = _participant_run_check(console)
    assert block.startswith("\n[run check]") and "known one" in block
    assert _participant_run_check("all good, wrote exports.json") == ""
    assert _participant_run_check("") == ""


def test_the_error_table_covers_what_the_fills_actually_printed():
    """Every error the 46 dead fills printed that this project has measured a fix for must be in the
    table -- the table is the write-time traps, keyed the other way round."""
    from tools.participant_lint import findings_from_output
    printed = [
        "TypeError: 'Dofs' object is not callable",
        "BilinearForm._assemble() missing 1 required positional argument: 'ubasis'",
        "AttributeError: Attribute 'y' not found in 'w'.",
        "AttributeError: type object 'MeshTri1' has no attribute 'init_rect'",
        "AttributeError: 'SplineGeometry' object has no attribute 'AddVertex'",
        "AttributeError: 'ngsolve.comp.Mesh' object has no attribute 'Faces'",
        "AttributeError: 'ngsolve.comp.MeshNode' object has no attribute 'ndof'",
        "ImportError: cannot import name 'inverse' from 'ngsolve'",
        "AttributeError: 'ngsolve.la.BaseVector' object has no attribute 'vec'",
        "AttributeError: 'FunctionSpace' object has no attribute 'subset_dofs'",
        "AttributeError: 'Geometry' object has no attribute 'point'",
        "AttributeError: 'Form' object has no attribute 'copy'",
        "TypeError: LinearProblem.__init__() missing 1 required keyword-only argument: 'petsc_options_prefix'",
    ]
    for line in printed:
        assert findings_from_output(line), f"no fix named for a failure that was measured: {line}"


def test_the_harness_surfaces_the_run_check_too():
    agent = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert "_participant_run_check(out)" in agent
    assert "_participant_run_check)" in agent or "_participant_run_check," in agent


def test_the_run_check_is_silent_on_what_the_rounds_actually_printed():
    """A gate on run output must not fire on ordinary traffic. Measured over the 2,525 tool results in
    rounds 48 and 49: zero fires. One earlier needle (the bare word petsc_options_prefix) fired on a
    KNOWLEDGE reply that merely mentioned it, so the needle is the exact message instead."""
    from tools.participant_lint import findings_from_output
    innocent = [
        "# WHAT DECIDES THIS RUN - fenics/heat\n# LinearProblem REQUIRES the keyword petsc_options_prefix",
        "wrote 12609 chars to /abs/side_A/participant_A.py",
        "Boundaries: ('outer_bottom', 'interface')\nNDOF: 60\nInterface vertices: 7",
        "processor 0 finished normally",
    ]
    for text in innocent:
        assert findings_from_output(text) == [], f"the run check fired on ordinary output: {text[:60]}"
    real = "TypeError: LinearProblem.__init__() missing 1 required keyword-only argument: 'petsc_options_prefix'"
    assert findings_from_output(real), "the real message is no longer caught"
