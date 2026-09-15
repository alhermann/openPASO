"""The two NGSolve names an agent cannot guess must fire before the run, and after it.

`cannot import name 'sym' from 'ngsolve'` is the most frequent NGSolve error
across the graded vector-elasticity cells, and a live full-task trial burned its
whole budget on `Div`, reporting "the NGSolve version installed has different
function names (lowercase div vs uppercase Div)". Both are unguessable: NGSolve
capitalises the opposite way to the other codes here, `Sym` and `Trace` act on a
matrix only, and `Id` needs a dimension.

They live in the LINT tables rather than in the knowledge payload on purpose.
The coupled knowledge reply is already over its character cap, and adding 767
characters to it was measured to evict fifteen lines of the
measured-not-modelled history rule. The lint channel has no cap.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

BAD_IMPORT = "from ngsolve import BilinearForm, sym, grad\n"
BAD_DIV = "sigma = 2*MU*eps + LAM*Div(u)*Id(2)\n"


def test_the_tables_name_sym_and_div():
    """Table content, checked directly, so this holds however the API is called."""
    text = (REPO / "src" / "tools" / "participant_lint.py").read_text()
    assert "cannot import name 'sym' from 'ngsolve'" in text
    assert "Sym(Grad(u))" in text, "the working form must be given, not only the error"
    assert "Sym of non-matrix called" in text, (
        "the run-side signal for Sym(u) on a vector is missing")
    assert "lowercase div(u)" in text, "the Div rule must name what to write instead"
    assert "Id(mesh.dim)" in text, "the identity-needs-a-dimension fact is missing"


def test_the_write_check_fires_before_the_run():
    """A script is cheapest to fix while it is still being written."""
    from tools.participant_lint import participant_findings

    sym_hits = [f for f in participant_findings(BAD_IMPORT) if "sym" in str(f)]
    assert sym_hits, "importing `sym` from ngsolve raises nothing at write time"
    assert "Sym(Grad(u))" in str(sym_hits[0]), "it must say what to write instead"

    div_hits = [f for f in participant_findings("import ngsolve\nq = Div(u)\n")
                if "no 'Div'" in str(f)]
    assert div_hits, "`Div(` raises nothing at write time"
    assert "lowercase div(u)" in str(div_hits[0])


def test_the_run_check_fires_on_the_traceback():
    """And when it was not caught at write time, the traceback must explain itself."""
    from tools.participant_lint import findings_from_output

    for signal, must_say in (
            ("ImportError: cannot import name 'sym' from 'ngsolve'", "Sym(Grad(u))"),
            ("NgException: Sym of non-matrix called", "not Sym(u)"),
            ("ImportError: cannot import name 'Div' from 'ngsolve'", "lowercase div(u)")):
        hits = findings_from_output(signal)
        assert hits, f"nothing fires on {signal!r}"
        assert must_say in str(hits[0]), f"{signal!r} does not say {must_say!r}"


def test_the_facts_are_true_of_this_install():
    """Measured, not remembered: the claims must match the installed ngsolve."""
    from backends.ngsolve.backend import _find_ngsolve_python

    python = _find_ngsolve_python()
    if not python:
        pytest.skip("no interpreter on this machine can run ngsolve")
    probe = (
        "import ngsolve as n\n"
        "from netgen.geom2d import unit_square\n"
        "m = n.Mesh(unit_square.GenerateMesh(maxh=0.5))\n"
        "u = n.VectorH1(m, order=1).TrialFunction()\n"
        "print('sym', hasattr(n, 'sym'))\n"
        "print('Div', hasattr(n, 'Div'))\n"
        "print('div', hasattr(n, 'div'))\n"
        "try:\n"
        "    n.Sym(u); print('Sym_on_vector ok')\n"
        "except Exception as e:\n"
        "    print('Sym_on_vector', type(e).__name__)\n"
        "n.Sym(n.Grad(u)); print('Sym_of_Grad ok')\n"
        "n.Id(m.dim); print('Id_dim ok')\n")
    done = subprocess.run([str(python), "-c", probe], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-1200:]
    out = done.stdout
    assert "sym False" in out, "ngsolve now exports `sym`; the lint text is out of date"
    assert "Div False" in out, "ngsolve now exports `Div`; the lint text is out of date"
    assert "div True" in out, "ngsolve no longer exports `div`"
    assert "Sym_on_vector ok" not in out, (
        "Sym now accepts a vector proxy; the 'matrix only' claim is out of date")
    assert "Sym_of_Grad ok" in out and "Id_dim ok" in out
