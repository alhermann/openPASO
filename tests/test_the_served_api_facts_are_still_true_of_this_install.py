"""Every API fact we serve, re-derived from the library that is installed.

The lint tables tell agents what a backend exports and what it refuses, and
each claim ends "Measured on this install". That was true when it was measured.
Nothing re-checks it, so a library release that starts exporting `sym` would
leave openPASO confidently telling every agent the opposite -- and the failure
would be invisible, because the text still reads correctly.

So each assertion below re-derives the claim from the installed package rather
than restating it. A red test here does not mean the code broke; it means the
world moved and the SERVED TEXT is now wrong.

The idea is a peer session's, adopted after it pointed out that the same
machinery existed on both sides and only one of us was checking.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.participant_lint import findings_from_output  # noqa: E402

ngsolve = pytest.importorskip("ngsolve")


def _served(error_text: str) -> str:
    out = findings_from_output(error_text)
    assert out, f"nothing is served for {error_text!r}"
    return " ".join(out)


# ── the NGSolve tensor helpers ─────────────────────────────────────────────

SERVED_EXIST = ["Sym", "Trace", "Det", "Grad", "grad", "div", "Id",
                "InnerProduct", "OuterProduct", "Inv", "Cof"]
SERVED_ABSENT = ["sym", "trace", "det", "Div", "Identity", "Transpose"]


@pytest.mark.parametrize("name", SERVED_EXIST)
def test_every_name_we_say_exists_does(name):
    assert hasattr(ngsolve, name), (
        f"the served text claims ngsolve exports {name!r} and this install does "
        f"not. The claim is in the ngsolve trap in participant_lint.py")


@pytest.mark.parametrize("name", SERVED_ABSENT)
def test_every_name_we_say_is_absent_is(name):
    assert not hasattr(ngsolve, name), (
        f"the served text claims ngsolve does NOT export {name!r}, and this "
        f"install does. Agents are being told to avoid a name that now works")


def test_the_served_text_still_lists_exactly_those_names():
    """If the fact changes the message must change with it, not drift apart."""
    msg = _served("ImportError: cannot import name 'sym' from 'ngsolve'")
    for n in ("Sym", "Trace", "Id"):
        assert n in msg, f"{n} dropped out of the served message"
    for n in ("sym", "Div"):
        assert n in msg, f"{n} no longer named as absent in the served message"


# ── the working forms the text recommends ──────────────────────────────────

def _vector_space():
    from ngsolve import H1, Mesh
    import netgen.geom2d as g2
    geo = g2.SplineGeometry()
    geo.AddRectangle((0, 0), (1, 1))
    return H1(Mesh(geo.GenerateMesh(maxh=0.5)), order=1, dim=2)


def test_the_strain_form_we_recommend_builds():
    """The text says the strain is Sym(Grad(u)) -- so that must build ..."""
    from ngsolve import Sym, Grad
    u, _ = _vector_space().TnT()
    assert Sym(Grad(u)) is not None


def test_and_the_shorthand_we_warn_against_still_raises():
    """... and Sym(u) on a vector proxy must still be the error it warns of."""
    from ngsolve import Sym
    u, _ = _vector_space().TnT()
    with pytest.raises(Exception):
        Sym(u)


def test_the_identity_needs_its_dimension_as_we_say():
    from ngsolve import Id
    assert Id(2) is not None
    with pytest.raises(Exception):
        Id()


# ── the CoefficientFunction claim ──────────────────────────────────────────

def test_a_coefficient_function_still_refuses_a_callable_and_a_string():
    from ngsolve import CF
    with pytest.raises(Exception):
        CF(lambda p: p[0])
    with pytest.raises(Exception):
        CF("x^2 + y^2")


def test_and_still_accepts_the_symbolic_form_we_recommend():
    from ngsolve import CF, x, y
    assert CF(x * x + y * y) is not None


# ── scikit-fem ─────────────────────────────────────────────────────────────

def test_the_symmetric_gradient_helper_is_still_named_sym_grad():
    helpers = pytest.importorskip("skfem.helpers")
    assert hasattr(helpers, "sym_grad"), "the served text names sym_grad"
    assert not hasattr(helpers, "sym"), (
        "the served text says there is no skfem.helpers.sym, and now there is")
