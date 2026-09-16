"""The largest ungated failure in the recorded set: 271 hits across 120 cells.

Three spellings of one mistake -- a Python callable, a string, or a sympy
expression handed to a CoefficientFunction. It lands hardest on the two
problem families that have never once produced a coupled level.

The fact is measured on this install: ngsolve exports x, y and z, and
CF(x*x + y*y) builds while CF(lambda ...) and CF('x^2 + y^2') raise.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.participant_lint import findings_from_output, participant_findings  # noqa: E402

SERVED = sorted((ROOT / "data" / "coupling_participants").glob("*.py"))


def _sym(findings):
    return [f for f in findings if "symbolic coordinates" in f]


@pytest.mark.parametrize("path", SERVED, ids=lambda p: p.name)
def test_no_served_contract_is_flagged(path):
    assert _sym(participant_findings(path.read_text())) == [], path.name


@pytest.mark.parametrize("payload", [
    "<function <lambda> at 0x7f8e1a2de>",
    "<function f_source at 0x7f8e1a2de>",
    "x of type <class 'str'>",
    "x^2 + y^2 of type <class 'str'>",
])
def test_every_recorded_spelling_is_answered_from_the_traceback(payload):
    out = f"ValueError: Cannot make CoefficientFunction from {payload}"
    fixes = findings_from_output(out)
    assert any("symbolic coordinates" in f for f in fixes), fixes
    assert any("from ngsolve import x, y, z" in f for f in fixes), fixes


def test_the_coordinate_name_error_is_answered_too():
    fixes = findings_from_output("NameError: name 'x' is not defined. Did you mean: 'dx'?")
    assert any("volume measure" in f for f in fixes), fixes


def test_a_lambda_is_caught_before_the_run():
    src = ("import ngsolve\n"
           "from ngsolve import CF\n"
           "f = CF(lambda p: p[0] * p[1])\n")
    assert _sym(participant_findings(src)), "the write gate stayed silent"


def test_the_working_form_is_not_flagged():
    src = ("import ngsolve\n"
           "from ngsolve import CF, x, y\n"
           "f = CF(x * x + y * y)\n")
    assert _sym(participant_findings(src)) == []


def test_a_lambda_that_is_not_a_coefficient_function_is_left_alone():
    """Sorting keys and numpy callbacks are ordinary Python and must not fire."""
    src = ("import ngsolve\n"
           "pts.sort(key=lambda v: v.point[1])\n"
           "vals = list(map(lambda z: z * 2, xs))\n")
    assert _sym(participant_findings(src)) == []
