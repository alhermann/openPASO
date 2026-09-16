"""The most common UFL mistake in the recorded set, and the worst-labelled one.

`|` between two UFL conditions raises a TypeError that names the operator and
not the fix, so it reads as a backend fault. One recorded run called it a
segmentation fault during grid compilation and abandoned the backend as
broken -- with `Or` already on its own import line. 22 distinct cells hit it.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.participant_lint import findings_from_output, participant_findings  # noqa: E402

SERVED = sorted((ROOT / "data" / "coupling_participants").glob("*.py"))


def _combining(findings):
    return [f for f in findings if "Or(a, b)" in f or "Or(lt(" in f]


@pytest.mark.parametrize("path", SERVED, ids=lambda p: p.name)
def test_no_served_contract_is_flagged(path):
    assert _combining(participant_findings(path.read_text())) == [], path.name


def test_two_conditionals_joined_by_a_pipe_are_named_at_write_time():
    src = ("import dolfinx\n"
           "from ufl import conditional, lt\n"
           "bc = conditional(lt(xc, a), 1, 0) | conditional(lt(yc, b), 1, 0)\n")
    assert _combining(participant_findings(src)), "the write gate stayed silent"


def test_two_comparisons_joined_inside_a_conditional_are_named_too():
    """The other written shape: the pipe sits between lt() calls, not conditionals."""
    src = ("import dune.fem\n"
           "from ufl import conditional, lt\n"
           "bc = conditional(lt(xc, a) | lt(xc, b), 1, 0)\n")
    assert _combining(participant_findings(src)), "the dune arm stayed silent"


def test_the_run_side_names_it_from_the_traceback():
    out = "TypeError: unsupported operand type(s) for |: 'Conditional' and 'Conditional'"
    fixes = findings_from_output(out)
    assert any("Or(" in f for f in fixes), fixes


def test_the_and_operator_is_named_as_well():
    assert any("And(" in f for f in
               findings_from_output("TypeError: unsupported operand type(s) for &: 'Condition'"))


@pytest.mark.parametrize("src", [
    "import dolfinx\nimport numpy as np\nmask = (arr > 1) & (arr < 2)\n",
    "import dolfinx\nimport numpy as np\nfree = np.isclose(p, q) | np.isclose(r, s)\n",
])
def test_numpy_masks_are_left_alone(src):
    """`&` and `|` on arrays are correct and common; the gate must not touch them."""
    assert _combining(participant_findings(src)) == []
