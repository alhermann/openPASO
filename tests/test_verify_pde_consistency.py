"""A check the agent can run that a refinement study cannot replace.

MEASURED over 464 single-code runs: among submissions with a complete level
set, the SELF-convergence order computed from the agent's own numbers has a
median of 1.96 (bare) and 1.99 (OASiS). The discretisations converge. So a
graded order near zero is almost never the method failing to converge — it is a
field converging cleanly to the WRONG function, which a refinement study cannot
see, and which the agent's own mesh-independence verdict cannot separate either
(it catches three quarters of the wrong runs and fires on half the correct
ones).

The weak identity used here needs only the operator and source from the task
and the values the agent already wrote. Executed on this campaign's NG1 runs:
18 CONSISTENT at rate 2.09-2.21, 8 INCONSISTENT at rate -0.01 to 0.15.

WHY IT IS LEGAL UNDER BLIND GRADING: every input is public or the agent's own,
no exact solution is consulted or revealed, and one scalar identity per test
function cannot be run backwards to reconstruct a field.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

np = pytest.importorskip("numpy")

from tools.pde_consistency import check_levels  # noqa: E402


def _grid(n: int):
    """The prescribed probe arrangement: cell midpoints of an n x n grid."""
    return [((i + 0.5) / n, (j + 0.5) / n) for i in range(n) for j in range(n)]


# A manufactured solution and its exact source for K = I:
#   u = x(1-x)y(1-y),  -laplace(u) = 2y(1-y) + 2x(1-x)
def _u(x, y):
    return x * (1 - x) * y * (1 - y)


SOURCE_I = "2*y*(1-y) + 2*x*(1-x)"
BOX = [(0.0, 1.0), (0.0, 1.0)]


def _levels(fn, ns=(16, 32, 64)):
    return {k + 1: [(x, y, fn(x, y)) for x, y in _grid(n)]
            for k, n in enumerate(ns)}


def test_the_exact_solution_is_consistent():
    r = check_levels(_levels(_u), SOURCE_I, 1.0, BOX)
    assert r.verdict == "CONSISTENT", r.explanation
    assert r.levels[-1].residual < 1e-3


def test_a_scaled_field_is_caught():
    """The commonest measured defect: right shape, wrong magnitude."""
    r = check_levels(_levels(lambda x, y: 2.5 * _u(x, y)), SOURCE_I, 1.0, BOX)
    assert r.verdict == "INCONSISTENT", r.explanation


def test_a_sign_error_in_the_source_is_caught():
    r = check_levels(_levels(lambda x, y: -_u(x, y)), SOURCE_I, 1.0, BOX)
    assert r.verdict == "INCONSISTENT"


def test_an_all_zero_field_is_caught():
    r = check_levels(_levels(lambda x, y: 0.0), SOURCE_I, 1.0, BOX)
    assert r.verdict == "INCONSISTENT"


def test_the_wrong_coefficient_is_caught():
    """Solving with a scalar where the task prescribes a tensor."""
    r = check_levels(_levels(_u), SOURCE_I, [[3, -1], [-1, 2]], BOX)
    assert r.verdict == "INCONSISTENT", (
        "a field solving the identity operator must not pass as the solution "
        "of an anisotropic one"
    )


def test_a_tensor_problem_passes_with_its_own_source():
    """K = [[3,-1],[-1,2]] applied to the same u, with the matching source."""
    # -div(K grad u) = -(3 u_xx - 2 u_xy + 2 u_yy)
    #   u_xx = -2y(1-y), u_yy = -2x(1-x), u_xy = (1-2x)(1-2y)
    src = "3*2*y*(1-y) + 2*(1-2*x)*(1-2*y) + 2*2*x*(1-x)"
    r = check_levels(_levels(_u), src, [[3, -1], [-1, 2]], BOX)
    assert r.verdict == "CONSISTENT", r.explanation


def test_a_non_symmetric_tensor_is_refused_not_answered_wrongly():
    with pytest.raises(ValueError, match="not symmetric|adjoint"):
        check_levels(_levels(_u), SOURCE_I, [[3, -1], [0, 2]], BOX)


def test_a_scatter_of_points_reports_not_applicable():
    """The midpoint rule is meaningless off its grid; say so, do not guess."""
    rng = np.random.default_rng(0)
    pts = rng.random((400, 2))
    lv = {k: [(float(x), float(y), _u(x, y)) for x, y in pts]
          for k in (1, 2, 3)}
    r = check_levels(lv, SOURCE_I, 1.0, BOX)
    assert r.verdict == "NOT_APPLICABLE"
    assert "midpoint" in r.explanation or "tensor grid" in r.explanation


def test_a_non_finite_value_is_reported_not_swallowed():
    lv = _levels(_u)
    lv[2][0] = (lv[2][0][0], lv[2][0][1], float("nan"))
    r = check_levels(lv, SOURCE_I, 1.0, BOX)
    detail = " ".join(x.detail for x in r.levels)
    assert "non-finite" in detail


def test_the_explanation_says_what_to_look_at_first():
    r = check_levels(_levels(lambda x, y: 2.5 * _u(x, y)), SOURCE_I, 1.0, BOX)
    low = r.explanation.lower()
    assert "source" in low and "element-local" in low, (
        "an INCONSISTENT verdict should name the likeliest causes in order; "
        f"it said: {r.explanation}"
    )


def test_it_tells_the_agent_honesty_scores_better():
    r = check_levels(_levels(lambda x, y: 0.0), SOURCE_I, 1.0, BOX)
    assert "honestly" in r.explanation.lower()


@pytest.mark.parametrize("run,expected", [
    ("NG1_27b_BARE_seed2", "CONSISTENT"),
    ("NG1_27b_BARE_seed3", "INCONSISTENT"),
    ("NG1_27b_MCP_seed60", "INCONSISTENT"),
])
def test_against_the_real_runs_it_was_measured_on(run, expected):
    import csv
    work = ROOT / "campaign3_blind" / "runs" / run / "work"
    if not work.is_dir():
        pytest.skip(f"{run} not present")
    src = ("36*x**3*y - 20*x**3/3 - 54*x**2*y**2 - 32*x**2*y/5 + 92*x**2/15 "
           "+ 54*x*y**3 - 18*x*y**2/5 - 448*x*y/15 - 32*x/15 - 66*y**3/5 "
           "+ 2*y**2 + 112*y/15 + 8/3")
    lv = {}
    for i, f in enumerate(sorted(work.glob("solution_level*.csv")), start=1):
        rows = []
        with f.open() as fh:
            for row in csv.reader(fh):
                try:
                    rows.append(tuple(float(c) for c in row[:3]))
                except ValueError:
                    continue
        if len(rows) > 100:
            lv[i] = rows
    if len(lv) < 2:
        pytest.skip(f"{run} has too few readable levels")
    assert check_levels(lv, src, [[3, -1], [-1, 2]], BOX).verdict == expected
