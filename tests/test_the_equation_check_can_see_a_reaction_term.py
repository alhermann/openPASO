"""A side with a reaction term could not be checked, and was told it was wrong.

The weak identity this check uses holds for any self-adjoint operator, and
-div(K grad u) + c u with a constant c is self-adjoint exactly as constant-K
diffusion is: the adjoint of the test function gains + c v and nothing else
changes. Without that term the check judges the field against a DIFFERENT
equation than the one the agent solved.

MEASURED on the two coupled development runs that graded correct this week
(C3 seeds 8862 and 8864, whose side A carries a reaction of 10 and whose side B
carries none), over their own delivered per-level probe files:

    side A, reaction in the operator     1.79e-02 -> 5.18e-03 -> 1.63e-03   CONSISTENT
    side A, reaction omitted             3.61e-01 -> 3.54e-01 -> 3.52e-01   INCONSISTENT

So the check called the one field that is right "converging to something that is
not the solution of the stated problem", flat at 35 % -- and it would do that on
every reacting side of every coupled problem. Side B, which has no reaction, is
unaffected either way (5.8e-03 -> 2.0e-03 -> 1.2e-03).

The separation is kept: on the same correct files a 5 % scaling error reads
4.4e-02 -> 4.8e-02 -> 4.9e-02 (flat, INCONSISTENT) and a 20 % error 1.9e-01
flat. What the check cannot see, and says so, is a perturbation lying in the
kernel of the operator -- adding a harmonic function to a pure-diffusion side
changes only the boundary data, which an interior identity with a compactly
supported test function is blind to by construction.

The tests below manufacture their own field, so they depend on no recorded run.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.pde_consistency import check_levels                 # noqa: E402

BOX = [(0.0, 1.0), (0.0, 0.625)]
K = 1.0
C = 10.0


def _u(x, y):
    return math.sin(math.pi * x) * (y + 0.3)


SOURCE = f"{math.pi ** 2} * sin(pi*x) * (y + 0.3) + {C} * sin(pi*x) * (y + 0.3)"


def _levels(scale: float = 1.0, add_harmonic: float = 0.0) -> dict:
    """Cell-midpoint grids, the shape a prescribed probe grid has."""
    out = {}
    for lvl, n in ((1, 8), (2, 16), (3, 32)):
        hx = (BOX[0][1] - BOX[0][0]) / n
        hy = (BOX[1][1] - BOX[1][0]) / n
        rows = []
        for i in range(n):
            for j in range(n):
                x = BOX[0][0] + (i + 0.5) * hx
                y = BOX[1][0] + (j + 0.5) * hy
                rows.append((x, y, scale * _u(x, y) + add_harmonic * x))
        out[lvl] = rows
    return out


def _verdict(res) -> str:
    return str(res.as_dict().get("verdict"))


def test_a_reacting_field_is_consistent_when_the_reaction_is_in_the_operator():
    res = check_levels(_levels(), SOURCE, K, BOX, reaction=C)
    assert _verdict(res) == "CONSISTENT", res.as_dict()


def test_the_same_field_reads_inconsistent_when_the_reaction_is_dropped():
    """This is the false accusation the term removes."""
    res = check_levels(_levels(), SOURCE, K, BOX)
    assert _verdict(res) == "INCONSISTENT", res.as_dict()


def test_a_wrong_field_is_still_caught_with_the_reaction_in():
    res = check_levels(_levels(scale=1.05), SOURCE, K, BOX, reaction=C)
    assert _verdict(res) == "INCONSISTENT", res.as_dict()
    res = check_levels(_levels(scale=1.20), SOURCE, K, BOX, reaction=C)
    assert _verdict(res) == "INCONSISTENT", res.as_dict()


def test_the_residual_falls_for_the_right_field_and_not_for_the_wrong_one():
    good = [l["relative_weak_residual"] for l in
            check_levels(_levels(), SOURCE, K, BOX, reaction=C).as_dict()["levels"]]
    bad = [l["relative_weak_residual"] for l in
           check_levels(_levels(scale=1.2), SOURCE, K, BOX, reaction=C).as_dict()["levels"]]
    assert good[-1] < good[0] / 4, good
    assert bad[-1] > bad[0] / 2, bad


def test_zero_reaction_is_exactly_the_old_behaviour():
    a = check_levels(_levels(), SOURCE, K, BOX, reaction=0.0).as_dict()
    b = check_levels(_levels(), SOURCE, K, BOX).as_dict()
    assert a == b


def test_the_tool_accepts_a_reaction_and_an_equation_that_states_one(tmp_path):
    """The served tool, not just the body: an equation with a reaction term is
    inside the implemented operator, not refused."""
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools

    class _MCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    mcp = _MCP()
    register_consolidated_tools(mcp)
    files = []
    for lvl, rows in sorted(_levels().items()):
        p = tmp_path / f"solution_level{lvl}.csv"
        p.write_text("x,y,u\n" + "".join(f"{x},{y},{u}\n" for x, y, u in rows))
        files.append(str(p))
    out = mcp.tools["verify_pde_consistency"](
        solution_files=",".join(files), source_term=SOURCE, coefficient="1.0",
        domain=json.dumps([[0.0, 1.0], [0.0, 0.625]]),
        equation="-div(k grad u) + c u = f", reaction=str(C))
    assert "REFUSED" not in out[:200], out[:600]
    assert '"verdict": "CONSISTENT"' in out, out[:900]
