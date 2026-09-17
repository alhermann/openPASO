"""A displacement is one field, not two independent columns.

The audit's sequences are built per CSV column, so a vector answer arrives as
`selfdiff_solution_A_ux` and `..._uy` and each is judged alone. The transverse
component of a correct answer is routinely far smaller than the axial one and
sits near the coupling iteration's own floor, so it reads as "refinement is
changing nothing" while the FIELD converges perfectly well.

Measured on the fork's own audit before this view existed: FLOOR fired on
C9 seed 8791, a cell graded CORRECT. It could not have shown up earlier because
every CORRECT cell in the record before that family was scalar -- one column per
side -- so the defect had nowhere to appear.

Over all 195 graded cells with solution files:

    before   fires on CORRECT 1/32   fires on wrong 5/27
    after    fires on CORRECT 0/32   fires on wrong 5/27

One false positive removed, no catch lost.
"""
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def view():
    from tools.result_audit import _vector_order_view
    return _vector_order_view


def test_a_scalar_workspace_passes_through_unchanged(view):
    seqs = {"selfdiff_solution_A_T": [1.0, 0.5, 0.25],
            "magnitude_solution_A_T": [3.0]}
    assert view(seqs) == seqs


def test_two_components_become_one_field(view):
    """And the join is exact: these are RMS values, so they add in quadrature."""
    seqs = {"selfdiff_solution_A_ux": [3.0, 1.5],
            "selfdiff_solution_A_uy": [4.0, 2.0]}
    out = view(seqs)
    assert "selfdiff_solution_A_ux" not in out
    assert "selfdiff_solution_A_uy" not in out
    assert out["selfdiff_solution_A_u|vector"] == pytest.approx([5.0, 2.5])


def test_a_column_with_no_axis_stands_alone(view):
    """A workspace shipping `T, ux, uy` must never fold temperature into it."""
    seqs = {"selfdiff_solution_A_T": [300.0, 299.0],
            "selfdiff_solution_A_ux": [3.0, 1.5],
            "selfdiff_solution_A_uy": [4.0, 2.0]}
    out = view(seqs)
    assert out["selfdiff_solution_A_T"] == [300.0, 299.0]
    assert out["selfdiff_solution_A_u|vector"] == pytest.approx([5.0, 2.5])


def test_one_axis_alone_is_not_a_vector(view):
    seqs = {"selfdiff_solution_A_ux": [3.0, 1.5]}
    assert view(seqs) == seqs


def test_ragged_components_are_left_alone(view):
    """Guessing across mismatched lengths would be worse than not joining."""
    seqs = {"selfdiff_solution_A_ux": [3.0, 1.5],
            "selfdiff_solution_A_uy": [4.0]}
    assert view(seqs) == seqs


def test_it_is_silent_on_the_correct_cell_that_proved_the_defect():
    """The admission test, on the cell itself."""
    import asyncio

    campaign = Path.home() / "Schreibtisch" / "ofa-v2" / "campaign3_blind"
    work = campaign / "runs" / "C9_27b_MCP_seed8791" / "work"
    if not work.is_dir():
        pytest.skip("the campaign run directory is not on this machine")

    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools
    load_all_backends()
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    register_consolidated_tools(_Recorder())
    reply = asyncio.run(captured["audit_results"](work_dir=str(work),
                                                  claimed_order=2.0))
    named = [ln.strip() for ln in reply.splitlines()
             if any(k in ln for k in ("FLOOR", "ORDER BELOW", "BACKWARDS",
                                      "NEAR-ZERO"))]
    assert not named, (
        "the audit speaks about a cell graded CORRECT:\n  "
        + "\n  ".join(n[:160] for n in named[:5]))
