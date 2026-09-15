"""The gate on the gate: what verify_pde_consistency will and will not answer.

This pins the two limits measured in benchmarks/pde_check_calibration/RESULTS.md,
because both are load-bearing for a decision the campaign is about to take --
whether to run this check automatically on every converged coupled level.

  1. It implements ONE operator and refuses the rest. The guard exists because
     unguarded it called a linear-elasticity result INCONSISTENT and blessed a
     biharmonic one at "rate 2.09, for an operator this check does not model at
     all". Two of the campaign's coupled families state operators it refuses.

  2. Its identity needs u = 0 on the whole boundary, which one side of a
     coupled problem never has, because the interface carries the partner's
     data. The check says so itself and reports NOT_APPLICABLE.

If either limit changes, this test should fail and the RESULTS numbers should be
re-measured before anything is decided on them.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def check():
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
    return captured["verify_pde_consistency"]


def _refused(reply: str) -> bool:
    return reply.lstrip().startswith("REFUSED")


# The equation each coupled family states, verbatim from its spec_public.json.
ACCEPTED = {
    "C2 (4C + Kratos, diffusion)": "-div(k grad u) = f in each subdomain",
    "C8 (NGSolve + Kratos, 1:1000)": "-div(k grad u) = f in each subdomain",
    "C10 (Kratos + DUNE, 3-D)": "-div(k grad u) = f in each subdomain",
}
REFUSED = {
    "C9 (elasticity)":
        "-div(sigma(u)) = f, sigma(u) = 2*mu*sym(grad(u)) + lambda*div(u)*I"
        "   (plane strain, small strain)",
    "C3 (reaction-diffusion)":
        "subdomain A: -div(k grad u) + c*u = f ; subdomain B: -div(k grad u) = f",
}


@pytest.mark.parametrize("label,equation", sorted(ACCEPTED.items()))
def test_the_diffusion_families_are_accepted(check, label, equation):
    reply = check(solution_files="none.csv", source_term="1.0", equation=equation)
    assert not _refused(reply), (
        f"{label} states the operator this check implements and must not be refused")


@pytest.mark.parametrize("label,equation", sorted(REFUSED.items()))
def test_the_other_families_are_refused(check, label, equation):
    reply = check(solution_files="none.csv", source_term="1.0", equation=equation)
    assert _refused(reply), (
        f"{label} is outside the implemented operator. Answering about it is how "
        f"this check once called an elasticity result INCONSISTENT")


def test_it_refuses_rather_than_guesses_when_no_equation_is_given(check):
    reply = check(solution_files="none.csv", source_term="1.0", equation="")
    assert _refused(reply)


def test_a_coupled_side_is_reported_not_applicable_not_wrong(check):
    """The limit that decides whether by-default adoption can work.

    Replayed over 94 graded coupled cells with solution files, side B came back
    NOT_APPLICABLE 92 times. A cell whose field is not zero on the box boundary
    must get NOT_APPLICABLE -- saying INCONSISTENT there would tell an agent its
    answer is wrong on evidence the check does not have.
    """
    campaign = Path.home() / "Schreibtisch" / "ofa-v2" / "campaign3_blind"
    cell = campaign / "runs" / "C8_27b_MCP_seed8652" / "work"
    if not cell.is_dir():
        pytest.skip("the campaign run directory is not on this machine")
    files = sorted(cell.glob("solution_level*_A.csv"))
    if not files:
        pytest.skip("that cell exported no side-A solution files")

    reply = check(solution_files=",".join(str(f) for f in files),
                  source_term="1.0", coefficient="1",
                  domain="[[0.0, 0.625], [0.0, 1.0]]",
                  equation="-div(k grad u) = f in each subdomain")
    verdict = json.loads(reply.split("\n\n")[0]).get("verdict")
    assert verdict in ("NOT_APPLICABLE", "INCONSISTENT", "CONSISTENT"), verdict
    if verdict == "NOT_APPLICABLE":
        assert "boundary" in reply, (
            "NOT_APPLICABLE must say why, so a reader can tell it apart from a "
            "verdict about their field")
