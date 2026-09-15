"""The gate on the gate: what verify_pde_consistency will and will not answer.

This pins the two limits measured in benchmarks/pde_check_calibration/RESULTS.md,
because both are load-bearing for a decision the campaign is about to take --
whether to run this check automatically on every converged coupled level.

  1. It implements ONE operator and refuses the rest. The guard exists because
     unguarded it called a linear-elasticity result INCONSISTENT and blessed a
     biharmonic one at "rate 2.09, for an operator this check does not model at
     all". Two of the campaign's coupled families state operators it refuses.

  2. It now ANSWERS a side whose boundary carries the partner's data, which is
     every side of a partitioned coupling. It did not use to: its test function
     vanished on the box but its slope did not, so the weak identity needed
     u = 0 there as well and the check refused 92 of 94 coupled sides. A test
     function whose value AND slope vanish removes that limit. The separation
     it buys is pinned below on a manufactured pair, so it is measured here and
     not only in the replay.

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


def _write_levels(tmp_path, amplitude):
    """A field that is NOT zero on the boundary, at three refinements.

    u = sin(pi x) sin(pi y) + x solves -lap u = 2 pi^2 sin(pi x) sin(pi y) on
    the unit square and carries u = x on three of its four faces, which is the
    shape of a coupled subdomain: the interface holds the partner's data. The
    `amplitude` scales only the sine part, so 1.0 is the field that solves the
    stated equation and anything else is a field that does not.
    """
    import math

    files = []
    for level, n in enumerate((32, 64, 128), start=1):
        path = tmp_path / f"solution_level{level}.csv"
        lines = ["x,y,u"]
        for i in range(n):
            x = (i + 0.5) / n
            for j in range(n):
                y = (j + 0.5) / n
                u = amplitude * math.sin(math.pi * x) * math.sin(math.pi * y) + x
                lines.append(f"{x!r},{y!r},{u!r}")
        path.write_text("\n".join(lines) + "\n")
        files.append(str(path))
    return ",".join(files)


def _verdict(check, files):
    reply = check(solution_files=files,
                  source_term="2*pi**2*sin(pi*x)*sin(pi*y)", coefficient="1",
                  domain="[[0.0, 1.0], [0.0, 1.0]]",
                  equation="-div(k grad u) = f")
    return json.loads(reply.split("\n\n")[0]).get("verdict")


def test_it_answers_a_field_that_is_not_zero_on_the_boundary(check, tmp_path):
    """The limit that decided whether by-default adoption can work at all.

    A coupled side used to come back NOT_APPLICABLE -- 92 of 94 graded cells --
    so the check could not speak about almost any coupled problem, which is
    precisely where it was about to run by default. This field carries data on
    its boundary and solves its equation, so the answer must be CONSISTENT.
    """
    assert _verdict(check, _write_levels(tmp_path, 1.0)) == "CONSISTENT"


def test_it_still_catches_a_wrong_field_with_the_same_boundary_data(check, tmp_path):
    """Answering more cells is worth nothing if it answers CONSISTENT to all.

    Same boundary trace, same source, amplitude off by half. Measured over the
    94 graded coupled cells this separation is 14 of 17 wrong cells caught
    against 0 of 14 correct ones accused (RESULTS.md); here it is pinned on a
    case with no campaign data behind it.
    """
    assert _verdict(check, _write_levels(tmp_path, 0.5)) == "INCONSISTENT"
