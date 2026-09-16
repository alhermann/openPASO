"""A field right in the interior and wrong on a face is invisible to this check.

This is the limit of the method, not a threshold, and it is pinned here because
the check is otherwise easy to describe as "the one that separates a right
answer from a converged wrong one" -- which it is not.

The campaign's C9 seed 8741 coupled three levels with both codes proven and
exchanged NOTHING: its interface traction columns read 9.5e-18 and 0.0, so each
side returned the answer it would have returned with no partner. Measured, the
equation check calls it CONSISTENT on both sides with residuals falling 18.3x
and 21.1x -- FASTER than either genuinely correct cell (16.6x/15.4x for 8631,
16.4x/16.9x for 8791). It is not a near miss that a better rule would catch:
each field really does satisfy its own equation in the interior.

So the campaign's interface check and this one are complementary. Neither
substitutes for the other, and a build running only one leaves a whole failure
mode unwatched.
"""
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LAM, MU = 480.0, 1200.0
BOX = [(0.0, 1.0), (0.0, 0.625)]


def _u_with_a_free_interface(x, y):
    """A field solving its own equation with the WRONG condition on one face.

    Manufactured the same way a dead exchange arises: the interface at y = 0.625
    carries a homogeneous natural condition instead of the partner's traction.
    Here that is expressed by a field that satisfies -div(sigma(u)) = f for the
    f derived FROM IT, while being nothing like the coupled solution.
    """
    return (math.sin(math.pi * x) * math.sin(math.pi * y),
            math.sin(math.pi * x) * math.sin(math.pi * y) / 2)


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


def test_a_field_satisfying_its_own_equation_reads_consistent_whatever_its_boundary(
        check, tmp_path):
    """The mechanism, on manufactured data: the interior is all this check reads.

    f is derived from u, so u solves its own equation exactly. Nothing about the
    boundary trace enters the identity, so the verdict cannot depend on it --
    which is the whole reason a dead exchange passes.
    """
    sp = pytest.importorskip("sympy")
    import json

    x, y = sp.symbols("x y")
    u1 = sp.sin(sp.pi * x) * sp.sin(sp.pi * y)
    u2 = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / 2
    exx, eyy = sp.diff(u1, x), sp.diff(u2, y)
    exy = (sp.diff(u1, y) + sp.diff(u2, x)) / 2
    tr = exx + eyy
    sxx, syy, sxy = 2 * MU * exx + LAM * tr, 2 * MU * eyy + LAM * tr, 2 * MU * exy
    fx = sp.simplify(-(sp.diff(sxx, x) + sp.diff(sxy, y)))
    fy = sp.simplify(-(sp.diff(sxy, x) + sp.diff(syy, y)))
    U1, U2 = sp.lambdify((x, y), u1), sp.lambdify((x, y), u2)

    files = []
    for level, n in enumerate((16, 32, 64), start=1):
        hx = (BOX[0][1] - BOX[0][0]) / n
        hy = (BOX[1][1] - BOX[1][0]) / n
        path = tmp_path / f"solution_level{level}_A.csv"
        lines = ["x,y,ux,uy"]
        for i in range(n):
            xv = BOX[0][0] + (i + 0.5) * hx
            for j in range(n):
                yv = BOX[1][0] + (j + 0.5) * hy
                lines.append(f"{xv!r},{yv!r},{U1(xv, yv)!r},{U2(xv, yv)!r}")
        path.write_text("\n".join(lines) + "\n")
        files.append(str(path))

    reply = check(solution_files=",".join(files),
                  source_term=json.dumps([str(fx), str(fy)]),
                  coefficient=json.dumps({"lambda": LAM, "mu": MU}),
                  domain=json.dumps([list(b) for b in BOX]),
                  equation="-div(sigma(u)) = f")
    verdict = json.loads(reply.split("\n\n")[0]).get("verdict")
    assert verdict == "CONSISTENT", (
        "a field that solves its own equation must read CONSISTENT; if this "
        "changed, the limit recorded in RESULTS.md has moved and the dead-exchange "
        "measurement needs redoing")


def test_the_served_text_does_not_claim_to_separate_right_from_wrong(check):
    """Wording matters when a check has a blind spot this size."""
    import json

    reply = check(solution_files="", source_term="1", coefficient="1",
                  domain="[[0,1],[0,1]]", equation="-lap(u) = f")
    lowered = reply.lower()
    assert "the one check that separates" not in lowered, (
        "this check cannot see a wrong condition on a face -- measured on a "
        "coupled pair that exchanged nothing and read CONSISTENT on both sides "
        "with residuals falling faster than either correct cell")
