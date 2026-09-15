"""Linear elasticity is the second operator this check models, and it is calibrated.

It used to be the headline REFUSAL, and rightly: the scalar adjoint applied to a
displacement field once reported it converging to the wrong solution, which was
meaningless. But for CONSTANT lambda and mu the elasticity operator is
self-adjoint exactly as constant-K diffusion is, so the same weak identity holds
with L*v = -div(sigma(v)), and the sin^2 test function kills both boundary terms
for a coupled side too.

That refusal covered the campaign's C9 family -- the one coupled family with a
recent CORRECT cell -- so the single coupled problem that had started working
was the one nothing could check.

These tests use a manufactured field, not campaign data, because C9 has exactly
ONE graded CORRECT cell in the whole record and a false-alarm rate over a
denominator of one is not a measurement.
"""
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

LAM, MU = 480.0, 1200.0
BOX = [[0.0, 1.0], [0.0, 0.625]]
EQUATION = ("-div(sigma(u)) = f, sigma(u) = 2*mu*sym(grad(u)) + "
            "lambda*div(u)*I   (plane strain, small strain)")


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


@pytest.fixture(scope="module")
def manufactured():
    """u = (sin sin + x, (sin sin)/2 + y/2), f from it symbolically.

    Deliberately NOT zero on the boundary: that is the shape of a coupled side,
    where the interface carries the partner's displacement.
    """
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    u1 = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) + x
    u2 = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) / 2 + y / 2
    exx, eyy = sp.diff(u1, x), sp.diff(u2, y)
    exy = (sp.diff(u1, y) + sp.diff(u2, x)) / 2
    tr = exx + eyy
    sxx, syy, sxy = 2 * MU * exx + LAM * tr, 2 * MU * eyy + LAM * tr, 2 * MU * exy
    fx = -(sp.diff(sxx, x) + sp.diff(sxy, y))
    fy = -(sp.diff(sxy, x) + sp.diff(syy, y))
    return (sp.lambdify((x, y), u1), sp.lambdify((x, y), u2),
            str(sp.simplify(fx)), str(sp.simplify(fy)))


def _write(tmp_path, manufactured, scale_x=1.0, scale_y=1.0):
    u1, u2, _, _ = manufactured
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
                lines.append(f"{xv!r},{yv!r},"
                             f"{scale_x * u1(xv, yv)!r},{scale_y * u2(xv, yv)!r}")
        path.write_text("\n".join(lines) + "\n")
        files.append(str(path))
    return ",".join(files)


def _verdict(check, files, manufactured, **over):
    _, _, fx, fy = manufactured
    kwargs = dict(solution_files=files, source_term=json.dumps([fx, fy]),
                  coefficient=json.dumps({"lambda": LAM, "mu": MU}),
                  domain=json.dumps(BOX), equation=EQUATION)
    kwargs.update(over)
    reply = check(**kwargs)
    if reply.lstrip().startswith("REFUSED"):
        return "REFUSED", reply
    return json.loads(reply.split("\n\n")[0]).get("verdict"), reply


def test_elasticity_is_no_longer_refused(check, tmp_path, manufactured):
    verdict, reply = _verdict(check, _write(tmp_path, manufactured), manufactured)
    assert verdict != "REFUSED", reply[:400]


def test_a_correct_displacement_field_reads_consistent(check, tmp_path, manufactured):
    verdict, reply = _verdict(check, _write(tmp_path, manufactured), manufactured)
    assert verdict == "CONSISTENT", reply[:600]


def test_three_percent_off_in_one_component_is_caught(check, tmp_path, manufactured):
    """The component-wise test is the point: one right component must not cover
    for a wrong one, so both momentum equations are tested and the worst wins."""
    files = _write(tmp_path, manufactured, scale_y=0.97)
    verdict, reply = _verdict(check, files, manufactured)
    assert verdict == "INCONSISTENT", reply[:600]


def test_it_refuses_rather_than_guesses_a_single_source_expression(
        check, tmp_path, manufactured):
    _, _, fx, _ = manufactured
    verdict, reply = _verdict(check, _write(tmp_path, manufactured), manufactured,
                              source_term=fx)
    assert verdict == "REFUSED" and "components" in reply, reply[:400]


def test_it_refuses_rather_than_guesses_the_lame_constants(
        check, tmp_path, manufactured):
    verdict, reply = _verdict(check, _write(tmp_path, manufactured), manufactured,
                              coefficient="480")
    assert verdict == "REFUSED" and "lambda" in reply, reply[:400]
