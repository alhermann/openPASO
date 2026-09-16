"""Tier-2: three dg_advection_diffusion statements on one SIPG family.

  dg_advection_diffusion#0  CellDiameter must come from dune.ufl as
                            DuneCellDiameter; ufl's own reaches model
                            generation and then raises.
  dg_advection_diffusion#2  advection makes the matrix nonsymmetric, so CG is
                            the wrong Krylov method and GMRES is the right one.
  dg_advection_diffusion#3  an unseen polynomial form JIT-compiles C++ before
                            its first solve, and the same form reused does not.

The form is the one openPASO's own generator emits
(src/backends/dune/generators/dg_advection_diffusion.py): SIPG diffusion with
the interior penalty, an upwind advection term, and weak Dirichlet data.

MUTATION CONTROL: T2_MUTATE=1 solves the CG slot with GMRES, so the matrix is
no longer asked the question the claim is about.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MUTATE = os.environ.get("T2_MUTATE") == "1"
FAILURES = []


def note(ok: bool, key: str, detail: str = "") -> None:
    print(f"{key}={ok}" + (f"   # {detail}" if detail else ""))
    if not ok:
        FAILURES.append(key)


# ── #0  the CellDiameter that reaches model generation and then refuses ──────
#
# This is a python-side codegen failure, so it costs no C++ build at all.
def cell_diameter_probe() -> None:
    from dune.alugrid import aluConformGrid
    from dune.fem.scheme import galerkin
    from dune.fem.space import dglagrange
    from dune.grid import cartesianDomain
    from ufl import CellDiameter, TestFunction, TrialFunction, dx, grad, inner

    grid = aluConformGrid(cartesianDomain([0, 0], [1, 1], [4, 4]), dimgrid=2)
    space = dglagrange(grid, order=1)
    u, v = TrialFunction(space), TestFunction(space)
    h = CellDiameter(space)
    raised, kind, text = False, "", ""
    try:
        galerkin([inner(grad(u), grad(v)) * dx + (1.0 / h) * u * v * dx == v * dx],
                 solver="gmres")
    except Exception as exc:                                  # noqa: BLE001
        raised, kind, text = True, type(exc).__name__, str(exc)
    note(raised, "ufl_celldiameter_is_refused", kind)
    note(kind == "TypeError", "and_it_is_a_TypeError", kind)
    note("DuneCellDiameter" in text, "and_it_names_the_replacement",
         "the message spells out `from dune.ufl import DuneCellDiameter`")


# ── the shared SIPG family ───────────────────────────────────────────────────
def build(nx: int, degree: int, source_value: float, solver: str):
    """One SIPG advection-diffusion scheme, solved once."""
    from dune.alugrid import aluConformGrid
    from dune.fem.scheme import galerkin
    from dune.fem.space import dglagrange
    from dune.grid import cartesianDomain
    from dune.ufl import Constant, DuneCellDiameter as CellDiameter
    from ufl import (FacetNormal, TestFunction, TrialFunction, as_vector, avg,
                     dS, dot, ds, dx, grad, inner, jump)

    epsilon, beta = 1.0, 20.0
    velocity = as_vector([2.0, 1.0])
    grid = aluConformGrid(cartesianDomain([0, 0], [1, 1], [nx, nx]), dimgrid=2)
    space = dglagrange(grid, order=degree)
    u, v = TrialFunction(space), TestFunction(space)
    n = FacetNormal(space)
    h = CellDiameter(space)
    h_avg = (h("+") + h("-")) / 2.0
    src = Constant(source_value, name="source")
    bval = Constant(0.0, name="boundary_value")

    vn = dot(velocity, n)
    upwind = ((vn("+") + abs(vn("+"))) / 2.0 * u("+")
              + (vn("+") - abs(vn("+"))) / 2.0 * u("-"))
    interior = (epsilon * inner(grad(u), grad(v)) * dx
                - epsilon * inner(avg(grad(u)), jump(v, n)) * dS
                - epsilon * inner(jump(u, n), avg(grad(v))) * dS
                + beta * degree * degree / h_avg * epsilon
                * inner(jump(u, n), jump(v, n)) * dS)
    advection = (-inner(u * velocity, grad(v)) * dx
                 + upwind * jump(v) * dS
                 + (vn + abs(vn)) / 2.0 * u * v * ds)
    bdiff = (-epsilon * dot(grad(u), n) * v * ds
             - epsilon * dot(grad(v), n) * u * ds
             + beta * degree * degree / h * epsilon * u * v * ds)
    brhs = (-epsilon * dot(grad(v), n) * bval * ds
            + beta * degree * degree / h * epsilon * bval * v * ds
            - (vn - abs(vn)) / 2.0 * bval * v * ds)

    scheme = galerkin([interior + advection + bdiff == src * v * dx + brhs],
                      solver=solver,
                      parameters={"linear.tolerance": 1e-10,
                                  "linear.maxiterations": 2000,
                                  "linear.preconditioning.method": "ssor"})
    uh = space.interpolate(0, name="u")
    info = scheme.solve(target=uh)
    return info, space


# ── #2  a nonsymmetric matrix is not CG's problem to solve ───────────────────
def krylov_probe() -> None:
    cg_solver = "gmres" if MUTATE else "cg"
    if MUTATE:
        print("mutation=the_cg_slot_is_solved_with_gmres")
    cg_info, _ = build(8, 1, 1.0, cg_solver)
    gm_info, _ = build(8, 1, 1.0, "gmres")
    cg_ok = bool(cg_info.get("converged"))
    gm_ok = bool(gm_info.get("converged"))
    print(f"   # cg converged={cg_ok} iterations={cg_info.get('linear_iterations')}")
    print(f"   # gmres converged={gm_ok} iterations={gm_info.get('linear_iterations')}")
    note(not cg_ok, "cg_does_not_solve_the_nonsymmetric_system")
    note(gm_ok, "gmres_does")


# ── #3  an unseen form compiles; the same form does not ──────────────────────
_CHILD = '''
import sys
sys.argv = ["child"]
from dune.alugrid import aluConformGrid
from dune.fem.scheme import galerkin
from dune.fem.space import dglagrange
from dune.grid import cartesianDomain
from dune.ufl import DuneCellDiameter as CellDiameter
from ufl import TestFunction, TrialFunction, dx, grad, inner
grid = aluConformGrid(cartesianDomain([0, 0], [1, 1], [4, 4]), dimgrid=2)
space = dglagrange(grid, order=1)
u, v = TrialFunction(space), TestFunction(space)
h = CellDiameter(space)
# A LITERAL BAKED INTO THE FORM, not a Constant: that is what makes the form
# unseen. A dune.ufl.Constant of a different value reuses the same module.
scheme = galerkin([inner(grad(u), grad(v)) * dx + (__MARK__ / h) * u * v * dx
                   == v * dx], solver="gmres")
print("CHILD_BUILT")
'''


def compile_probe() -> None:
    import random

    mark = f"{random.random():.12f}"
    body = _CHILD.replace("__MARK__", mark)
    with tempfile.TemporaryDirectory() as tmp:
        child = Path(tmp) / "child.py"
        child.write_text(body)
        first = subprocess.run([sys.executable, str(child)], cwd=tmp,
                               capture_output=True, text=True, timeout=1500)
        again = subprocess.run([sys.executable, str(child)], cwd=tmp,
                               capture_output=True, text=True, timeout=1500)
    blob1 = first.stdout + first.stderr
    blob2 = again.stdout + again.stderr

    def compiles(blob: str) -> int:
        return sum(1 for ln in blob.splitlines()
                   if "Compiling" in ln and "(new)" in ln)

    n1, n2 = compiles(blob1), compiles(blob2)
    print(f"   # first run compiles={n1}, same form again compiles={n2}")
    note("CHILD_BUILT" in blob1, "the_probe_ran")
    note(n1 >= 1, "an_unseen_form_compiles_before_its_first_solve")
    note(n2 == 0, "the_same_form_reused_does_not_compile_again")
    # The literal wording, because guidance keyed to a string nobody will see
    # is unreachable exactly when it is needed.
    note(any("Compiling" in ln and "(new)" in ln for ln in blob1.splitlines()),
         "and_the_line_reads_Compiling_X_new")


def main() -> int:
    cell_diameter_probe()
    krylov_probe()
    compile_probe()
    ok = not FAILURES
    note(ok, "dune_sipg_traps_verified", "" if ok else f"missing {FAILURES}")
    if not ok:
        print("FAIL: " + ", ".join(FAILURES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
