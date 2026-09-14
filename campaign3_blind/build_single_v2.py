#!/usr/bin/env python
"""Single-code blind instances: two per backend, GENERATED, never hand-written.

Why this file exists
--------------------
The seven pre-existing single-code tasks (B1-B7) were written by hand, one at a
time, and the brief drifted away from the grader.  The measured consequence:
tasks asking for a probe count the grader does not accept, so every submission
would have been rejected as INVALID_SUBMISSION however good the physics was.
A hand-written task can drift.  A generated one cannot, because it IMPORTS the
grader's constants instead of restating them:

    from grade_blind import PROBE_M, MAX_COARSE_REL, probe_grid

and then asserts, per cell, that the probe rule printed in the task text
rebuilds exactly the point set ``grade_blind.probe_grid`` will demand.

What a cell must satisfy before it is allowed out of here
---------------------------------------------------------
1. the manufactured solution verifies SYMBOLICALLY (residual to exact zero
   against the stated f, coefficients and boundary conditions) and again
   NUMERICALLY by finite differences that share no machinery with the
   derivation;
2. the leak gate is clean -- in particular no eigenfunction draw, where
   ``f = c*u`` and one division off the printed brief is the answer;
3. the element, the integrator and (for a transient cell) the dt sequence are
   PINNED IN THE TASK TEXT.  The theoretical order depends on all three.  An
   agent that picks MINI instead of Taylor-Hood earns order 2 against a
   theoretical 3 and is graded CONFIDENTLY_WRONG for a perfectly good answer;
4. the theoretical order is the MINIMUM over the graded fields, and a
   multi-field cell states which single field is graded.  The grader pools all
   reported components into one unweighted norm, so a thermo-mechanical cell
   reporting temperature next to displacement grades the temperature and
   nothing else;
5. the required output precision is stated, and the FINEST level's error clears
   that precision floor by ~100x.  A saturated error sequence is reported by the
   grader as a malformed submission, not as saturation;
6. ``exact_rms`` is sealed into the key.  Without it the grader's magnitude
   check silently degrades to rate-only, and a rate can be manufactured from the
   published source term with no solver involved.

Custody
-------
This file holds no solution.  The hidden coefficients are drawn from a CSPRNG at
build time and only the draw seed is written, into the sealed key.  Reading this
file tells you the FAMILY, never the instance.

Run:  .venv/bin/python campaign3_blind/build_single_v2.py [--only FE1] [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-blind-eval/src")

import grade_blind as G                                     # noqa: E402
from blind_eval.leakgate import scan                        # noqa: E402

x, y, z, t = sp.symbols("x y z t", real=True)

# THE GRADER'S CONSTANTS, IMPORTED. Never restate them here -- that restatement
# is exactly the drift this generator exists to remove.
PROBE_M = G.PROBE_M
MAX_COARSE_REL = G.MAX_COARSE_REL

# Output precision demanded of the agent, and the floor it implies. A value
# written with S significant digits carries a relative rounding error of about
# 10**-S; over the probe grid that is a noise floor of exact_rms * 10**-S in the
# RMS. The finest level's true error must clear it by this factor or the error
# sequence saturates and the grader -- correctly -- calls the submission
# malformed rather than converged.
SIG_DIGITS = 12
PRECISION_HEADROOM = 100.0


class Draw:
    """Hidden-field coefficients from a CSPRNG. The builder never holds one."""

    def __init__(self, seed: int | None = None):
        self.seed = seed if seed is not None else secrets.randbits(64)
        self._st = self.seed

    def _next(self) -> int:
        self._st = (self._st * 6364136223846793005 + 1442695040888963407) % (2 ** 64)
        return (self._st >> 33) & 0xFFFFFFFF

    def rat(self, lo=1, hi=9, allow_negative=True) -> sp.Rational:
        num = lo + self._next() % (hi - lo + 1)
        den = 1 + self._next() % 5
        s = -1 if (allow_negative and self._next() % 2) else 1
        return sp.Rational(s * num, den)

    def nonzero(self, lo=1, hi=9) -> sp.Rational:
        v = self.rat(lo, hi)
        return v if v != 0 else sp.Rational(lo, 1)


# ── the probe rule, generated from the grader's own constant ──────────────
def probe_rule(dim: int) -> str:
    M = PROBE_M[dim]
    names = ["x", "y", "z"][:dim]
    parts = "; ".join(f"{n} = (i_{n}+0.5)/{M}" for n in names)
    return (f"the {M ** dim} points given by {parts}, for "
            f"i_{', i_'.join(names)} = 0, 1, ..., {M - 1} independently, "
            f"ordered with the last index varying fastest")


def probe_points(dim: int):
    """The points the task text describes -- and, asserted below, the points
    the grader will demand."""
    M = PROBE_M[dim]
    axes = [[(i + 0.5) / M for i in range(M)] for _ in range(dim)]
    pts = [()]
    for ax in axes:
        pts = [p + (v,) for p in pts for v in ax]
    return pts


def assert_no_drift(dim: int) -> None:
    """The one check that makes a generated task safe: the rule printed in the
    brief rebuilds the grader's evaluation set exactly."""
    mine, theirs = probe_points(dim), G.probe_grid(dim)
    if len(mine) != len(theirs):
        raise AssertionError(f"probe count drift: task says {len(mine)}, "
                             f"grader demands {len(theirs)}")
    for a, b in zip(mine, theirs):
        for u, v in zip(a, b):
            if abs(u - v) > G.PROBE_TOL:
                raise AssertionError(f"probe point drift: {a} vs {b}")


# ── the task text ─────────────────────────────────────────────────────────
OUTPUT_CONTRACT = """
REQUIRED OUTPUT (this is how your work is assessed; follow it exactly):
Your solution is assessed at a FIXED set of probe points that does not depend
on your mesh. For every mesh level, evaluate your computed solution at every
probe point and write one CSV file `solution_level<k>.csv` (k = 1, 2, ..., {nlev})
with a header line and one row per probe point, in the order defined below:
    {header}

PROBE POINTS: {probe}
Evaluate (interpolate) your computed solution at these points. They are
deliberately not mesh nodes. Every probe point must appear exactly once and no
other point may appear.

PRECISION: write every value with AT LEAST {sig} significant digits (for
example the format %.{sigm1}e). Values rounded to fewer digits do not carry the
information the assessment needs and the submission is rejected.
{graded_note}
RUN LOG (required, one per level): for every mesh level k, write a file
    run_level<k>.log
containing BOTH of the following:
  (a) THE CONSOLE OUTPUT YOUR SOLVER ITSELF PRODUCED for that level, captured
      verbatim. Redirect the solver's own output into this file, for example
          <your run command>  > run_level<k>.log 2>&1
      or, from Python, raise the solver's log/verbosity level so its own
      messages land there. Do not retype, summarise, translate or paraphrase
      it, and do not write it yourself: it must be the text the named code
      emitted.
  (b) the line
          NDOF = <integer>
      where <integer> is {ndof_desc} at that level.
The assessment reads (a) to establish WHICH code ran and (b) as the number it
cross-checks against the mesh. A level without (b) counts as not run at all. A
log that carries no output the NAMED code itself produced cannot be credited to
that code, however plausible its numbers are.

VERBOSITY IS NOT FREE. Four of the codes print nothing useful at their default
settings, so capturing (a) requires one extra line. These were measured on this
machine; use the one for your code:
  * FEniCSx / dolfinx  — silent by default. Before creating the mesh:
        import dolfinx; dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
    (the DOLFINX_LOGLEVEL environment variable is NOT honoured)
  * deal.II            — needs BOTH:
        deallog.depth_console(2);
        SolverControl ctl(max_it, tol, /*log_history=*/true, /*log_result=*/true);
    depth_console alone prints nothing.
  * NGSolve            — ngsolve.ngsglobals.msg_level = 3
    (or use ngsolve.solvers.CG(..., printrates=True), which works at the default)
  * DUNE-fem           — pass parameters={"linear.verbose": True} to galerkin(...)
  * scikit-fem         — logging.basicConfig(level=logging.INFO), or print(basis).
    Its log goes to STDERR, so you must redirect with 2>&1 or it is lost.
  * Kratos, 4C, FEBio, SPARTA — nothing to set; they print at default settings.

Then write the single summary file `RESULT.txt` containing exactly these lines:
    LEVELS = <number of mesh levels you solved>
    FILES = <comma-separated list of the CSV files you wrote>
    MESH_INDEPENDENCE = <CONVERGED or NOT_CONVERGED>
    MAX_REL_CHANGE = <largest relative change between the two finest levels>
Judge MESH_INDEPENDENCE solely from how your own solution behaves under
refinement, globally and at individual points: no reference answer is available
to you.
"""

INTEGRITY = """
INTEGRITY (absolute): every number you write must come from ACTUALLY RUNNING
the named solver. Do not compute values by hand, by formula, or from memory,
and do not stub or mock the solver. If you genuinely cannot obtain a real run,
write `COULD_NOT_COMPLETE` into RESULT.txt and explain why. Inventing numbers
is a failure, worse than an honest incomplete.
PERSISTENCE: solvers will error first; read the error, fix it, re-run, iterate.
"""


def build_task(spec: dict, f_text: str) -> str:
    dim = spec["dim"]
    comps = spec.get("components", ["u"])
    header = ", ".join(spec["coords"]) + ", " + ", ".join(comps)
    lines = [f"Using {spec['code_label']}, solve the following problem.", "",
             f"DOMAIN: {spec['domain']}",
             f"EQUATION: {spec['equation']}",
             f"COEFFICIENTS: {spec['coefficients']}",
             f"SOURCE TERM: {f_text}"]
    if spec.get("initial_condition"):
        lines.append(f"INITIAL CONDITION: {spec['initial_condition']}")
    lines.append(f"BOUNDARY CONDITIONS: {spec['bc_text']}")
    if spec.get("side_condition"):
        lines.append(f"SIDE CONDITION: {spec['side_condition']}")
    # The element, the integrator and the dt sequence decide the theoretical
    # order. Leaving any of them to the agent means grading it against an order
    # its (perfectly legitimate) choice does not have.
    lines.append(f"DISCRETISATION (use exactly this, it is part of the "
                 f"problem): {spec['element']}")
    if spec.get("time_grid"):
        lines.append(f"TIME DISCRETISATION: {spec['time_grid']}")
    if spec.get("solver_pin"):
        lines.append(f"SOLVER: {spec['solver_pin']}")
    lines.append(f"MESH SEQUENCE: {spec['mesh_text']}")
    if spec.get("notes_public"):
        lines.append(f"NOTE: {spec['notes_public']}")
    lines += ["",
              OUTPUT_CONTRACT.format(
                  nlev=len(spec["mesh_N"]), header=header,
                  probe=probe_rule(dim), sig=SIG_DIGITS, sigm1=SIG_DIGITS - 1,
                  graded_note=spec.get("graded_note", ""),
                  ndof_desc=spec.get(
                      "ndof_desc",
                      "the total number of degrees of freedom of YOUR "
                      "discretisation")),
              INTEGRITY]
    return "\n".join(lines)


# ── operators ─────────────────────────────────────────────────────────────
def lap(u, coords):
    return sum(sp.diff(u, c, 2) for c in coords)


def div_k_grad(u, coords, K):
    """-div(K grad u) for a constant or variable matrix K."""
    n = len(coords)
    flux = [sum(K[i, j] * sp.diff(u, coords[j]) for j in range(n))
            for i in range(n)]
    return -sum(sp.diff(flux[i], coords[i]) for i in range(n))


def elastic_source(u, coords, lam, mu):
    """-div(sigma(u)) for isotropic linear elasticity, small strain."""
    n = len(coords)
    eps = sp.Matrix(n, n, lambda i, j: (sp.diff(u[i], coords[j])
                                        + sp.diff(u[j], coords[i])) / 2)
    tr = sum(eps[i, i] for i in range(n))
    sig = sp.Matrix(n, n, lambda i, j: 2 * mu * eps[i, j]
                    + (lam * tr if i == j else 0))
    return [sp.expand(-sum(sp.diff(sig[i, j], coords[j]) for j in range(n)))
            for i in range(n)]


def bubble(coords, power=1):
    b = sp.Integer(1)
    for c in coords:
        b *= (c * (1 - c)) ** power
    return b


# ── the constructions ─────────────────────────────────────────────────────
def scalar_diffusion(d: Draw, K, dim, extra=None):
    """-div(K grad u) = f, u = 0 on the boundary of the unit box."""
    coords = (x, y, z)[:dim]
    c = [d.nonzero(1, 6), d.rat(1, 4), d.rat(1, 4), d.rat(1, 3)]
    poly = c[0] + c[1] * x + c[2] * y + (c[3] * z if dim == 3 else c[3] * x * y)
    u = sp.expand(bubble(coords) * poly * (extra if extra is not None else 1))
    f = sp.expand(div_k_grad(u, coords, K))
    return u, f, coords


def nonlinear_diffusion(d: Draw):
    """-div(a(u) grad u) = f with a(u) = 1 + u^2/2. Newton is unavoidable."""
    coords = (x, y)
    c = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
    u = sp.expand(bubble(coords) * (c[0] + c[1] * x + c[2] * y))
    a = 1 + u ** 2 / 2
    flux = [a * sp.diff(u, cc) for cc in coords]
    f = sp.expand(-sum(sp.diff(flux[i], coords[i]) for i in range(2)))
    return u, f, coords


def elasticity(d: Draw, lam, mu, dim, amp=1):
    coords = (x, y, z)[:dim]
    b = bubble(coords)
    comps = []
    for _ in range(dim):
        c = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
        comps.append(sp.expand(amp * b * (c[0] + c[1] * coords[0]
                                          + c[2] * coords[1])))
    f = elastic_source(comps, coords, lam, mu)
    return comps, f, coords


def stokes(d: Draw, mu):
    """-mu*lap(u) + grad p = f, div u = 0, u = 0 on the whole boundary.

    ``u = curl(psi)`` with ``psi`` carrying DOUBLE roots on every side makes the
    velocity divergence-free by construction and zero on the boundary, both
    components. Only the velocity is graded, so the pressure's additive constant
    -- the one genuinely undetermined quantity in a pure Dirichlet Stokes
    problem -- cannot reach the grade.
    """
    coords = (x, y)
    c = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    psi = bubble(coords, 2) * (c[0] + c[1] * x + c[2] * y)
    u = [sp.expand(sp.diff(psi, y)), sp.expand(-sp.diff(psi, x))]
    q = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    praw = q[0] * x ** 2 * y + q[1] * x * y + q[2] * sp.cos(sp.pi * x) * y
    p = sp.expand(praw - sp.integrate(sp.integrate(praw, (x, 0, 1)), (y, 0, 1)))
    f = [sp.expand(-mu * lap(u[0], coords) + sp.diff(p, x)),
         sp.expand(-mu * lap(u[1], coords) + sp.diff(p, y))]
    return u, p, f, coords


def near_incompressible(d: Draw, mu, lam):
    """Linear elasticity at nu = 0.49999, built so the PRESSURE is O(1).

    The trap only exists if the solution is the one a near-incompressible body
    actually has.  Manufacture ``u`` freely and ``div u`` is O(1), so
    ``p = -lambda div u`` is O(lambda): the cell would then be a compressible
    problem wearing a large lambda, and a locking element would do fine on it.

    So the pressure is chosen first.  ``u = curl(psi) + grad(phi)/lambda`` with
    both potentials carrying DOUBLE roots on every side: ``curl psi`` is exactly
    divergence-free, ``grad phi`` vanishes on the boundary, and
    ``div u = lap(phi)/lambda``, i.e. ``p = -lap(phi)`` is O(1) while ``div u``
    is O(1/lambda).  That is the incompressible limit, and a pure-displacement
    P1/P2 element locks on it.
    """
    coords = (x, y)
    c = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    e = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    psi = bubble(coords, 2) * (c[0] + c[1] * x + c[2] * y)
    phi = bubble(coords, 2) * (e[0] + e[1] * x + e[2] * y)
    u = [sp.expand(sp.diff(psi, y) + sp.diff(phi, x) / lam),
         sp.expand(-sp.diff(psi, x) + sp.diff(phi, y) / lam)]
    p = sp.expand(-lap(phi, coords))
    f = elastic_source(u, coords, lam, mu)
    return u, p, f, coords


def navier_stokes(d: Draw, nu):
    """Steady incompressible Navier-Stokes. Same divergence-free construction,
    plus the convective term; the Reynolds number is O(1/10) so the steady
    solution is unique and Newton converges from rest."""
    coords = (x, y)
    c = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    psi = bubble(coords, 2) * (c[0] + c[1] * x + c[2] * y)
    u = [sp.expand(sp.diff(psi, y)), sp.expand(-sp.diff(psi, x))]
    q = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    praw = q[0] * x ** 2 * y + q[1] * x * y + q[2] * sp.cos(sp.pi * x) * y
    p = sp.expand(praw - sp.integrate(sp.integrate(praw, (x, 0, 1)), (y, 0, 1)))
    conv = [sum(u[j] * sp.diff(u[i], coords[j]) for j in range(2))
            for i in range(2)]
    f = [sp.expand(-nu * lap(u[i], coords) + conv[i] + sp.diff(p, coords[i]))
         for i in range(2)]
    return u, p, f, coords


def stvk_plane_strain(d: Draw, lam, mu, amp):
    """FINITE-strain St.Venant-Kirchhoff in plane strain, manufactured exactly.

    FEBio has no small-strain solid material: its "isotropic elastic" is
    St.Venant-Kirchhoff pushed forward (read off the source --
    ``s = b*(lam*trE - mu) + b2*mu`` over ``J``, which is exactly
    ``J^-1 F (lam tr(E) I + 2 mu E) F^T``). Manufacturing against the LINEAR
    operator and shrinking the amplitude until the difference hides is a model
    error that has to be argued about; manufacturing against the operator the
    code actually integrates is exact and needs no argument.

    ``u3 = 0`` and no Z dependence gives ``P13 = P23 = P31 = P32 = 0`` and a
    ``P33`` independent of Z, so the third equilibrium equation is satisfied
    identically and the body force is purely in-plane. That is what makes a
    one-element-thick slab an exact plane-strain model rather than an
    approximation.
    """
    coords = (x, y)
    b = bubble(coords)
    c = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
    e = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
    u = [sp.expand(amp * b * (c[0] + c[1] * x + c[2] * y)),
         sp.expand(amp * b * (e[0] + e[1] * x + e[2] * y))]
    G = sp.zeros(3, 3)
    for i in range(2):
        for j in range(2):
            G[i, j] = sp.diff(u[i], coords[j])
    F = sp.eye(3) + G
    Eg = (F.T * F - sp.eye(3)) / 2
    S = lam * sum(Eg[i, i] for i in range(3)) * sp.eye(3) + 2 * mu * Eg
    P = sp.expand(F * S)
    f = [sp.expand(-(sp.diff(P[i, 0], x) + sp.diff(P[i, 1], y)))
         for i in range(3)]
    if sp.simplify(f[2]) != 0:
        raise AssertionError("plane strain broken: the out-of-plane body "
                             "force does not vanish")
    return u, P, f[:2], coords


def helmholtz(d: Draw, k):
    """-lap(u) - k^2 u = f. The exponential factor keeps the k^2 u part from
    lining up term by term with anything printed, so the brief cannot be
    inverted by collecting like monomials."""
    coords = (x, y)
    c = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    a1, a2 = d.rat(1, 2), d.rat(1, 2)
    u = sp.expand(bubble(coords) * (c[0] + c[1] * x + c[2] * y)
                  * sp.exp(a1 * x + a2 * y))
    f = sp.expand(-lap(u, coords) - k ** 2 * u)
    return u, f, coords


def biharmonic(d: Draw):
    """lap(lap(u)) = f, clamped: u = du/dn = 0 on the whole boundary."""
    coords = (x, y)
    c = [d.nonzero(1, 4), d.rat(1, 3), d.rat(1, 3)]
    u = sp.expand(bubble(coords, 2) * (c[0] + c[1] * x + c[2] * y))
    f = sp.expand(lap(lap(u, coords), coords))
    return u, f, coords


def advection_diffusion(d: Draw, eps, b):
    """-eps*lap(u) + b.grad(u) = f, diffusion-dominated on purpose.

    Diffusion-dominated matters for the GRADE, not for the physics: for upwinded
    DG on an advection-dominated problem the sharp bound is h^(k+1/2), not
    h^(k+1), and a cell graded against k+1 would fail a correct run.
    """
    coords = (x, y)
    c = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
    u = sp.expand(bubble(coords) * (c[0] + c[1] * x + c[2] * y))
    f = sp.expand(-eps * lap(u, coords)
                  + sum(b[i] * sp.diff(u, coords[i]) for i in range(2)))
    return u, f, coords


def disk_diffusion(d: Draw, R, cx, cy):
    """-lap(u) = f on a DISK. u carries the factor (R^2 - r^2), so the
    homogeneous datum on the curved boundary is exact and discloses nothing."""
    coords = (x, y)
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    c = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
    u = sp.expand((R ** 2 - r2) * (c[0] + c[1] * x + c[2] * y))
    f = sp.expand(-lap(u, coords))
    return u, f, coords


def transient_diffusion(d: Draw, kappa, t_end):
    """du/dt - kappa*lap(u) = f with u(0) = 0, so the initial datum is
    information-free. T(t) = t*exp(t/2) is deliberately not quadratic: a
    quadratic T makes Crank-Nicolson exact in time and the cell silently stops
    testing the integrator."""
    coords = (x, y)
    c = [d.nonzero(1, 5), d.rat(1, 4), d.rat(1, 4)]
    U = bubble(coords) * (c[0] + c[1] * x + c[2] * y)
    T = t * sp.exp(t / 2)
    u = sp.expand(T * U)
    f = sp.expand(sp.diff(u, t) - kappa * lap(u, coords))
    return u, f, coords, sp.expand(u.subs(t, t_end))


# ── verification ──────────────────────────────────────────────────────────
def _fd2_raw(uf, pt, i, j, h):
    def at(di=0.0, dj=0.0):
        q = list(pt)
        q[i] += di
        q[j] += dj
        return uf(*q)
    if i == j:
        q1, q2 = list(pt), list(pt)
        q1[i] += h
        q2[i] -= h
        return (uf(*q1) - 2 * uf(*pt) + uf(*q2)) / h ** 2
    return (at(h, h) - at(h, -h) - at(-h, h) + at(-h, -h)) / (4 * h ** 2)


def _fd2(uf, pt, i, j, h):
    d1 = _fd2_raw(uf, pt, i, j, h)
    d2 = _fd2_raw(uf, pt, i, j, h / 2)
    return (4 * d2 - d1) / 3


def _fd1(uf, pt, i, h):
    p1, p2 = list(pt), list(pt)
    p1[i] += h
    p2[i] -= h
    return (uf(*p1) - uf(*p2)) / (2 * h)


def numeric_residual(op, coords, n=60, seed=7, box=None, h=1e-3, transient=False):
    """Apply the operator by CENTRAL FINITE DIFFERENCES of a lambdified field.

    Not symbolically: sympy canonicalises both sides to the same expression tree
    and the "numeric" check then returns exactly zero for any input, which is
    the defect it exists to catch, reintroduced in the code written to catch it.
    Finite differences share no machinery with sp.diff.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    syms = list(coords) + ([t] if transient else [])
    box = box or ([(0.06, 0.94)] * len(coords) + ([(0.02, 0.25)]
                                                 if transient else []))
    diffs, scales = [], []
    for _ in range(n):
        pt = [float(rng.uniform(lo, hi)) for lo, hi in box]
        lhs, rhs = op(pt, syms, h)
        diffs.append(abs(lhs - rhs))
        scales.append(abs(rhs))
    return max(diffs) / max(max(scales), 1e-12)


def check_scalar(u, f, coords, K, reaction=0, advect=None, transient=False,
                 nonlinear_a=None, box=None):
    """Symbolic residual to exact zero, then the same thing by finite
    differences."""
    n = len(coords)
    if nonlinear_a is not None:
        a = nonlinear_a(u)
        lhs = -sum(sp.diff(a * sp.diff(u, coords[i]), coords[i])
                   for i in range(n))
    else:
        lhs = div_k_grad(u, coords, K)
    if reaction:
        lhs = lhs + reaction * u
    if advect:
        lhs = lhs + sum(advect[i] * sp.diff(u, coords[i]) for i in range(n))
    if transient:
        lhs = lhs + sp.diff(u, t)
    sym = sp.simplify(sp.expand(lhs - f))

    syms = list(coords) + ([t] if transient else [])
    uf = sp.lambdify(syms, u, "math")
    ff = sp.lambdify(syms, f, "math")
    Km = None if nonlinear_a is not None else sp.Matrix(K)

    def op(pt, _s, h):
        if nonlinear_a is not None:
            # -div(a(u) grad u) by nested differences of a(u)*du/dx_i
            val = 0.0
            for i in range(n):
                def flux(*q, _i=i):
                    ui = uf(*q)
                    return float(nonlinear_a(ui)) * _fd1(uf, list(q), _i, h)
                val -= _fd1(flux, pt, i, h)
            lhs_v = val
        else:
            lhs_v = 0.0
            for i in range(n):
                for j in range(n):
                    kij = Km[i, j]
                    if kij == 0:
                        continue
                    if kij.free_symbols:
                        kf = sp.lambdify(coords, kij, "math")
                        # variable coefficient: differentiate the flux itself
                        def fl(*q, _i=i, _j=j, _kf=kf):
                            return _kf(*q[:n]) * _fd1(uf, list(q), _j, h)
                        lhs_v -= _fd1(fl, pt, i, h)
                    else:
                        lhs_v -= float(kij) * _fd2(uf, pt, i, j, h)
        if reaction:
            lhs_v += float(reaction) * uf(*pt)
        if advect:
            lhs_v += sum(float(advect[i]) * _fd1(uf, pt, i, h) for i in range(n))
        if transient:
            lhs_v += _fd1(uf, pt, len(syms) - 1, h)
        return lhs_v, ff(*pt)

    num = numeric_residual(op, coords, transient=transient, box=box)
    return sym, num


def check_vector(u, f, coords, lam, mu):
    n = len(coords)
    sym_ok = True
    resid = elastic_source(u, coords, lam, mu)
    for i in range(n):
        if sp.simplify(resid[i] - f[i]) != 0:
            sym_ok = False
    ufs = [sp.lambdify(coords, c, "math") for c in u]
    ffs = [sp.lambdify(coords, c, "math") for c in f]
    import numpy as np
    rng = np.random.default_rng(23)
    worst, scale = 0.0, 1e-12
    h = 1e-3
    for _ in range(40):
        pt = [float(rng.uniform(0.08, 0.92)) for _ in range(n)]
        for i in range(n):
            # -div(sigma)_i = -mu*lap(u_i) - (lam+mu)*d/dx_i (div u)
            v = -float(mu) * sum(_fd2(ufs[i], pt, k, k, h) for k in range(n))
            v -= float(lam + mu) * sum(_fd2(ufs[j], pt, i, j, h)
                                       for j in range(n))
            worst = max(worst, abs(v - ffs[i](*pt)))
            scale = max(scale, abs(ffs[i](*pt)))
    return sym_ok, worst / scale


def check_stokes(u, p, f, coords, mu):
    sym = [sp.simplify(-mu * lap(u[i], coords) + sp.diff(p, coords[i]) - f[i])
           for i in range(2)]
    divu = sp.simplify(sum(sp.diff(u[i], coords[i]) for i in range(2)))
    ufs = [sp.lambdify(coords, c, "math") for c in u]
    pf = sp.lambdify(coords, p, "math")
    ffs = [sp.lambdify(coords, c, "math") for c in f]
    import numpy as np
    rng = np.random.default_rng(31)
    worst, scale, wdiv = 0.0, 1e-12, 0.0
    h = 1e-3
    for _ in range(40):
        pt = [float(rng.uniform(0.08, 0.92)) for _ in range(2)]
        for i in range(2):
            v = -float(mu) * sum(_fd2(ufs[i], pt, k, k, h) for k in range(2))
            v += _fd1(pf, pt, i, h)
            worst = max(worst, abs(v - ffs[i](*pt)))
            scale = max(scale, abs(ffs[i](*pt)))
        wdiv = max(wdiv, abs(sum(_fd1(ufs[i], pt, i, h) for i in range(2))))
    return (all(s == 0 for s in sym) and divu == 0), worst / scale, wdiv


def check_mixed_elasticity(u, p, f, coords, mu, lam):
    """The two-field residual, NOT the one-field one.

    Checking ``-div(2 mu eps + lam div(u) I) - f`` numerically at
    lambda = 1.7e7 subtracts two quantities of size 1e7 to get one of size 300:
    the finite-difference noise in ``div u`` is amplified by lambda and the
    honest construction fails its own check. The two-field residual carries no
    such cancellation because ``p`` is O(1) by construction.
    """
    n = 2
    eps = sp.Matrix(n, n, lambda i, j: (sp.diff(u[i], coords[j])
                                        + sp.diff(u[j], coords[i])) / 2)
    sym = []
    for i in range(n):
        r = -sum(sp.diff(2 * mu * eps[i, j], coords[j]) for j in range(n))
        sym.append(sp.simplify(sp.expand(r + sp.diff(p, coords[i]) - f[i])))
    divu = sp.simplify(sp.expand(sum(sp.diff(u[i], coords[i]) for i in range(n))
                                 + p / lam))
    ufs = [sp.lambdify(coords, c, "math") for c in u]
    pf = sp.lambdify(coords, p, "math")
    ffs = [sp.lambdify(coords, c, "math") for c in f]
    import numpy as np
    rng = np.random.default_rng(53)
    worst, scale = 0.0, 1e-12
    h = 1e-3
    for _ in range(40):
        pt = [float(rng.uniform(0.08, 0.92)) for _ in range(n)]
        for i in range(n):
            v = -float(mu) * sum(_fd2(ufs[i], pt, k, k, h) for k in range(n))
            v -= float(mu) * sum(_fd2(ufs[j], pt, i, j, h) for j in range(n))
            v += _fd1(pf, pt, i, h)
            worst = max(worst, abs(v - ffs[i](*pt)))
            scale = max(scale, abs(ffs[i](*pt)))
    return (all(s == 0 for s in sym) and divu == 0), worst / scale


def check_navier_stokes(u, p, f, coords, nu):
    sym = []
    for i in range(2):
        conv = sum(u[j] * sp.diff(u[i], coords[j]) for j in range(2))
        sym.append(sp.simplify(sp.expand(-nu * lap(u[i], coords) + conv
                                         + sp.diff(p, coords[i]) - f[i])))
    divu = sp.simplify(sum(sp.diff(u[i], coords[i]) for i in range(2)))
    ufs = [sp.lambdify(coords, c, "math") for c in u]
    pf = sp.lambdify(coords, p, "math")
    ffs = [sp.lambdify(coords, c, "math") for c in f]
    import numpy as np
    rng = np.random.default_rng(61)
    worst, scale, wdiv = 0.0, 1e-12, 0.0
    h = 1e-3
    for _ in range(40):
        pt = [float(rng.uniform(0.08, 0.92)) for _ in range(2)]
        for i in range(2):
            v = -float(nu) * sum(_fd2(ufs[i], pt, k, k, h) for k in range(2))
            v += sum(ufs[j](*pt) * _fd1(ufs[i], pt, j, h) for j in range(2))
            v += _fd1(pf, pt, i, h)
            worst = max(worst, abs(v - ffs[i](*pt)))
            scale = max(scale, abs(ffs[i](*pt)))
        wdiv = max(wdiv, abs(sum(_fd1(ufs[i], pt, i, h) for i in range(2))))
    return (all(s == 0 for s in sym) and divu == 0), worst / scale, wdiv


def check_stvk(u, f, coords, lam, mu):
    """Reference-configuration equilibrium, by finite differences of u.

    The whole stress chain -- F, E, S, P -- is rebuilt here from a lambdified
    displacement, so nothing is shared with the sympy derivation above.
    """
    import numpy as np
    ufs = [sp.lambdify(coords, c, "math") for c in u]
    ffs = [sp.lambdify(coords, c, "math") for c in f]
    lam_f, mu_f = float(lam), float(mu)
    h = 1e-4

    def Pmat(pt):
        G = np.zeros((3, 3))
        for i in range(2):
            for j in range(2):
                G[i, j] = _fd1(ufs[i], pt, j, h)
        F = np.eye(3) + G
        Eg = (F.T @ F - np.eye(3)) / 2
        S = lam_f * np.trace(Eg) * np.eye(3) + 2 * mu_f * Eg
        return F @ S

    rng = np.random.default_rng(71)
    worst, scale = 0.0, 1e-12
    hh = 1e-3
    for _ in range(30):
        pt = [float(rng.uniform(0.1, 0.9)) for _ in range(2)]
        div = np.zeros(3)
        for j in range(2):
            p1, p2 = list(pt), list(pt)
            p1[j] += hh
            p2[j] -= hh
            div += (Pmat(p1)[:, j] - Pmat(p2)[:, j]) / (2 * hh)
        for i in range(2):
            worst = max(worst, abs(-div[i] - ffs[i](*pt)))
            scale = max(scale, abs(ffs[i](*pt)))
        worst = max(worst, abs(div[2]) * 0)      # out-of-plane checked above
    return True, worst / scale


def check_biharmonic(u, f, coords):
    sym = sp.simplify(lap(lap(u, coords), coords) - f)
    uf = sp.lambdify(coords, u, "math")
    ff = sp.lambdify(coords, f, "math")
    import numpy as np
    rng = np.random.default_rng(41)
    worst, scale = 0.0, 1e-12
    h = 2e-2                              # fourth derivative: a wider stencil
    for _ in range(30):
        pt = [float(rng.uniform(0.2, 0.8)) for _ in range(2)]

        def lap_fn(*q):
            return sum(_fd2(uf, list(q), k, k, h) for k in range(2))

        v = sum(_fd2(lap_fn, pt, k, k, h) for k in range(2))
        worst = max(worst, abs(v - ff(*pt)))
        scale = max(scale, abs(ff(*pt)))
    return sym == 0, worst / scale


# ── probe-grid quantities the key needs ───────────────────────────────────
def exact_rms(exprs, coords, dim):
    """RMS of the graded field on the grader's probe grid.

    Pooled exactly as ``grade_blind.rms_error`` pools: the sum over components
    at one point, averaged over POINTS, not over point-component pairs. A
    different convention here would make the magnitude bound the grader applies
    mean something other than it says.
    """
    syms = [{"x": x, "y": y, "z": z}[c] for c in coords]
    fns = [sp.lambdify(syms, e, "math") for e in exprs]
    acc, n = 0.0, 0
    for pt in G.probe_grid(dim):
        acc += sum(float(fn(*pt)) ** 2 for fn in fns)
        n += 1
    return (acc / n) ** 0.5


def all_probes_inside(dim, predicate) -> bool:
    return all(predicate(*pt) for pt in G.probe_grid(dim))


# ── the cells ─────────────────────────────────────────────────────────────
BAND_P1 = [0.8, 3.2]


def _base(pid, code, label, physics, regime, dim, mesh_N, theo, tol, band,
          **kw):
    s = dict(id=pid, kind="single", code=code, code_label=label,
             physics=physics, regime=regime, dim=dim,
             coords=["x", "y", "z"][:dim], mesh_N=mesh_N,
             theoretical_order=theo, tol=tol, band=band,
             sig_digits=SIG_DIGITS,
             qoi="convergence order of the error on a fixed probe grid",
             graded_against="hidden manufactured solution (true error)")
    s.update(kw)
    return s


def cell_FE1(d):
    """FEniCSx, nonlinear a(u) diffusion -- kept from the existing set."""
    u, f, co = nonlinear_diffusion(d)
    s = _base("FE1", "fenics", "FEniCSx (dolfinx)", "nonlinear diffusion",
              "nonlinear", 2, [8, 16, 32, 64], 2.0, 0.4, BAND_P1,
              domain="the unit square (0,1) x (0,1)",
              equation="-div(a(u) grad u) = f, with the coefficient depending "
                       "on the solution itself",
              coefficients="a(u) = 1 + u**2/2. The problem is NONLINEAR; use a "
                           "Newton solver and converge it to a relative "
                           "residual of 1e-10 or smaller at every level.",
              bc_text="u = 0 on the entire boundary",
              element="Lagrange P1 (first-order, continuous) elements on "
                      "triangles",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32, "
                        "64 elements per side (h = 1/N), all four levels")
    return s, u, f, co, dict(kind="scalar", K=sp.eye(2),
                             nonlinear_a=lambda v: 1 + v ** 2 / 2)


def cell_FE2(d):
    """FEniCSx, NEAR-INCOMPRESSIBLE elasticity at nu = 0.49999.

    The locking trap.  A pure-displacement P1 or P2 element on this material
    carries a volumetric error that does not go away under refinement, and the
    agent's own reference-free mesh-independence check does not see it: the
    solution is stably, smoothly, confidently wrong.  The element is pinned to a
    mixed Taylor-Hood pair because otherwise there is no theoretical order to
    grade against -- a locked run has no order at all.
    """
    # mu and lambda are chosen as the PRIMARY data, both integers, so that
    # nu = lambda/(2(lambda+mu)) is exactly 0.49999 and the printed source term
    # carries ordinary coefficients. Deriving them from a round E instead gives
    # lambda = 2499950000000/149999, which is exact, correct and unreadable.
    mu, lam = sp.Integer(1500), sp.Integer(74998500)
    nu = sp.Rational(lam, 2 * (lam + mu))
    E = sp.Rational(mu * (3 * lam + 2 * mu), lam + mu)
    u, p, f, co = near_incompressible(d, mu, lam)
    s = _base("FE2", "fenics", "FEniCSx (dolfinx)",
              "near-incompressible linear elasticity (nu = 0.49999)",
              "near-incompressible/locking", 2, [8, 16, 32], 3.0, 0.5,
              [1.8, 4.4], components=["ux", "uy"],
              domain="the unit square (0,1) x (0,1)",
              equation="-div(sigma(u)) = f, with "
                       "sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I and "
                       "eps(u) = (grad(u) + grad(u)^T)/2  (plane strain, small "
                       "strain, u = (ux, uy) is the displacement)",
              coefficients=f"Lame parameters lambda = {lam} and mu = {mu} "
                           f"(exactly), i.e. Poisson ratio nu = "
                           f"{float(nu):.5f} and Young's modulus "
                           f"E = {float(E):.2f}. The material is NEARLY "
                           f"INCOMPRESSIBLE: lambda/mu = {int(lam / mu)}.",
              bc_text="u = 0 (both components) on the entire boundary",
              element="a MIXED displacement-pressure formulation with the "
                      "Taylor-Hood pair: continuous Lagrange P2 elements for "
                      "BOTH displacement components and continuous Lagrange P1 "
                      "elements for the pressure p, where p = -lambda*div(u), "
                      "on triangles. Use exactly this formulation and this "
                      "pair. A pure-displacement discretisation of this "
                      "material has no convergence order to be graded against, "
                      "and an equal-order or MINI pair has a different one.",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32 "
                        "elements per side (h = 1/N), all three levels",
              graded_note="\nONLY THE DISPLACEMENT IS GRADED. Report ux and uy "
                          "at the probe points; do not report the pressure.\n")
    return s, u, f, co, dict(kind="mixed_elasticity", mu=mu, lam=lam, p=p)


def cell_DL1(d):
    """deal.II, variable-coefficient diffusion -- kept from the existing set."""
    a = 2 + sp.exp(-x) * sp.sin(sp.pi * y)
    K = sp.Matrix([[a, 0], [0, a]])
    u, f, co = scalar_diffusion(d, K, 2)
    s = _base("DL1", "dealii", "deal.II (build and run the C++ program)",
              "variable-coefficient diffusion", "scalar-straightforward", 2,
              [8, 16, 32, 64], 2.0, 0.4, BAND_P1,
              domain="the unit square (0,1) x (0,1)",
              equation="-div(a(x,y) grad u) = f",
              coefficients="a(x,y) = 2 + exp(-x)*sin(pi*y)",
              bc_text="u = 0 on the entire boundary",
              element="Q1 elements (bilinear, continuous) on quadrilaterals, "
                      "with a quadrature rule of at least degree 3",
              mesh_text="globally refined uniform meshes of the unit square "
                        "with N = 8, 16, 32, 64 cells per side, all four levels")
    return s, u, f, co, dict(kind="scalar", K=K)


def cell_DL2(d):
    """deal.II, 3D vector elasticity."""
    E, nu = sp.Integer(2000), sp.Rational(1, 4)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    u, f, co = elasticity(d, lam, mu, 3)
    s = _base("DL2", "dealii", "deal.II (build and run the C++ program)",
              "3D linear elasticity", "3D", 3, [4, 8, 16], 2.0, 0.4, BAND_P1,
              components=["ux", "uy", "uz"],
              domain="the unit cube (0,1) x (0,1) x (0,1)",
              equation="-div(sigma(u)) = f, with "
                       "sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I and "
                       "eps(u) = (grad(u) + grad(u)^T)/2",
              coefficients=f"Young's modulus E = {E}, Poisson ratio nu = {nu}, "
                           f"so lambda = {lam} and mu = {mu}",
              bc_text="u = 0 (all three components) on the entire boundary of "
                      "the cube",
              element="Q1 vector elements (trilinear, continuous) on hexahedra, "
                      "with a quadrature rule of at least degree 3",
              mesh_text="globally refined uniform meshes of the unit cube with "
                        "N = 4, 8, 16 cells per side, all three levels")
    return s, u, f, co, dict(kind="vector", lam=lam, mu=mu)


def cell_NG1(d):
    """NGSolve, anisotropic diffusion -- kept from the existing set."""
    K = sp.Matrix([[3, -1], [-1, 2]])
    u, f, co = scalar_diffusion(d, K, 2)
    s = _base("NG1", "ngsolve", "NGSolve", "anisotropic diffusion",
              "scalar-straightforward", 2, [8, 16, 32, 64], 2.0, 0.4, BAND_P1,
              domain="the unit square (0,1) x (0,1)",
              equation="-div(K grad u) = f",
              coefficients="constant symmetric positive-definite tensor "
                           "K = [[3, -1], [-1, 2]]",
              bc_text="u = 0 on the entire boundary",
              element="Lagrange elements of order 1 (H1, order=1) on triangles",
              mesh_text="uniform meshes of the unit square with maximum element "
                        "size h = 1/8, 1/16, 1/32, 1/64, all four levels")
    return s, u, f, co, dict(kind="scalar", K=K)


def cell_NG2(d):
    """NGSolve, steady incompressible Navier-Stokes. Nonlinear AND saddle-point."""
    nu = sp.Integer(1)
    u, p, f, co = navier_stokes(d, nu)
    s = _base("NG2", "ngsolve", "NGSolve",
              "steady incompressible Navier-Stokes", "steady Navier-Stokes",
              2, [8, 16, 32], 3.0, 0.5, [1.8, 4.4],
              components=["ux", "uy"],
              domain="the unit square (0,1) x (0,1)",
              equation="-nu*lap(u) + (u . grad)u + grad(p) = f  and  "
                       "div(u) = 0  (steady incompressible Navier-Stokes; "
                       "u = (ux, uy) is the velocity and p the pressure). Note "
                       "the CONVECTIVE term: the problem is nonlinear.",
              coefficients="kinematic viscosity nu = 1, density 1. The "
                           "Reynolds number of this solution is well below 1, "
                           "so the steady solution is unique and a Newton "
                           "iteration started from rest converges.",
              bc_text="u = 0 (BOTH components) on the entire boundary",
              side_condition="the velocity boundary condition is Dirichlet "
                             "everywhere, so p is determined only up to an "
                             "additive constant; fix it by requiring the "
                             "integral of p over the domain to be zero. This "
                             "choice does not affect what is graded.",
              element="Taylor-Hood: continuous Lagrange elements of order 2 for "
                      "BOTH velocity components and continuous Lagrange "
                      "elements of order 1 for the pressure, on triangles. Use "
                      "exactly this pair; an equal-order or MINI pair has a "
                      "different convergence order. Solve the nonlinear system "
                      "with Newton's method to a relative residual of 1e-10 or "
                      "smaller at every level.",
              mesh_text="uniform meshes of the unit square with maximum element "
                        "size h = 1/8, 1/16, 1/32, all three levels",
              graded_note="\nONLY THE VELOCITY IS GRADED. Report ux and uy at "
                          "the probe points; do not report the pressure.\n")
    return s, u, f, co, dict(kind="navier_stokes", nu=nu, p=p)


def cell_SK1(d):
    """scikit-fem, Stokes, Taylor-Hood, element pinned."""
    mu = sp.Integer(1)
    u, p, f, co = stokes(d, mu)
    s = _base("SK1", "skfem", "scikit-fem", "Stokes flow, mixed "
              "velocity-pressure", "mixed", 2, [8, 16, 32], 3.0, 0.5,
              [1.8, 4.4], components=["ux", "uy"],
              domain="the unit square (0,1) x (0,1)",
              equation="-mu*lap(u) + grad(p) = f  and  div(u) = 0  "
                       "(steady Stokes flow, u = (ux, uy) is the velocity and "
                       "p the pressure)",
              coefficients="dynamic viscosity mu = 1",
              bc_text="u = 0 (BOTH components) on the entire boundary",
              side_condition="the boundary condition on u is Dirichlet "
                             "everywhere, so p is determined only up to an "
                             "additive constant; fix it by requiring the "
                             "integral of p over the domain to be zero. This "
                             "choice does not affect what is graded.",
              element="Taylor-Hood: ElementVector(ElementTriP2) for the "
                      "velocity and ElementTriP1 for the pressure, on "
                      "triangles. Use exactly this pair; an equal-order or MINI "
                      "element has a different convergence order and will be "
                      "judged against the order of the pair named here.",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32 "
                        "elements per side (h = 1/N), all three levels",
              graded_note="\nONLY THE VELOCITY IS GRADED. Report ux and uy at "
                          "the probe points; do not report the pressure.\n")
    return s, u, f, co, dict(kind="stokes", mu=mu, p=p)


def cell_SK2(d):
    """scikit-fem, biharmonic with the element PINNED.

    "Biharmonic" alone does not define a pass band: Morley gives L2 order 2,
    Argyris 5 or more. The element is therefore part of the problem statement.
    """
    u, f, co = biharmonic(d)
    s = _base("SK2", "skfem", "scikit-fem", "biharmonic (clamped plate)",
              "fourth-order", 2, [8, 16, 32, 64], 2.0, 0.5, BAND_P1,
              domain="the unit square (0,1) x (0,1)",
              equation="lap(lap(u)) = f  (the biharmonic equation)",
              coefficients="none beyond the equation itself (unit bending "
                           "stiffness)",
              bc_text="CLAMPED on the entire boundary: both u = 0 and the "
                      "normal derivative du/dn = 0",
              element="the MORLEY element (ElementTriMorley), the nonconforming "
                      "quadratic triangle. Use exactly this element: the "
                      "convergence order of a biharmonic discretisation runs "
                      "from 2 for Morley to 5 and above for Argyris, so the "
                      "element is part of the problem.",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32, "
                        "64 elements per side (h = 1/N), all four levels")
    return s, u, f, co, dict(kind="biharmonic")


def cell_KR1(d):
    """Kratos, 3D linear elasticity."""
    E, nu = sp.Integer(1000), sp.Rational(3, 10)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    u, f, co = elasticity(d, lam, mu, 3)
    s = _base("KR1", "kratos", "Kratos Multiphysics", "3D linear elasticity",
              "3D", 3, [4, 8, 16], 2.0, 0.4, BAND_P1,
              components=["ux", "uy", "uz"],
              domain="the unit cube (0,1) x (0,1) x (0,1)",
              equation="-div(sigma(u)) = f, with "
                       "sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I and "
                       "eps(u) = (grad(u) + grad(u)^T)/2",
              coefficients=f"Young's modulus E = {E}, Poisson ratio nu = {nu}, "
                           f"so lambda = {lam} and mu = {mu}. Small strain, "
                           f"linear elastic.",
              bc_text="u = 0 (all three components) on the entire boundary of "
                      "the cube",
              element="first-order (linear) tetrahedral or hexahedral "
                      "displacement elements, small-displacement formulation",
              mesh_text="uniform meshes of the unit cube with N = 4, 8, 16 "
                        "cells per side, all three levels")
    return s, u, f, co, dict(kind="vector", lam=lam, mu=mu)


def cell_KR2(d):
    """Kratos, CURVED geometry. Every probe point lies strictly inside the disk,
    which is asserted at build time, so the grader's fixed grid still applies."""
    R, cx, cy = sp.Rational(3, 4), sp.Rational(1, 2), sp.Rational(1, 2)
    u, f, co = disk_diffusion(d, R, cx, cy)
    s = _base("KR2", "kratos", "Kratos Multiphysics",
              "steady diffusion on a curved (circular) domain", "curved", 2,
              [8, 16, 32], 2.0, 0.4, BAND_P1,
              domain="the DISK of radius 3/4 centred at (1/2, 1/2), i.e. all "
                     "(x,y) with (x-1/2)**2 + (y-1/2)**2 < (3/4)**2. The "
                     "boundary is a circle, not a polygon; the mesh must "
                     "follow it.",
              equation="-lap(u) = f  (steady diffusion, unit conductivity)",
              coefficients="conductivity 1",
              bc_text="u = 0 on the whole circular boundary",
              element="first-order (linear) triangular elements with STRAIGHT "
                      "edges, so that the polygon the mesh forms approximates "
                      "the circle. Use first order: with straight-edged "
                      "elements a higher order would be limited by the "
                      "geometry error anyway.",
              mesh_text="unstructured triangular meshes of the disk with target "
                        "element size h = 1/8, 1/16, 1/32, all three levels. "
                        "Halve the target size from level to level.",
              notes_public="every probe point listed below lies strictly "
                           "inside the disk.")
    return s, u, f, co, dict(kind="scalar", K=sp.eye(2))


def cell_DU1(d):
    """DUNE, continuous-Galerkin MMS."""
    K = sp.Matrix([[sp.Rational(5, 2), 0], [0, 1]])
    u, f, co = scalar_diffusion(d, K, 2)
    s = _base("DU1", "dune", "DUNE (dune-fem, the Python bindings)",
              "orthotropic diffusion, continuous Galerkin",
              "scalar-straightforward", 2, [8, 16, 32, 64], 2.0, 0.4, BAND_P1,
              domain="the unit square (0,1) x (0,1)",
              equation="-div(K grad u) = f",
              coefficients="constant diagonal tensor K = [[5/2, 0], [0, 1]]",
              bc_text="u = 0 on the entire boundary",
              element="continuous Lagrange elements of order 1 on triangles "
                      "(a conforming, continuous Galerkin discretisation -- not "
                      "a discontinuous one)",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32, "
                        "64 cells per side, all four levels")
    return s, u, f, co, dict(kind="scalar", K=K)


def cell_DU2(d):
    """DUNE, interior-penalty DG. Penalty and degree pinned, and the problem is
    diffusion-dominated so that h^(k+1) is the right bound."""
    eps = sp.Integer(1)
    b = [sp.Integer(2), sp.Integer(1)]
    u, f, co = advection_diffusion(d, eps, b)
    s = _base("DU2", "dune", "DUNE (dune-fem, the Python bindings)",
              "advection-diffusion, discontinuous Galerkin", "DG", 2,
              [8, 16, 32], 3.0, 0.5, [1.8, 4.4],
              domain="the unit square (0,1) x (0,1)",
              equation="-eps*lap(u) + b . grad(u) = f  "
                       "(steady advection-diffusion)",
              coefficients="eps = 1 and b = (2, 1). The Peclet number is order "
                           "one, so the problem is diffusion-dominated.",
              bc_text="u = 0 on the entire boundary, imposed WEAKLY by the "
                      "penalty terms of the scheme (Nitsche / interior-penalty "
                      "boundary terms), not by eliminating degrees of freedom",
              element="symmetric interior-penalty discontinuous Galerkin "
                      "(SIPG) with discontinuous Lagrange elements of order 2 "
                      "on triangles, and the penalty parameter fixed at "
                      "beta = 20 (so the penalty term is beta*p*p/h times the "
                      "jump, with p the polynomial order). Use exactly this "
                      "scheme, order and penalty.",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32 "
                        "cells per side, all three levels")
    return s, u, f, co, dict(kind="scalar", K=eps * sp.eye(2), advect=b)


# FEBio parses every math expression through MObjBuilder::Create, which does
#
#     char szcopy[512] = { 0 };
#     strcpy(szcopy, ex.c_str());
#
# so an expression of 512 characters or more overruns the buffer and the
# process dies with "*** buffer overflow detected ***" AFTER printing "Reading
# file ... SUCCESS!". Measured on the installed 4.12.0 build: a 1262-character
# body-force component killed it every time, in both the modern
# `<force type="math">` form and the legacy `<body_load type="non-const">` one.
#
# So an FEBio cell's source term is length-constrained, and the constraint is
# checked here rather than discovered by a failed campaign run.
FEBIO_EXPR_LIMIT = 511


def cell_FB1(d):
    """FEBio, plane-strain elasticity on a one-element-thick slab.

    Small strain, and deliberately SMALL: FEBio has no small-strain solid
    material -- its "isotropic elastic" is St.Venant-Kirchhoff pushed forward
    (read off FEIsotropicElastic::Stress). The finite-strain correction is
    relative O(|grad u|), so the amplitude is set to make |grad u| ~ 5e-5, two
    orders below the discretisation error at the finest prescribed level. The
    alternative -- manufacturing against the exact St.Venant-Kirchhoff operator
    -- was built and rejected: its source term runs to 1262 characters per
    component and FEBio's math parser overflows at 512.
    """
    E, nu = sp.Integer(1000), sp.Rational(1, 4)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    u, f, co = elasticity(d, lam, mu, 2, amp=sp.Rational(1, 100000))
    s = _base("FB1", "febio",
              "FEBio (write a .feb input file and run the febio4 binary)",
              "linear elasticity, plane strain", "vector", 2, [8, 16, 32],
              2.0, 0.4, BAND_P1, components=["ux", "uy"],
              domain="the unit square (0,1) x (0,1), in PLANE STRAIN. FEBio is "
                     "a three-dimensional code, so model this as the slab "
                     "(0,1) x (0,1) x (0, 1/8) with exactly ONE element through "
                     "the thickness and the z displacement fixed to zero at "
                     "EVERY node. With the load and the geometry independent of "
                     "z that reproduces plane strain exactly, and the in-plane "
                     "solution does not depend on the slab thickness.",
              equation="-div(sigma(u)) = f, with "
                       "sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I and "
                       "eps(u) = (grad(u) + grad(u)^T)/2 (small strain, plane "
                       "strain). f is a force per unit volume.",
              coefficients=f"Young's modulus E = {E}, Poisson ratio nu = 1/4, "
                           f"so lambda = {lam} and mu = {mu}. Mass density 1.",
              bc_text="u = 0 (both in-plane components) on the four lateral "
                      "faces x = 0, x = 1, y = 0 and y = 1; and uz = 0 at every "
                      "node of the model",
              element="hex8 trilinear elements, ONE element through the "
                      "thickness, standard displacement formulation. If the "
                      "solver is a nonlinear one, converge it tightly: "
                      "displacement and energy tolerances of 1e-10 or smaller "
                      "at every level.",
              notes_public="two things about this problem are deliberate and "
                           "must not be 'fixed'. (1) The displacement scale "
                           "here is small, of order 1e-6; do not rescale the "
                           "source term, and do report full precision. (2) f "
                           "is defined by the "
                           "equation exactly as printed. A solver's body-force "
                           "input may use the opposite sign convention, or a "
                           "force per unit MASS rather than per unit volume; "
                           "check what the code you use expects before you "
                           "supply it.",
              mesh_text="uniform meshes of the square with N = 8, 16, 32 "
                        "elements per side in x and y (h = 1/N) and one element "
                        "in z, all three levels",
              graded_note="\nReport the two IN-PLANE displacement components "
                          "ux and uy. They may be taken at any z: the solution "
                          "does not depend on it.\n")
    return s, u, f, co, dict(kind="vector", lam=lam, mu=mu,
                             expr_limit=FEBIO_EXPR_LIMIT)


def cell_FC2(d):
    """4C, near-incompressible plane strain with the EAS element pinned.

    The Cook's-membrane trap in the form the grader can score: a plain
    displacement HEX8 at nu = 0.4999 locks, and locking has no convergence
    order at all, so the element technology has to be part of the statement for
    the theoretical order to mean anything. Same pressure-first construction as
    FE2 -- p is O(1) and div u is O(1/lambda) -- because a manufactured field
    with div u of order one turns a near-incompressible cell back into a
    compressible one.
    """
    mu, lam = sp.Integer(1000), sp.Integer(4999000)
    nu = sp.Rational(lam, 2 * (lam + mu))
    E = sp.Rational(mu * (3 * lam + 2 * mu), lam + mu)
    u, p, f, co = near_incompressible(d, mu, lam)
    s = _base("FC2", "4C", "4C (write a .4C.yaml input file and run the 4C "
              "binary)",
              "near-incompressible linear elasticity (nu = 0.4999), plane "
              "strain", "near-incompressible/locking", 2, [8, 16, 32], 2.0,
              0.4, BAND_P1, components=["ux", "uy"],
              domain="the unit square (0,1) x (0,1), in PLANE STRAIN. 4C's "
                     "solid elements are three-dimensional, so model this as "
                     "the slab (0,1) x (0,1) x (0, 1/8) with exactly ONE "
                     "element through the thickness and the z displacement "
                     "fixed to zero at EVERY node. With the load and the "
                     "geometry independent of z that reproduces plane strain "
                     "exactly, and the in-plane solution does not depend on the "
                     "slab thickness.",
              equation="-div(sigma(u)) = f, with "
                       "sigma(u) = 2*mu*eps(u) + lambda*tr(eps(u))*I and "
                       "eps(u) = (grad(u) + grad(u)^T)/2 (small strain, plane "
                       "strain). f is a force per unit volume.",
              coefficients=f"Lame parameters lambda = {lam} and mu = {mu} "
                           f"(exactly), i.e. Poisson ratio nu = "
                           f"{float(nu):.4f} and Young's modulus "
                           f"E = {float(E):.1f}. The material is NEARLY "
                           f"INCOMPRESSIBLE: lambda/mu = {int(lam / mu)}.",
              bc_text="u = 0 (both in-plane components) on the four lateral "
                      "faces x = 0, x = 1, y = 0 and y = 1; and uz = 0 at every "
                      "node of the model",
              element="HEX8 solid elements with LINEAR kinematics and the "
                      "ENHANCED ASSUMED STRAIN technology switched on (in 4C: "
                      "the SOLID element with KINEM linear and TECH eas_full). "
                      "Use exactly this. A plain displacement HEX8 locks "
                      "volumetrically at this Poisson ratio: its error stops "
                      "falling under refinement, so it has no convergence order "
                      "to be judged against.",
              notes_public="f is defined by the equation exactly as printed. A "
                           "solver's volume-load input may use a different sign "
                           "convention or a different normalisation; check what "
                           "the code you use expects before you supply it.",
              mesh_text="uniform meshes of the square with N = 8, 16, 32 "
                        "elements per side in x and y (h = 1/N) and one element "
                        "in z, all three levels",
              graded_note="\nONLY THE DISPLACEMENT IS GRADED. Report the two "
                          "in-plane components ux and uy; do not report the "
                          "pressure. They may be taken at any z: the solution "
                          "does not depend on it.\n")
    return s, u, f, co, dict(kind="mixed_elasticity", mu=mu, lam=lam, p=p)


def cell_FC1(d):
    """4C, TRANSIENT heat conduction MMS, integrator and dt sequence pinned.

    The D8 lesson applies: T(t) = t*exp(t/2) is deliberately not a low-degree
    polynomial, because a quadratic T makes one-step-theta at theta = 1/2
    exact in time and the cell silently stops testing the integrator.
    """
    kappa = sp.Integer(1)
    tend = sp.Rational(1, 4)
    u, f, co, u_tend = transient_diffusion(d, kappa, tend)
    s = _base("FC1", "4C", "4C (write a .4C.yaml input file and run the 4C "
              "binary)",
              "transient heat conduction", "transient", 2, [8, 16, 32], 2.0,
              0.4, BAND_P1,
              domain="the unit square (0,1) x (0,1)",
              equation="dT/dt - div(k grad T) = f  (transient heat conduction "
                       "with volumetric heat capacity 1)",
              coefficients="conductivity k = 1, volumetric heat capacity 1",
              initial_condition="T = 0 everywhere at t = 0",
              bc_text="T = 0 on the entire boundary at all times",
              element="THERMO QUAD4 elements (bilinear, continuous) on "
                      "quadrilaterals",
              time_grid="one-step-theta with THETA = 1/2 (Crank-Nicolson), "
                        "integrated to t_end = 1/4 with dt = h/4, i.e. "
                        "dt = 1/32, 1/64, 1/128 on the three levels (8, 16 and "
                        "32 steps). Use exactly this integrator, this theta and "
                        "these step counts: the temporal order is part of what "
                        "is assessed, and a different theta has a different "
                        "one. Report the solution at t = t_end.",
              mesh_text="uniform meshes of the unit square with N = 8, 16, 32 "
                        "elements per side (h = 1/N), with dt halved alongside "
                        "h as stated above, all three levels",
              graded_note="\nReport the temperature at t = t_end = 1/4 at the "
                          "probe points.\n")
    return s, u, f, co, dict(kind="transient_scalar", K=kappa * sp.eye(2),
                             u_tend=u_tend, tend=tend)


def cell_FB2(d):
    """FEBio, transient quasi-static VISCOELASTICITY, manufactured history.

    u(x,t) = g(t) U(x) with g(t) = (t/tend)^2. At small strain the FEBio
    viscoelastic stress (G(t) = g0 + g1 exp(-t/t1) convolving the FULL elastic
    PK2 stress -- read off FEViscoElasticMaterial::Stress) separates:
    S(t) = Gconv(t) * sigma_e(U), so equilibrium gives the body force
    f(x,t) = Gconv(t) * f_spatial(x) with f_spatial = -div(sigma_e(U)) and

        Gconv(t) = g0 g(t) + g1 * int_0^t exp(-(t-s)/t1) g'(s) ds
                 = [g0 t^2 + 2 g1 t1 (t - t1 (1 - e^(-t/t1)))] / tend^2.

    g quadratic, NOT linear, for the D8 reason in viscoelastic form: FEBio's
    recurrence integrates the exponential exactly for a stress history that is
    linear over the step, so a linear g would make the time integration exact
    and the cell would silently stop testing it. Quadratic g leaves the O(dt^2)
    per-history error in place, matching the spatial order with dt = h.
    """
    E, nu = sp.Integer(1000), sp.Rational(1, 4)
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    g0, g1, t1 = sp.Rational(1, 2), sp.Rational(1, 2), sp.Rational(1, 4)
    tend = sp.Integer(1)
    U, f_spatial, co = elasticity(d, lam, mu, 2, amp=sp.Rational(1, 100000))
    gt = (t / tend) ** 2
    Gconv = sp.expand(g0 * t ** 2
                      + 2 * g1 * t1 * (t - t1 * (1 - sp.exp(-t / t1)))) / tend ** 2
    u_tend = [sp.expand(c * 1) for c in U]          # g(tend) = 1
    s = _base("FB2", "febio",
              "FEBio (write a .feb input file and run the febio4 binary)",
              "quasi-static viscoelasticity, transient load", "transient", 2,
              [8, 16, 32], 2.0, 0.4, BAND_P1, components=["ux", "uy"],
              domain="the unit square (0,1) x (0,1), in PLANE STRAIN. FEBio is "
                     "a three-dimensional code, so model this as the slab "
                     "(0,1) x (0,1) x (0, 1/8) with exactly ONE element through "
                     "the thickness and the z displacement fixed to zero at "
                     "EVERY node.",
              equation="QUASI-STATIC VISCOELASTICITY: div(sigma(t)) + f(x,t) = "
                       "0 at every instant, with the viscoelastic stress "
                       "sigma(t) = integral_0^t G(t-s) d/ds sigma_e(u(s)) ds, "
                       "relaxation function G(t) = g0 + g1*exp(-t/t1), and "
                       "sigma_e the small-strain elastic stress "
                       "2*mu*eps(u) + lambda*tr(eps(u))*I. The body starts "
                       "unloaded and at rest: u = 0 and sigma = 0 at t = 0.",
              coefficients=f"elastic part: E = {E}, nu = 1/4 (lambda = {lam}, "
                           f"mu = {mu}). Viscoelastic part: g0 = 1/2, "
                           f"g1 = 1/2, t1 = 1/4 (so G(0) = 1 and "
                           f"G(inf) = 1/2). Mass density 1.",
              initial_condition="u = 0 everywhere at t = 0; the load below is "
                                "zero at t = 0 and grows in time",
              bc_text="u = 0 (both in-plane components) on the four lateral "
                      "faces x = 0, x = 1, y = 0 and y = 1 at all times; and "
                      "uz = 0 at every node of the model",
              element="hex8 trilinear elements, ONE element through the "
                      "thickness, standard displacement formulation, "
                      "quasi-static (no inertia)",
              time_grid="integrate from t = 0 to t_end = 1 with UNIFORM steps "
                        "dt = h, i.e. dt = 1/8, 1/16, 1/32 on the three levels "
                        "(8, 16 and 32 steps). Use exactly these step counts; "
                        "the time discretisation of the relaxation integral is "
                        "part of what is assessed. Report the solution at "
                        "t = t_end.",
              notes_public="two things are deliberate and must not be 'fixed'. "
                           "(1) The displacement scale is small, of order "
                           "1e-6; do not rescale the load, and do report full "
                           "precision. (2) f is defined by the equilibrium "
                           "equation exactly as printed. A solver's body-force "
                           "input may use the opposite sign convention, or a "
                           "force per unit MASS rather than per unit volume; "
                           "check what the code you use expects before you "
                           "supply it.",
              mesh_text="uniform meshes of the square with N = 8, 16, 32 "
                        "elements per side in x and y (h = 1/N) and one element "
                        "in z, with dt = h as stated above, all three levels",
              graded_note="\nReport the two IN-PLANE displacement components "
                          "ux and uy AT t = t_end = 1. They may be taken at "
                          "any z: the solution does not depend on it.\n")
    return s, u_tend, (f_spatial, Gconv), co, dict(
        kind="viscoelastic", lam=lam, mu=mu, U=U, gt=gt,
        expr_limit=FEBIO_EXPR_LIMIT)


CELLS = [cell_FE1, cell_FE2, cell_DL1, cell_DL2, cell_NG1, cell_NG2,
         cell_SK1, cell_SK2, cell_KR1, cell_KR2, cell_DU1, cell_DU2,
         cell_FB1, cell_FC2, cell_FC1, cell_FB2]


# ── build, verify, seal ───────────────────────────────────────────────────
def _fmt_source(spec, f):
    coords = ", ".join(spec["coords"])
    comps = spec.get("components")
    if comps and isinstance(f, (list, tuple)):
        pad = " " * (len(coords) + 4)
        return "\n".join(
            (f"f_{c}({coords}) = " if i == 0 else f"{pad}f_{c}({coords}) = ")
            + sp.sstr(fi) for i, (c, fi) in enumerate(zip(comps, f)))
    return f"f({coords}) = {sp.sstr(f)}"


def build_one(fn, seed=None):
    d = Draw(seed)
    spec, u, f, coords, aux = fn(d)
    spec["draw_seed"] = d.seed
    dim = spec["dim"]
    assert_no_drift(dim)
    G.assert_probe_grid_incommensurate(dim, spec["mesh_N"])

    kind = aux["kind"]
    checks, numeric = {}, {}
    src_override, f_text_override = None, None
    if kind == "transient_scalar":
        sym, num = check_scalar(u, f, coords, aux["K"], transient=True)
        checks["strong_form"] = (sym == 0)
        numeric["residual"] = num
        graded = [aux["u_tend"]]
        src_override = sp.sstr(f)
        f_text_override = f"f(x, y, t) = {src_override}"
    elif kind == "viscoelastic":
        f_spatial, GC = f
        ok, num = check_vector(aux["U"], f_spatial, coords, aux["lam"],
                               aux["mu"])
        checks["spatial_strong_form"] = ok
        numeric["residual"] = num
        # The time factor must satisfy the convolution ODE it claims to:
        # for I(t) = int_0^t e^{-(t-s)/t1} g'(s) ds,  I' + I/t1 = g'(t), I(0)=0.
        g0v, g1v, t1v = (sp.Rational(1, 2), sp.Rational(1, 2),
                         sp.Rational(1, 4))
        part = sp.expand(GC - g0v * aux["gt"])
        ode = sp.simplify(sp.diff(part, t) + part / t1v
                          - g1v * sp.diff(aux["gt"], t))
        checks["relaxation_ode"] = (ode == 0 and sp.simplify(GC.subs(t, 0)) == 0)
        graded = list(u)                       # u here is u(., t_end)
        GCs = sp.sstr(GC)
        src_override = [f"({GCs})*({sp.sstr(c)})" for c in f_spatial]
        f_text_override = ("f_ux(x, y, t) = " + src_override[0]
                           + "\n               f_uy(x, y, t) = "
                           + src_override[1])
    elif kind == "scalar":
        sym, num = check_scalar(u, f, coords, aux["K"],
                                reaction=aux.get("reaction", 0),
                                advect=aux.get("advect"),
                                nonlinear_a=aux.get("nonlinear_a"),
                                transient=aux.get("transient", False))
        checks["strong_form"] = (sym == 0)
        numeric["residual"] = num
        graded = [u]
    elif kind == "vector":
        ok, num = check_vector(u, f, coords, aux["lam"], aux["mu"])
        checks["strong_form"] = ok
        numeric["residual"] = num
        graded = list(u)
    elif kind == "stokes":
        ok, num, ndiv = check_stokes(u, aux["p"], f, coords, aux["mu"])
        checks["strong_form_and_incompressibility"] = ok
        numeric["residual"] = num
        numeric["div_u"] = ndiv
        graded = list(u)
    elif kind == "navier_stokes":
        ok, num, ndiv = check_navier_stokes(u, aux["p"], f, coords, aux["nu"])
        checks["strong_form_and_incompressibility"] = ok
        numeric["residual"] = num
        numeric["div_u"] = ndiv
        graded = list(u)
    elif kind == "mixed_elasticity":
        ok, num = check_mixed_elasticity(u, aux["p"], f, coords, aux["mu"],
                                         aux["lam"])
        checks["two_field_strong_form"] = ok
        numeric["residual"] = num
        graded = list(u)
    elif kind == "stvk":
        ok, num = check_stvk(u, f, coords, aux["lam"], aux["mu"])
        checks["reference_equilibrium"] = ok
        numeric["residual"] = num
        graded = list(u)
    elif kind == "biharmonic":
        ok, num = check_biharmonic(u, f, coords)
        checks["strong_form"] = ok
        numeric["residual"] = num
        graded = [u]
    else:
        raise ValueError(kind)

    # boundary data, verified rather than asserted
    bmax = 0.0
    for c in coords:
        for val in (0, 1):
            for gu in graded:
                r = sp.simplify(gu.subs(c, val))
                if spec["id"] == "KR2":
                    continue                      # curved boundary, checked below
                bmax = max(bmax, 0.0 if r == 0 else 1.0)
    if spec["id"] == "KR2":
        R = sp.Rational(3, 4)
        th = sp.Symbol("th", real=True)
        onb = sp.simplify(u.subs({x: sp.Rational(1, 2) + R * sp.cos(th),
                                  y: sp.Rational(1, 2) + R * sp.sin(th)}))
        bmax = 0.0 if sp.simplify(onb) == 0 else 1.0
        if not all_probes_inside(2, lambda px, py: (px - 0.5) ** 2
                                 + (py - 0.5) ** 2 < float(R) ** 2):
            raise AssertionError("KR2: a probe point lies outside the disk")
    if spec["id"] == "SK2":
        # clamped: the normal derivative must vanish too
        for c, other in ((x, y), (y, x)):
            for val in (0, 1):
                if sp.simplify(sp.diff(u, c).subs(c, val)) != 0:
                    bmax = 1.0
    checks["boundary_data"] = (bmax == 0.0)

    exprs = [sp.sstr(e) for e in graded]
    if src_override is not None:
        src = src_override
    else:
        src = ([sp.sstr(e) for e in f] if isinstance(f, (list, tuple))
               else sp.sstr(f))
    lim = aux.get("expr_limit")
    if lim:
        for i, e in enumerate(src if isinstance(src, list) else [src]):
            if len(e) > lim:
                raise AssertionError(
                    f"{spec['id']}: source component {i} is {len(e)} characters "
                    f"and the target solver's math parser overflows at {lim}. "
                    f"This is a hard property of the installed build, not a "
                    f"style preference: it segfaults after reporting the input "
                    f"read successfully.")
    f_text = (f_text_override if f_text_override is not None
              else _fmt_source(spec, f))
    task = build_task(spec, f_text)
    gate = scan(task, {"exact_solution": exprs, "source_term": src},
                spec["id"])

    rms = exact_rms(graded, spec["coords"], dim)
    spec["exact_rms"] = rms

    ok = (all(checks.values()) and all(v < 1e-5 for v in numeric.values())
          and gate.clean)
    return dict(spec=spec, task=task, exact=exprs, source=src, checks=checks,
                numeric=numeric, gate=gate, exact_rms=rms, ok=ok)


def precision_floor_ok(rms: float, finest_error: float) -> tuple[bool, float]:
    """The finest level's error must clear the stated output precision."""
    floor = rms * 10.0 ** (-SIG_DIGITS)
    return (finest_error > PRECISION_HEADROOM * floor,
            finest_error / max(floor, 1e-300))


# ── SPARTA band-only cells ────────────────────────────────────────────────
# DSMC has no manufactured solution: the method's own error is transport error
# plus statistical noise going as 1/sqrt(particles per cell), so mesh halving
# at fixed particle count makes the "error" RISE and an order-fit grades the
# noise. These two cells are therefore BAND-ONLY: the graded quantity is a
# single steady scalar, judged against a band PRE-REGISTERED FROM KINETIC
# THEORY (written into the sealed key with its derivation, before any run),
# plus a conservation identity a real steady DSMC run must satisfy. They are
# evidence grade 3 and are never pooled with the manufactured-solution cells.
#
# The refinement rule replaces mesh halving: the GRID IS FIXED (cell size
# ~ lambda/3) and the particle count is scaled 4x per level, so the
# statistical error halves per level while the transport error stays fixed.
# dt is pinned below the mean collision time.
#
# All numbers below are derived, not tuned: argon VHS viscosity from Bird's
# eq. 4.63 with the species parameters printed in the task, hard-sphere mean
# free path, first-order slip / temperature-jump corrections. The bands are
# wide (about +-20% around the slip-corrected value) to cover the VSS-vs-VHS
# transport difference, higher-order Knudsen corrections and residual noise;
# they exist to catch answers wrong in kind, not to police the third digit.
_SP = dict(
    m=6.63e-26, dref=4.17e-10, omega=0.81, Tref=273.0, alpha=1.4,
    kB=1.380649e-23, n=2.6e22, H=1.0e-3, T0=273.0)


def _sparta_theory():
    import math
    p = _SP
    mu_ref = (15.0 * math.sqrt(math.pi * p["m"] * p["kB"] * p["Tref"])
              / (2.0 * math.pi * p["dref"] ** 2
                 * (5.0 - 2.0 * p["omega"]) * (7.0 - 2.0 * p["omega"])))
    lam = 1.0 / (math.sqrt(2.0) * math.pi * p["n"] * p["dref"] ** 2)
    vmean = math.sqrt(8.0 * p["kB"] * p["T0"] / (math.pi * p["m"]))
    tau_c = lam / vmean
    kth = 3.75 * (p["kB"] / p["m"]) * mu_ref     # monatomic: k = 15/4 R mu
    return mu_ref, lam, vmean, tau_c, kth


SPARTA_CONTRACT = """
REQUIRED OUTPUT (this is how your work is assessed; follow it exactly):
For every level k = 1, 2, 3 write
    qoi_level<k>.csv
with the header line `qoi, conservation_residual` and exactly one data row:
  * qoi -- {qoi_desc}, time-averaged over the stated sampling window, in SI
    units ({qoi_units});
  * conservation_residual -- {cons_desc}.
Also write run_level<k>.log containing at least the line
    NDOF = <integer>
where <integer> is the TOTAL number of simulator particles at that level (the
particle count plays the role the degree-of-freedom count plays elsewhere).
Then write RESULT.txt with exactly these lines:
    LEVELS = <number of levels run>
    FILES = <comma-separated list of the CSV files you wrote>
    MESH_INDEPENDENCE = <CONVERGED or NOT_CONVERGED>
    MAX_REL_CHANGE = <largest relative change of the qoi between the two
finest levels>
Judge MESH_INDEPENDENCE solely from your own results: the qoi should move by
less than the statistical scatter between the two finest levels.

INTEGRITY (absolute): every number must come from ACTUALLY RUNNING SPARTA.
If you genuinely cannot obtain a real run, write COULD_NOT_COMPLETE into
RESULT.txt and explain why. Inventing numbers is a failure, worse than an
honest incomplete.
"""


def emit_sparta(problems: Path, keys: Path):
    import math
    mu_ref, lam, vmean, tau_c, kth = _sparta_theory()
    p = _SP
    Kn = lam / p["H"]
    dt = 2.0e-8
    assert dt < tau_c / 5
    gas = (f"argon, molecular mass m = {p['m']:.3g} kg, VHS/VSS reference "
           f"diameter d_ref = {p['dref']:.3g} m at T_ref = {p['Tref']:.0f} K, "
           f"viscosity-temperature exponent omega = {p['omega']}, VSS alpha = "
           f"{p['alpha']}. Use exactly these species parameters")
    common = (f"the 2D channel 0 < y < H with H = {p['H']:.1e} m between two "
              f"infinite parallel walls (model the x direction as periodic, "
              f"width at least 10 grid cells). Gas: {gas}. Number density "
              f"n = {p['n']:.3g} 1/m^3, so the hard-sphere mean free path is "
              f"lambda = 1/(sqrt(2) pi n d_ref^2) = {lam:.3g} m and "
              f"Kn = lambda/H = {Kn:.3f} (slip regime)")
    refine = (f"THE GRID IS FIXED AND THE PARTICLE COUNT REFINES. Use a "
              f"uniform grid with cell size no larger than lambda/3 "
              f"(at least 60 cells across the channel) at EVERY level, and "
              f"timestep dt = {dt:.1e} s, which is below a fifth of the mean "
              f"collision time lambda/vmean = {tau_c:.3g} s. Level 1: at "
              f"least 20 particles per cell on average. Level 2: 4x the "
              f"particles of level 1 (quarter the fnum). Level 3: 4x the "
              f"particles of level 2. DSMC statistical error scales as "
              f"1/sqrt(particles per cell), so it halves per level while the "
              f"transport error stays fixed; do NOT refine the grid instead, "
              f"and do NOT change dt. SAMPLING WINDOW: run 20000 steps to "
              f"reach steady state without sampling, then average the "
              f"reported quantities over the next 30000 steps, at every level")

    mkcell = {}
    tau0 = mu_ref * 100.0 / p["H"]
    tau_slip = mu_ref * 100.0 / (p["H"] + 2 * 1.14 * lam)
    mkcell["SP1"] = dict(
        physics="plane Couette flow, wall shear stress (DSMC)",
        qoi_desc=("the magnitude of the time-averaged shear stress tau_xy "
                  "exerted by the gas on EITHER wall (state which; at steady "
                  "state they agree)"),
        qoi_units="Pa",
        cons_desc=("|tau_top - tau_bottom| / (0.5*(|tau_top| + |tau_bottom|)) "
                   "-- at steady state the momentum flux through the channel "
                   "is constant, so the two wall shear stresses must balance"),
        setup=(f"{common}. The two walls are fully diffuse at T_wall = "
               f"{p['T0']:.0f} K and move in x with velocities +50 m/s (top, "
               f"y = H) and -50 m/s (bottom, y = 0), so the relative speed is "
               f"U = 100 m/s. Initialise the gas at rest at {p['T0']:.0f} K"),
        band=[1.5, 2.3], cons_tol=0.05,
        derivation=(
            f"VHS viscosity (Bird 1994 eq. 4.63) from the printed species "
            f"parameters: mu_ref = 15 sqrt(pi m kB Tref) / (2 pi dref^2 "
            f"(5-2w)(7-2w)) = {mu_ref:.4g} Pa s at 273 K (true argon: "
            f"2.12e-5). Continuum Couette: tau0 = mu U / H = {tau0:.4g} Pa. "
            f"First-order slip with c_m = 1.14: tau = mu U / (H + 2 c_m "
            f"lambda) = {tau_slip:.4g} Pa at Kn = {Kn:.3f}. Band [1.5, 2.3] "
            f"Pa brackets the slip value by about +-20% to cover the "
            f"VSS-vs-VHS transport difference, higher-order Knudsen "
            f"corrections and residual statistical noise. Derived from "
            f"theory before any run; no SPARTA run informed these numbers."))

    q0 = kth * 100.0 / p["H"]
    q_jump = kth * 100.0 / (p["H"] + 2 * 2.18 * lam)
    mkcell["SP2"] = dict(
        physics="heat conduction between parallel walls, wall heat flux (DSMC)",
        qoi_desc=("the magnitude of the time-averaged normal heat flux "
                  "through EITHER wall (state which; at steady state they "
                  "agree)"),
        qoi_units="W/m^2",
        cons_desc=("|q_hot - q_cold| / (0.5*(|q_hot| + |q_cold|)) -- at "
                   "steady state no energy accumulates in the gas, so the "
                   "heat entering at the hot wall must leave at the cold one"),
        setup=(f"{common}. Both walls are stationary and fully diffuse: the "
               f"bottom wall (y = 0) at T = 223 K, the top wall (y = H) at "
               f"T = 323 K. Initialise the gas at rest at 273 K"),
        band=[1050.0, 1700.0], cons_tol=0.05,
        derivation=(
            f"Monatomic-gas conductivity k = (15/4)(kB/m) mu = {kth:.4g} "
            f"W/(m K) at 273 K from the VHS viscosity {mu_ref:.4g} Pa s. "
            f"Continuum Fourier: q0 = k dT / H = {q0:.4g} W/m^2 for dT = "
            f"100 K. Temperature-jump correction with c_t = 2.18 (gamma = "
            f"5/3, Pr = 2/3, full accommodation): q = k dT / (H + 2 c_t "
            f"lambda) = {q_jump:.4g} W/m^2. Band [1050, 1700] W/m^2 brackets "
            f"the jump-corrected value, the k(T) ~ T^omega nonlinearity of "
            f"the profile across 223..323 K, VSS-vs-VHS differences and "
            f"noise. Derived from theory before any run; no SPARTA run "
            f"informed these numbers."))

    for pid, c in mkcell.items():
        task = "\n".join([
            f"Using SPARTA (the DSMC code; run the spa_serial binary), "
            f"simulate the following problem.",
            "",
            f"SETUP: {c['setup']}.",
            f"QUANTITY OF INTEREST: {c['qoi_desc']}, in {c['qoi_units']}.",
            f"DISCRETISATION AND REFINEMENT: {refine}.",
            "",
            SPARTA_CONTRACT.format(qoi_desc=c["qoi_desc"],
                                   qoi_units=c["qoi_units"],
                                   cons_desc=c["cons_desc"]),
        ])
        spec = dict(
            id=pid, kind="sparta_band", code="sparta", code_label="SPARTA",
            physics=c["physics"], regime="DSMC/band", dim=2,
            coords=["x", "y"], mesh_N=[1, 2, 3],
            grading="band-only, grade 3, pending grader path",
            evidence_grade=3,
            evidence_grade_reason=(
                "DSMC: no manufactured solution exists. Graded on a "
                "pre-registered kinetic-theory band plus a conservation "
                "identity; NEVER pooled with the manufactured-solution "
                "cells."),
            refinement_rule="fixed grid, particles x4 per level, dt fixed "
                            "below the mean collision time",
            qoi=c["qoi_desc"],
        )
        pdir = problems / pid
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "task.txt").write_text(task, encoding="utf-8")
        (pdir / "spec_public.json").write_text(
            json.dumps(spec, indent=2), encoding="utf-8")
        kdir = keys / pid
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "key.json").write_text(json.dumps({
            "id": pid, "kind": "sparta_band", "code": "sparta",
            "band": c["band"], "band_units": c["qoi_units"],
            "conservation_tol": c["cons_tol"],
            "conservation": c["cons_desc"],
            "derivation": c["derivation"],
            "evidence_grade": 3,
            "grading": "band-only, pending grader path",
        }, indent=2), encoding="utf-8")
        print(f"wrote problems/{pid}/ and keys/{pid}/ (band-only, "
              f"pending grader path; NOT path-verified)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--problems-root", default=None,
                    help="write task/spec here instead of campaign3_blind/"
                         "problems — use a fresh root for a new draw so a "
                         "spent instance is never overwritten")
    ap.add_argument("--keys-root", default=None,
                    help="write answer keys here instead of $OPENPASO_BLIND_KEYS")
    ap.add_argument("--overwrite-spent", action="store_true",
                    help="permit replacing instances that already exist; "
                         "refused by default because a drawn instance, its "
                         "key and the runs graded against it belong together")
    ap.add_argument("--sparta", action="store_true",
                    help="emit the two band-only SPARTA cells (task + spec + "
                         "sealed band key); they carry no manufactured "
                         "solution and are NOT recorded path-verified")
    a = ap.parse_args()

    if a.sparta:
        keys = Path(os.environ.get(
            "OPENPASO_BLIND_KEYS",
            "/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys"))
        emit_sparta(HERE / "problems", keys)
        return 0

    built, failed = [], []
    for fn in CELLS:
        pid = fn.__name__.split("_")[-1]
        if a.only and pid not in a.only:
            continue
        r = build_one(fn, a.seed)
        s = r["spec"]
        worst = max(r["numeric"].values()) if r["numeric"] else 0.0
        print(f"[{s['id']}] {s['code']:8s} {s['physics']}")
        print(f"      regime={s['regime']}  theo={s['theoretical_order']} "
              f"+-{s['tol']}  levels={s['mesh_N']}")
        print(f"      symbolic={ {k: ('OK' if v else 'FAIL') for k, v in r['checks'].items()} }  "
              f"numeric_worst_rel={worst:.2e}  exact_rms={r['exact_rms']:.4g}")
        print(f"      leak gate={'CLEAN' if r['gate'].clean else 'LEAK'} "
              f"{[f.rule for f in r['gate'].findings] or ''}")
        for fnd in r["gate"].findings:
            print(f"        [{fnd.severity}] {fnd.detail[:150]}")
        (built if r["ok"] else failed).append(r)

    print(f"\n{len(built)}/{len(built) + len(failed)} single-code instances "
          f"built, verified and blind.")
    if failed:
        print("FAILED:", [r["spec"]["id"] for r in failed])
        return 1
    if not a.apply:
        print("(dry run -- pass --apply to write tasks and keys)")
        return 0

    problems = Path(a.problems_root) if a.problems_root else HERE / "problems"
    keys = Path(a.keys_root) if a.keys_root else Path(os.environ.get(
        "OPENPASO_BLIND_KEYS",
        "/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys"))

    # A REBUILD MUST NOT DESTROY A SPENT INSTANCE OR ITS ANSWER KEY. Same
    # reasoning as build_balanced.py: this wrote a FIXED id list into
    # problems/<id> and keys/<id>, so drawing the evaluation set would have
    # overwritten the development tasks and the sealed keys the graded rounds
    # depend on.
    clash = [r["spec"]["id"] for r in built
             if (problems / r["spec"]["id"]).exists()
             or (keys / r["spec"]["id"]).exists()]
    if clash and not a.overwrite_spent:
        print(f"REFUSING: {len(clash)} instance(s) already exist and would be "
              f"overwritten: {clash[:8]}{' ...' if len(clash) > 8 else ''}")
        print("  Use --problems-root/--keys-root for a new draw, or "
              "--overwrite-spent to replace them deliberately.")
        return 1

    for r in built:
        s = r["spec"]
        pdir = problems / s["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "task.txt").write_text(r["task"], encoding="utf-8")
        public = {k: v for k, v in s.items()
                  if k not in ("draw_seed", "exact_rms")}
        (pdir / "spec_public.json").write_text(
            json.dumps(public, indent=2, default=str), encoding="utf-8")
        kdir = keys / s["id"]
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "key.json").write_text(json.dumps({
            "id": s["id"], "kind": "single", "code": s["code"],
            "dim": s["dim"], "coords": s["coords"],
            "components": s.get("components", ["u"]),
            "exact_solution": (r["exact"] if len(r["exact"]) > 1
                               else r["exact"][0]),
            "source_term": r["source"],
            "mesh_N": s["mesh_N"], "probe_M": PROBE_M[s["dim"]],
            "theoretical_order": s["theoretical_order"], "tol": s["tol"],
            "band": s["band"],
            # Without this the grader silently degrades to rate-only, and a rate
            # can be manufactured from the published source term alone.
            "exact_rms": r["exact_rms"],
            "sig_digits": SIG_DIGITS,
            "physics": s["physics"], "regime": s["regime"],
            "draw_seed": s["draw_seed"],
            "verification": {"symbolic": r["checks"],
                             "numeric_relative": r["numeric"]},
            "qoi": s["qoi"], "graded_against": s["graded_against"],
        }, indent=2, default=str), encoding="utf-8")
        print(f"wrote problems/{s['id']}/ and keys/{s['id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
