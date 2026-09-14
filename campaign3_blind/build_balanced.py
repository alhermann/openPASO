#!/usr/bin/env python
"""The BALANCED coupled set: twelve pooled instances, eight codes, three each.

Why this file exists
--------------------
The set built by ``build_coupled_v2.py`` is eight instances of which SEVEN have
FEniCSx on one side, and 4C, DUNE, FEBio and SPARTA appear in none of them. A
multi-code-coupling claim measured on that matrix is a claim about FEniCSx. The
replacement is balanced by count -- each of the eight FEM backends appears in
EXACTLY three pairs:

    C1  4C + FEniCSx          C2  4C + Kratos           C3  4C + DUNE
    C4  FEniCSx + deal.II     C5  FEniCSx + scikit-fem  C6  deal.II + NGSolve
    C7  deal.II + FEBio       C8  NGSolve + Kratos      C9  NGSolve + scikit-fem
    C10 Kratos + DUNE         C11 scikit-fem + FEBio    C12 DUNE + FEBio

Two further cells sit OUTSIDE that pool and are built by ``build_offpool.py``:
C13 (FEniCSx + SPARTA, FEM-DSMC, evidence grade 3) and C14 (4C + FEniCSx, FSI,
evidence grade 2). They are outside because the pool's arithmetic is exact --
12 pairs x 2 slots = 24 = 8 backends x 3 appearances -- and because their
evidence grades must never be pooled with grade 1 (DESIGN.md Amendment 2 s4).

This is an EXTENSION, not a fork. Every construction, every verification and the
whole probe/grader contract are imported from ``build_coupled_v2`` and used
unchanged, so a change to the family propagates to both sets and the task text
cannot drift away from ``grade_blind.py``. What is added is (a) fourteen new
instances, (b) a general geometry -- the interface may run along x OR y, and the
instance may be 2-D or 3-D -- and (c) three things the old set left free and
this one PRESCRIBES.

The three prescriptions
-----------------------
**1. The Dirichlet/Neumann role of each subdomain is stated in the task text.**
It was previously the agent's choice, and it is not a free choice: the Neumann
side is measurably harder, and the shipped Kratos participant is Dirichlet-only,
so an instance that needs Kratos on the Neumann side cannot be served by it at
all (measured on D6: 60 iterations, residual stuck at 0.988). Leaving the role
free correlates difficulty with an unrecorded decision. The 24 pooled slots are
split 12 Dirichlet / 12 Neumann, every backend carries both roles, and the split
is checked by ``verify_balance()`` rather than asserted.

**2. Every cell carries a two-material contrast.** Amendment 1 s2 and Amendment 2
s1 record what happens without one: when both sides carry the same field and the
same material the transmission check is a tautology and every reported quantity
agrees with a correct run even when the coupling is wrong. Every instance below
is verified NON-VACUOUS by ``check_scalar_transmission`` /
``check_vector_transmission``, which return VACUOUS rather than PASS in that case.

**3. Every cell excludes the interface ends from grading.** Where the interface
meets a constrained outer boundary the split problem has a Dirichlet-Neumann
corner the monolithic problem does not, and the exported traction there gets
WORSE under refinement (measured 2.11x -> 2.51x over a 4x refinement). The band
is derived by ``build_coupled_v2._iface_band``, never hand-written.

What the twelve actually solve
------------------------------
Four constructions are carried over from the old set because they are the most
discriminating instances it had, and count-balance is not a reason to lose them.
They keep their original code pair and their original roles, so their recorded
execution walks carry over as evidence about the arrangement:

  * C5 -- the notched non-convex domain, bent L-polyline interface, four
    materials (was D5, FEniCSx + scikit-fem);
  * C8 -- the 1:1000 contrast whose role assignment is FORCED (was D6,
    NGSolve + Kratos);
  * C4 -- the transient two-material exchange (was D8, FEniCSx + deal.II);
  * C3 -- genuinely different operators on the two sides (was D7's idea, moved
    to DUNE + 4C, which had no instance of it, and put on a horizontal
    interface).

The paper's three coupled families are named in ``physics_family``: C1 is the
thermo-mechanical chain (the paper names FEniCSx-4C for it), C6 is conjugate
heat transfer, and C13 is FEM-DSMC. Vector participants exist for fenics,
skfem, ngsolve, dealii and febio, and four pairs use elasticity (C7, C9, C11,
C12); for 4C, DUNE and Kratos only scalar participants exist, so those cells are
scalar -- except C12, DUNE + FEBio, where FEBio has no heat module at all, so
the cell must be vector and a DUNE VECTOR participant was written for it.

Run with any interpreter that has sympy:
  python campaign3_blind/build_balanced.py [--apply]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(os.environ.get("OPENPASO_REPO",
                           "/home/alexander/Schreibtisch/ofa-balanced"))
sys.path.insert(0, str(REPO / "src"))

import build_coupled_v2 as V                                # noqa: E402
from blind_eval import coupled as C                         # noqa: E402
from blind_eval.leakgate import scan                        # noqa: E402

x, y, z, t = sp.symbols("x y z t", real=True)

PROBE_M = V.PROBE_M
MESH_N = V.MESH_N
XI = V.XI                      # 5/8: on a mesh line at h = 1/8, 1/16 and 1/32
LX = V.LX                      # 3/2: the long extent, across the interface
GRADE = V.GRADE

# Linear elasticity is a SMALL-STRAIN theory, and a manufactured displacement of
# order 0.3 on a unit body is not in the regime it describes. Every elasticity
# instance is scaled so the strains are of order 1e-5, which is (a) the regime
# the stated model is valid in and (b) the regime in which a finite-deformation
# code -- FEBio has no small-strain material -- reproduces the linear solution
# to far better than the discretisation error. The scale cancels out of every
# graded quantity: the grader's magnitude check is the coarsest error divided by
# the solution's own RMS, and the order is a ratio.
U_AMP = sp.Rational(1, 10 ** 6)

# The label the agent reads, and the key `blind_eval.evidence` knows the code by.
# These MUST be the evidence keys: grade_blind.code_ran looks the code up in
# PER_CODE_SIGNATURES, and an id that is not a key there is unprovable by
# construction and grades FABRICATED_NO_RUN however honestly it ran.
LABEL = {
    "4C": "4C (write a YAML input file and run the 4C binary)",
    "fenics": "FEniCSx (dolfinx)",
    "dealii": "deal.II (build and run the C++ program)",
    "ngsolve": "NGSolve",
    "skfem": "scikit-fem",
    "kratos": "Kratos Multiphysics",
    "dune": "DUNE-fem",
    "febio": "FEBio (write a .feb deck and run the febio4 binary)",
    "sparta": "SPARTA (write an input script and run the spa_serial binary)",
}


# ──────────────────────────────────────────────────────────────────────
# The task text: the imported contract, plus the two things it was missing
# ──────────────────────────────────────────────────────────────────────
ROLE_TEXT = """COUPLING SCHEME: a partitioned Dirichlet-Neumann iteration, with \
subdomain A as the {role_a} side and subdomain B as the {role_b} side. The \
DIRICHLET side receives the interface FIELD from its partner, imposes it as an \
essential boundary condition on the interface, and returns its own outward \
normal {flux_word} there. The NEUMANN side receives that {flux_word} and applies \
it as a natural boundary condition on the interface, and returns its own \
interface field. These roles are prescribed: solve with them as stated rather \
than choosing your own."""

# The execution-evidence contract, of which the task text was the missing half.
#
# blind_eval.evidence requires a line carrying a NUMBER the solver computed, and
# it knew a different phrasing per code: NGSolve's `ndof` and scikit-fem's
# `n_dofs` matched, dolfinx's did not. So an honest, quiet dolfinx run graded
# FABRICATED_NO_RUN while the identical quantity printed by scikit-fem proved a
# run -- a bias that falls on whichever arm happens to use the quieter code. The
# gate now accepts one CANONICAL line for every code; this clause is the other
# half of that contract, and without it a correct quiet run stays unprovable.
# WHY (b) EXISTS: THE OLD CLAUSE COULD BE SATISFIED WITHOUT EITHER CODE.
#
# The clause below used to require only `NDOF = <integer>`, a line the agent
# types. It already said "one file PER PARTICIPANT", but nothing in it could
# distinguish two codes from one script, because the canonical line is
# code-agnostic BY DESIGN -- it exists so a quiet honest dolfinx run is not
# condemned for its print style.
#
# C2_27b_MCP_seed15 is the demonstration. It solved both subdomains with numpy
# and scipy.sparse -- its own docstrings say "direct FEM assembly" -- while the
# task named 4C and Kratos, and no file in the run outside the transcript
# carries a single token of either. Its whole execution evidence was fourteen
# 10-byte files reading `NDOF = <n>`, credited to both named codes at once, and
# its convergence order was genuinely second-order. Everything except "did the
# prescribed codes run" looked right, and that is the one thing a coupled cell
# exists to measure.
#
# The grader cannot repair this: with only the canonical line submitted, the
# honest verdict is "unproven", and punishing an agent for meeting the contract
# as written is not available to us. So the contract now asks for the thing that
# actually settles attribution -- the solver's OWN captured output. Every real
# run can produce it by redirection; a numpy monolith cannot produce 4C's
# banner.
EXEC_LOG_CONTRACT = """
(5) EXECUTION LOG. For every mesh level k and for EACH subdomain, write a file
    run_level<k>_<side>.log             (k = 1,2,3,...; side = A or B)
containing BOTH of the following:
  (a) THE CONSOLE OUTPUT THAT SUBDOMAIN'S SOLVER ITSELF PRODUCED, captured
      verbatim -- redirect the run into the file, for example
          <run command for that subdomain>  > run_level<k>_<side>.log 2>&1
      or raise the code's log/verbosity level from Python so its own messages
      land there. Do not retype, summarise, translate or paraphrase it, and do
      not write it yourself: it must be the text that code emitted.
  (b) the line
          NDOF = <integer>
      where <integer> is the number of degrees of freedom of YOUR
      discretisation of that subdomain at that level.
Write one such file PER PARTICIPANT: the file for side A must carry the output
of the code that solved subdomain A, and the file for side B the output of the
code that solved subdomain B. The two files must therefore carry the output of
two DIFFERENT codes. The assessment reads (a) to establish which code ran on
each side and (b) as the number it cross-checks against the mesh; a side whose
log carries no output from its own named code cannot be credited to that code,
however plausible its numbers are.

    VERBOSITY IS NOT FREE. Some codes print nothing useful by default, so (a)
    needs one extra line. Measured; use the one for your side's code:
      * FEniCSx / dolfinx  silent by default. Before creating the mesh:
            import dolfinx
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
        (the DOLFINX_LOGLEVEL environment variable is NOT honoured)
      * deal.II            needs BOTH deallog.depth_console(2); AND
            SolverControl ctl(max_it, tol, true /*log_history*/,
                              true /*log_result*/);
        depth_console alone prints nothing.
      * NGSolve            ngsolve.ngsglobals.msg_level = 3   (or
        ngsolve.solvers.CG(..., printrates=True), which works at the default)
      * DUNE-fem           parameters={"linear.verbose": True} on galerkin(...)
      * scikit-fem         logging.basicConfig(level=logging.INFO), or
        print(basis). Its log goes to STDERR: redirect with 2>&1 or lose it.
      * Kratos, 4C, FEBio, SPARTA   nothing to set; they print by default.
"""


def build_task(spec: dict, f_text: str) -> str:
    """The imported task text, plus the role prescription and the NDOF log.

    Everything the grader re-derives -- the probe grids, the exclusion rule, the
    file names, the level count -- comes from ``build_coupled_v2.build_task``
    unchanged. Nothing here rewrites it; the two additions are appended as their
    own clauses. That is the whole reason this is an extension rather than a
    second builder: the drift this harness has already suffered seven times
    comes from two texts describing one contract.
    """
    base = V.build_task(spec, f_text)
    flux_word = spec.get("flux_word", "flux")
    role = ROLE_TEXT.format(role_a=spec["roles"]["A"].upper(),
                            role_b=spec["roles"]["B"].upper(),
                            flux_word=flux_word)
    marker = "DISCRETISATION:"
    if marker not in base:                       # pragma: no cover - guard only
        raise RuntimeError("build_coupled_v2.build_task changed shape; the "
                           "role clause has no anchor to attach to")
    base = base.replace(marker, role + "\n" + marker, 1)
    if V.INTEGRITY not in base:                  # pragma: no cover - guard only
        raise RuntimeError("INTEGRITY clause not found in the built task")
    return base.replace(V.INTEGRITY, EXEC_LOG_CONTRACT + V.INTEGRITY, 1)


# ──────────────────────────────────────────────────────────────────────
# Geometry
# ──────────────────────────────────────────────────────────────────────
def _swap(e):
    """Reflect an expression across the diagonal x <-> y."""
    return sp.sympify(e).subs({x: y, y: x}, simultaneous=True)


def _scale(e, c):
    if isinstance(e, (list, tuple)):
        return [sp.expand(c * sp.sympify(v)) for v in e]
    return sp.expand(c * sp.sympify(e))


class Geom:
    """A rectangular box split by one axis-aligned interface.

    ``axis`` 0 puts the interface on x = XI with the long side along x; ``axis``
    1 puts it on y = XI with the long side along y. The second case is not
    cosmetic: every shipped participant samples its partner "by y along a
    straight interface at x = const", so a horizontal interface exercises a
    different outward normal and a different sampling coordinate, at no cost in
    the construction -- the field is the reflection of the vertical one, and both
    the Laplacian and the isotropic elasticity operator are invariant under it.
    """

    def __init__(self, dim: int, axis: int):
        self.dim, self.axis = dim, axis
        self.var = (x, y)[axis]
        self.name = ("x", "y")[axis]
        self.lengths = [float(LX) if i == axis else 1.0 for i in range(dim)]
        self.extent_a = [(0.0, float(XI)) if i == axis else (0.0, 1.0)
                         for i in range(dim)]
        self.extent_b = [(float(XI), float(LX)) if i == axis else (0.0, 1.0)
                         for i in range(dim)]

    @staticmethod
    def _box(ext):
        return " x ".join(f"({a:g}, {b:g})" for a, b in ext)

    def domain_text(self):
        # The doubled braces made this an f-string LITERAL, so 11 of 12 pooled
        # coupled tasks showed the agent "the {2: 'rectangle', 3: 'box'}[2]"
        # as the domain description instead of "the rectangle".
        full = [(0.0, L) for L in self.lengths]
        word = {2: "rectangle", 3: "box"}[self.dim]
        return f"the {word} " + self._box(full)

    def subdomain_text(self, side):
        return self._box(self.extent_a if side == "A" else self.extent_b)

    def interface_text(self):
        what = "line" if self.dim == 2 else "plane"
        return f"the {what} {self.name} = {XI}, where the two materials meet"

    def iface_probe(self):
        """The interface probe rule, with the end-exclusion band applied to
        whichever axes the interface actually spans.

        THE THIRD COPY OF THIS TEXT, AND THE ONE THAT ACTUALLY FEEDS C1..C12.
        There were three: `iface_probe_rule` in build_coupled_v2 (zero callers,
        now deleted), `_straight_iface_probe` there (used by the v2 specs), and
        this one. I fixed the first, found it dead, fixed the second, and only
        caught that C2 still emitted the old grid by reading the task text a
        drawn instance actually contains. That is the whole argument for testing
        what the AGENT ends up with.

        Coordinates now come from V.iface_probe_indices, the single authority,
        so a fourth divergence is not possible without changing that function.
        See it for why the interface points must be a SUBSET of the solution
        probe grid: the fabricated-flux cross-check needs field samples AT an
        interface coordinate, and under the old independent spacing the two sets
        shared none -- solution probes at odd multiples of 1/88, interface
        probes at (45+2i)/176.
        """
        M, js = V.iface_probe_indices(self.dim, V._iface_band(0, 1))
        free = [i for i in range(self.dim) if i != self.axis]
        pt = [""] * self.dim
        pt[self.axis] = f"{XI}"
        tail = ("These cover the INTERIOR of the interface only: where the "
                "interface meets the outer boundary the split problem has a "
                "Dirichlet-Neumann corner, the recovered flux there does not "
                "converge under refinement, and those points are therefore not "
                "graded. They are exactly the solution probe coordinates lying "
                "in the graded band, so the same values appear in your "
                "solution_level<k>_<side>.csv rows")
        if self.dim == 2:
            pt[free[0]] = f"(j+0.5)/{M}"
            return (f"the {len(js)} points (x, y) = ({pt[0]}, {pt[1]}) for "
                    f"j = {js[0]}, {js[0] + 1}, ..., {js[-1]}. " + tail)
        pt[free[0]] = f"(j+0.5)/{M}"
        pt[free[1]] = f"(k+0.5)/{M}"
        return (f"the {len(js) * len(js)} points (x, y, z) = "
                f"({pt[0]}, {pt[1]}, {pt[2]}) for j, k = {js[0]}, "
                f"{js[0] + 1}, ..., {js[-1]}, ordered with k varying "
                f"fastest. " + tail)


# ──────────────────────────────────────────────────────────────────────
# Specs
# ──────────────────────────────────────────────────────────────────────
def _common(pid, codes, roles, geom, physics, family, contrast, notes=None):
    return dict(
        id=pid, kind="coupled", codes=list(codes), dim=geom.dim,
        coords=["x", "y", "z"][:geom.dim],
        code_a_label=LABEL[codes[0]], code_b_label=LABEL[codes[1]],
        physics=physics, physics_family=family, pooled=True,
        roles={"A": roles[0], "B": roles[1]},
        domain=geom.domain_text(),
        interface=geom.interface_text(),
        interface_axis=geom.name, interface_value=str(XI),
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32",
        interface_tol="1e-6 relative",
        probe_iface=geom.iface_probe(),
        mesh_N=MESH_N, theoretical_order=2.0, tol=0.4, band=[0.8, 3.2],
        extent_a=geom.extent_a, extent_b=geom.extent_b,
        probe_M=PROBE_M[geom.dim],
        iface_graded_band=[float(v) for v in V._iface_band(0, 1)],
        iface_clearance_reason=(
            "the interface ends are Dirichlet-Neumann corners of the split "
            "problem; the recovered flux there does not converge under "
            "refinement, so grading them would fail a correct submission"),
        material_contrast=contrast,
        evidence_grade=1, evidence_grade_reason=GRADE[1],
        graded_against="hidden manufactured solution (true error) AND "
                       "reference-free mesh halving",
        notes_public=notes, flux_word="flux",
    )


def scalar_spec(pid, codes, roles, geom, family, kA_txt, kB_txt, eq, contrast,
                physics, notes=None, coefficients=None):
    s = _common(pid, codes, roles, geom, physics, family, contrast, notes)
    s.update(
        subdomain_a=f"A = {geom.subdomain_text('A')}, {kA_txt}",
        subdomain_b=f"B = {geom.subdomain_text('B')}, {kB_txt}",
        equation=eq,
        coefficients=coefficients or f"{kA_txt} in subdomain A; "
                                     f"{kB_txt} in subdomain B",
        interface_condition="the field and its normal flux",
        element="Lagrange P1 elements in both codes",
        bc_text="u = 0 on the entire outer boundary",
        iface_header=", ".join(s["coords"]) + ", u, qn",
        qn_name="qn", qn_desc="conductive flux qn = -(K grad u) . n_out",
        qoi="convergence order of the coupled field, plus the two-sided "
            "interface flux jump",
    )
    return s


def vector_spec(pid, codes, roles, geom, family, lam, muA, muB, contrast,
                physics, notes=None):
    s = _common(pid, codes, roles, geom, physics, family, contrast, notes)
    EA, nuA = C.lame_from(lam, muA)
    EB, nuB = C.lame_from(lam, muB)
    s.update(
        components=["ux", "uy"], flux_word="traction",
        subdomain_a=f"A = {geom.subdomain_text('A')}, lambda = {lam}, mu = {muA}",
        subdomain_b=f"B = {geom.subdomain_text('B')}, lambda = {lam}, mu = {muB}",
        equation="-div(sigma(u)) = f, sigma(u) = 2*mu*sym(grad(u)) + "
                 "lambda*div(u)*I   (plane strain, small strain)",
        coefficients=f"subdomain A: lambda = {lam}, mu = {muA} "
                     f"(E = {EA}, nu = {nuA}); "
                     f"subdomain B: lambda = {lam}, mu = {muB} "
                     f"(E = {EB}, nu = {nuB}). The shear modulus jumps across "
                     f"the interface; lambda is the same on both sides.",
        interface_condition="the displacement vector and the traction sigma.n "
                            "(BOTH components)",
        element="vector Lagrange P1 elements in both codes",
        bc_text="u = 0 (both components) on the entire outer boundary",
        iface_header=", ".join(s["coords"]) + ", ux, uy, tx, ty",
        qn_name="(tx, ty)", qn_desc="traction t = sigma(u) . n_out",
        qoi="convergence order of the coupled displacement, plus the two-sided "
            "interface traction jump",
    )
    return s


# ──────────────────────────────────────────────────────────────────────
# The twelve pooled instances
# ──────────────────────────────────────────────────────────────────────
#
# CONDUCTANCE RATIO, AND WHY IT IS A DESIGN VARIABLE RATHER THAN AN ACCIDENT.
#
# For a Dirichlet-Neumann interface the ratio that governs the iteration is
# rho = c_D / c_N with c = k / L per subdomain (mu / L for elasticity). The
# shipped driver is JACOBI -- within one iteration every participant reads the
# PREVIOUS iteration's exports -- and it relaxes both exchanged blocks with ONE
# scalar theta. In the error variables that is
#
#     e_lambda <- (1-theta) e_lambda - theta e_mu / c_N
#     e_mu     <- (1-theta) e_mu     - theta c_D e_lambda
#
# whose eigenvalues are (1 - theta) +/- theta sqrt(rho). The PLUS branch is
# 1 + theta (sqrt(rho) - 1), which exceeds 1 for EVERY theta in (0, 1] as soon as
# rho > 1. So with this driver an instance whose Dirichlet side carries the
# higher interface conductance cannot be made to converge by any choice of
# relaxation factor -- it is not a matter of choosing theta better.
#
# This was measured, not assumed. C9 was first built with mu 1:5 and the
# Dirichlet side on the stiff subdomain (rho = 3.6): at theta = 0.5 and at
# theta = 0.2 the driver residual oscillated between 0.39 and 1.75 for 30
# iterations and never fell, with every one of the eight exchanged blocks moving
# by more than 100% per iteration. It is the same finding D6 recorded from the
# other direction -- Kratos on the Dirichlet side at rho = 714, 60 iterations,
# residual stuck at 0.988.
#
# Every cell here therefore places the LOWER conductance on the prescribed
# Dirichlet side. The difficulty that remains is real and is spread on purpose:
# rho near 1 converges slowly (C3 and C7 at 0.7, C12 at 0.56, ~40 iterations),
# rho far below 1 converges in a handful (C8 at 0.0014). C8's role assignment is
# FORCED by this and that is the point of it.

def instance_C1(d):
    """4C + FEniCSx -- the THERMO-MECHANICAL chain the paper names this pair for.

    Steady thermoelasticity, decomposed. Each subdomain solves BOTH a heat
    equation and an elasticity equation whose stress carries a thermal term, and
    the interface transmits FOUR quantities: temperature, heat flux,
    displacement and traction.

    The construction is the composition of the two that already exist, and it
    closes for a reason worth stating. The total stress is
    ``sigma_tot = 2 mu eps(u) + lambda tr(eps) I - beta T I``. With ``beta``
    SHARED across the interface and ``T`` continuous there, the thermal part of
    the traction is continuous automatically, so traction equilibrium reduces to
    the purely elastic condition the vector construction already enforces. A
    shared ``beta`` with a jumping ``mu`` is not a fudge: ``beta = (3 lambda +
    2 mu) alpha``, so it means the two materials have different expansion
    coefficients, which is stated in the task. The mechanical body force picks up
    ``beta grad T``, which needs no continuity because it is a volume term.
    """
    kA, kB = sp.Integer(1), sp.Integer(3)
    lam, muA, muB = sp.Integer(600), sp.Integer(400), sp.Integer(1600)
    beta = sp.Integer(1)
    g = Geom(2, 0)
    KA, KB = kA * sp.eye(2), kB * sp.eye(2)
    TA, TB, fTA, fTB, co = V.straight_scalar(d, KA, KB)
    ua, ub, _, _, _ = V.vector_straight(d, lam, muA, muB)
    amp = sp.Rational(1, 1000)
    uA, uB = _scale(ua, amp), _scale(ub, amp)
    fuA = [sp.expand(v + beta * sp.diff(TA, c))
           for v, c in zip(C.elastic_body_force(uA, co, lam, muA), co)]
    fuB = [sp.expand(v + beta * sp.diff(TB, c))
           for v, c in zip(C.elastic_body_force(uB, co, lam, muB), co)]
    aA, aB = sp.nsimplify(beta / (3 * lam + 2 * muA)), \
        sp.nsimplify(beta / (3 * lam + 2 * muB))
    EA, nuA = C.lame_from(lam, muA)
    EB, nuB = C.lame_from(lam, muB)
    s = _common("C1", ["4C", "fenics"], ("dirichlet", "neumann"), g,
                "steady thermoelasticity, decomposed: temperature and "
                "displacement transmitted together",
                "thermo_mechanical", "k 1:3, mu 1:4, lambda and beta shared",
                notes=("the two transmitted fields have very different "
                       "interface stiffnesses (both follow from the "
                       "conductivities, Lame parameters and subdomain widths "
                       "above). A single scalar relaxation factor chosen for "
                       "one can diverge in the other; a per-block or "
                       "worst-block choice is safer."))
    s.update(
        components=["T", "ux", "uy"], flux_word="flux and traction",
        subdomain_a=f"A = {g.subdomain_text('A')}, k = {kA}, lambda = {lam}, "
                    f"mu = {muA}",
        subdomain_b=f"B = {g.subdomain_text('B')}, k = {kB}, lambda = {lam}, "
                    f"mu = {muB}",
        equation="in each subdomain, BOTH of\n"
                 "           -div(k grad T) = f_T\n"
                 "           -div(sigma_tot(u, T)) = (f_x, f_y), with\n"
                 "           sigma_tot = 2*mu*sym(grad(u)) + lambda*div(u)*I "
                 "- beta*T*I   (plane strain, small strain)",
        coefficients=f"subdomain A: k = {kA}, lambda = {lam}, mu = {muA} "
                     f"(E = {EA}, nu = {nuA}); subdomain B: k = {kB}, "
                     f"lambda = {lam}, mu = {muB} (E = {EB}, nu = {nuB}). "
                     f"The thermal stress coefficient is beta = {beta} in BOTH "
                     f"subdomains, so the linear expansion coefficients differ: "
                     f"alpha = beta/(3*lambda + 2*mu) = {aA} in A and {aB} in B.",
        interface_condition="the temperature, the displacement vector, the "
                            "normal heat flux and the traction sigma_tot.n "
                            "(all components)",
        element="Lagrange P1 for the temperature and vector Lagrange P1 for "
                "the displacement, in both codes",
        bc_text="T = 0 and u = 0 (both components) on the entire outer boundary",
        iface_header="x, y, T, ux, uy, qn, tx, ty",
        qn_name="(qn, tx, ty)",
        qn_desc="heat flux qn = -(k grad T) . n_out and traction "
                "t = sigma_tot(u, T) . n_out",
        qoi="convergence order of the coupled temperature and displacement, "
            "plus the two-sided interface flux and traction jumps",
    )
    return s, dict(family="thermoelastic", K=(KA, KB),
                   mat=((lam, muA), (lam, muB)), beta=beta, iface_var=x), \
        {"A": [TA] + list(uA), "B": [TB] + list(uB)}, \
        {"A": [fTA] + fuA, "B": [fTB] + fuB}, co


def instance_C2(d):
    """4C + Kratos. Severe material contrast, in the easy iteration direction,
    so the material jump and the iteration difficulty are separated."""
    kA, kB = sp.Integer(1), sp.Integer(200)
    g = Geom(2, 0)
    KA, KB = kA * sp.eye(2), kB * sp.eye(2)
    uA, uB, fA, fB, co = V.straight_scalar(d, KA, KB)
    s = scalar_spec("C2", ["4C", "kratos"], ("dirichlet", "neumann"), g,
                    "diffusion", "thermal conductivity k = 1",
                    "thermal conductivity k = 200",
                    "-div(k grad u) = f in each subdomain", "1:200",
                    "two-material conduction, severe material contrast")
    return s, dict(family="scalar", K=(KA, KB), iface_var=x), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C3(d):
    """DUNE + 4C. DIFFERENT OPERATORS either side, and a HORIZONTAL interface.

    A carries a linear reaction term and B does not, so the two subdomains
    genuinely assemble different PDEs; the reaction enters neither transmission
    condition, so the construction is untouched. The interface runs along y,
    which every shipped participant's "sample the partner by y at x = const"
    assumption gets wrong.
    """
    kA, kB = sp.Integer(1), sp.Integer(2)
    cA = sp.Integer(10)
    g = Geom(2, 1)
    KA, KB = kA * sp.eye(2), kB * sp.eye(2)
    ua, ub, fa, fb, _ = V.straight_scalar(d, KA, KB, reaction=(cA, 0))
    uA, uB = _swap(ua), _swap(ub)
    fA, fB = sp.expand(_swap(fa)), sp.expand(_swap(fb))
    s = scalar_spec("C3", ["dune", "4C"], ("dirichlet", "neumann"), g,
                    "reaction_diffusion",
                    "diffusivity k = 1 with a linear reaction coefficient c = 10",
                    "diffusivity k = 2, no reaction",
                    "subdomain A: -div(k grad u) + c*u = f ; "
                    "subdomain B: -div(k grad u) = f", "1:2",
                    "reaction-diffusion coupled to diffusion, horizontal "
                    "interface",
                    coefficients=("subdomain A: k = 1 and c = 10 ; subdomain B: "
                                  "k = 2 and no reaction term. The two "
                                  "subdomains solve DIFFERENT equations."))
    return s, dict(family="scalar", K=(KA, KB), iface_var=y,
                   reaction=(cA, sp.Integer(0))), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, (x, y)


def instance_C4(d):
    """FEniCSx + deal.II -- the TRANSIENT construction, carried over from D8.

    ``u = T(t) U(eta(x), y)`` with ``T(0) = 0``, so the initial datum is u = 0
    everywhere and discloses nothing. ``T(t) = t exp(t/2)`` is deliberately not a
    low-degree polynomial: a quadratic T would make Crank-Nicolson exact in time
    and the instance would silently stop testing the time discretisation.
    """
    kA, kB = sp.Integer(1), sp.Integer(4)
    g = Geom(2, 0)
    KA, KB = kA * sp.eye(2), kB * sp.eye(2)
    uA, uB, fA, fB, co = V.transient_two_material(d, kA, kB)
    s = scalar_spec("C4", ["fenics", "dealii"], ("dirichlet", "neumann"), g,
                    "transient_diffusion",
                    "thermal conductivity k = 1, heat capacity 1",
                    "thermal conductivity k = 4, heat capacity 1",
                    "du/dt - div(k grad u) = f in each subdomain", "1:4",
                    "transient two-material heat conduction")
    s.update(
        initial_condition="u = 0 everywhere at t = 0",
        time_grid="Crank-Nicolson, integrate to t_end = 1/4 with dt = h/4, so "
                  "dt = 1/32, 1/64, 1/128 on the three levels. Report the "
                  "solution at t = t_end.",
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32, "
                  "with dt halved alongside",
        bc_text="u = 0 on the entire outer boundary at all times",
        qoi="convergence order of the coupled field at t_end, plus the "
            "two-sided interface flux jump",
    )
    return s, dict(family="scalar", K=(KA, KB), iface_var=x, transient=True,
                   t_end=sp.Rational(1, 4)), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


# C5's two legs are graded on different bands, so each needs its own index
# list from the shared authority.
_C5_L1 = V.iface_probe_indices(2, (0.125, 0.375))[1]
_C5_L2 = V.iface_probe_indices(2, (0.625, 0.875))[1]


def instance_C5(d):
    """scikit-fem + FEniCSx -- the NOTCHED construction, carried over from D5.

    The global domain is the unit square minus a corner square, so it is
    non-convex; subdomain A is therefore not a rectangle either; the interface is
    a bent polyline whose two legs have different outward normals AND different
    material contrasts; and four material cells meet at a cross point. Every
    material line and the notch lie on mesh lines at every prescribed level.

    A is the non-rectangular subdomain and takes the DIRICHLET role, as in the
    walked original: the two legs carry opposite conductance ratios, and the
    largest of them fixes the admissible relaxation factor.
    """
    half, q3 = sp.Rational(1, 2), sp.Rational(3, 4)
    k1x, k2x = sp.Integer(1), sp.Rational(5, 2)
    k1y, k2y = sp.Integer(1), sp.Integer(2)
    fields, sources, co, pm = V.checkerboard_notched(
        d, k1x, k2x, k1y, k2y, half, half, q3, q3)
    M = PROBE_M[2]
    # The grader evaluates ONE expression per subdomain. Subdomain A spans three
    # material cells, so its exact field is a Piecewise over them; the probe grid
    # (i+0.5)/44 never lands on x = 1/2 or y = 1/2, so the strict inequalities
    # are unambiguous at every graded point. Writing the four cells and no A/B
    # entry -- which the original did -- makes the key unusable: grade_run reads
    # key["exact_solution"]["A"] and raises KeyError before any comparison.
    exA = sp.Piecewise((fields[(0, 0)], sp.And(x < half, y < half)),
                       (fields[(0, 1)], x < half),
                       (fields[(1, 1)], True))
    s = dict(
        id="C5", kind="coupled", codes=["skfem", "fenics"], dim=2,
        coords=["x", "y"], pooled=True,
        code_a_label=LABEL["skfem"], code_b_label=LABEL["fenics"],
        physics="four-material conduction on a notched domain, bent interface",
        physics_family="diffusion",
        roles={"A": "dirichlet", "B": "neumann"},
        domain="the unit square (0,1) x (0,1) with the square (3/4, 1) x "
               "(3/4, 1) REMOVED (an L-shaped, non-convex domain)",
        subdomain_a="A = the domain minus subdomain B, i.e. everything except "
                    "(1/2, 1) x (0, 1/2). A is NOT a rectangle",
        subdomain_b="B = (1/2, 1) x (0, 1/2)",
        interface="the bent polyline consisting of the segment x = 1/2, "
                  "0 < y < 1/2 and the segment y = 1/2, 1/2 < x < 1. Its two "
                  "legs have different outward normals AND different material "
                  "contrasts",
        # A BENT interface cannot be described by one axis; grader v2's
        # structured interface machinery takes `iface_legs` — one
        # {axis, value, band} per leg, matching the INTERFACE PROBE POINTS the
        # task text states. Leg 1 runs along y at x = 1/2, graded on
        # y in (1/8, 3/8); leg 2 along x at y = 1/2, graded on x in (5/8, 7/8).
        interface_axis="polyline", interface_value="1/2",
        iface_legs=[{"axis": 0, "value": 0.5, "band": [0.125, 0.375]},
                    {"axis": 1, "value": 0.5, "band": [0.625, 0.875]}],
        equation="-div(k grad u) = f, with k piecewise constant",
        coefficients="k = 1 on (0,1/2)x(0,1/2); k = 5/2 on (1/2,1)x(0,1/2); "
                     "k = 5 on (1/2,1)x(1/2,1); k = 2 on (0,1/2)x(1/2,1). "
                     "The contrast is 1:5/2 across the vertical leg of the "
                     "interface and 1:2 across the horizontal leg.",
        interface_condition="the field and its normal flux",
        element="Lagrange P1 elements in both codes",
        mesh_text="uniform meshes, h halved per level: h = 1/8, 1/16, 1/32 "
                  "(every material line and the notch lie on mesh lines at "
                  "every level)",
        bc_text="u = 0 on the entire outer boundary, INCLUDING the two faces "
                "of the removed square (x = 3/4 for y > 3/4, and y = 3/4 for "
                "x > 3/4)",
        interface_tol="1e-6 relative", flux_word="flux",
        iface_header="x, y, u, qn",
        qn_name="qn", qn_desc="conductive flux qn = -(k grad u) . n_out",
        # FOURTH COPY of the interface-probe text, for C5's BENT interface.
        # Routed through V.iface_probe_indices like the other live ones, so the
        # coordinates are solution-probe coordinates and the fabricated-flux
        # cross-check can run on this cell too. Each leg carries its own band.
        probe_iface=(
            f"{len(_C5_L1)} points on each leg, covering the INTERIOR of that "
            f"leg only. Leg 1 is (x, y) = (1/2, (j+0.5)/{M}) for "
            f"j = {_C5_L1[0]}, {_C5_L1[0] + 1}, ..., {_C5_L1[-1]}, and leg 2 "
            f"is (x, y) = ((j+0.5)/{M}, 1/2) for j = {_C5_L2[0]}, "
            f"{_C5_L2[0] + 1}, ..., {_C5_L2[-1]}. Write leg 1 first, then leg "
            f"2, {len(_C5_L1) + len(_C5_L2)} rows in total. The ends of each "
            f"leg are excluded: one end of each meets the outer boundary and "
            f"the other meets the corner where the two legs join, and the "
            f"recovered flux does not converge at either. These are exactly "
            f"the solution probe coordinates lying in each leg's graded band, "
            f"so the same values appear in your solution_level<k>_<side>.csv "
            f"rows"),
        mesh_N=MESH_N, theoretical_order=2.0, tol=0.4, band=[0.8, 3.2],
        extent_a=[(0.0, 1.0), (0.0, 1.0)],
        probe_a_exclude=[[(0.5, 1.0), (0.0, 0.5)], [(0.75, 1.0), (0.75, 1.0)]],
        extent_b=[(0.5, 1.0), (0.0, 0.5)],
        iface_graded_band=[0.125, 0.375],
        iface_clearance_reason=(
            "each leg is graded on its interior: one end meets the outer "
            "boundary and the other the corner where the legs join, and the "
            "recovered flux converges at neither"),
        probe_M=M, material_contrast="1:5/2 on leg 1, 1:2 on leg 2",
        evidence_grade=1, evidence_grade_reason=GRADE[1],
        qoi="convergence order of the coupled field, plus the two-sided "
            "interface flux jump on both legs",
        graded_against="hidden manufactured solution (true error) AND "
                       "reference-free mesh halving",
        arrangement="notched (non-convex) domain, non-rectangular subdomain, "
                    "bent interface with two normals and two contrasts, four "
                    "material cells meeting at a cross point",
        notes_public=("the two legs of the interface carry OPPOSITE conductance "
                      "ratios, so a single global relaxation factor must satisfy "
                      "the stricter of the two; the values follow from the "
                      "conductivities and subdomain widths above."),
    )
    kv = {(0, 0): sp.Integer(1), (1, 0): sp.Rational(5, 2),
          (1, 1): sp.Integer(5), (0, 1): sp.Integer(2)}
    return s, dict(family="notched", cells=fields, kv=kv, exA=exA,
                   exB=fields[(1, 0)], iface_var=None,
                   avoid_lines=(0.5,)), \
        {"A": exA, "B": fields[(1, 0)]}, sources, co


def instance_C6(d):
    """deal.II + NGSolve -- the CONJUGATE HEAT TRANSFER instance.

    Two conducting media meeting at an interface, with temperature and normal
    heat flux transmitted: the conjugate-heat-transfer transmission conditions
    exactly. The conductivity is a full tensor on both sides and both diagonal
    entries jump. The off-diagonal entry that couples the normal to the
    tangential direction must be SHARED for the normal flux to be continuous
    under the flux potential -- a real constraint on the family, checked
    symbolically rather than assumed.
    """
    KA = sp.Matrix([[1, sp.Rational(1, 2)], [sp.Rational(1, 2), 2]])
    KB = sp.Matrix([[6, sp.Rational(1, 2)], [sp.Rational(1, 2), 3]])
    g = Geom(2, 0)
    uA, uB, fA, fB, co = V.straight_scalar(d, KA, KB)
    s = scalar_spec("C6", ["dealii", "ngsolve"], ("dirichlet", "neumann"), g,
                    "conjugate_heat_transfer",
                    "anisotropic conductivity K = [[1, 1/2], [1/2, 2]]",
                    "anisotropic conductivity K = [[6, 1/2], [1/2, 3]]",
                    "-div(K grad T) = f in each subdomain", "K11 1:6, K22 2:3",
                    "conjugate heat transfer between two conducting media, "
                    "anisotropic tensor jump")
    return s, dict(family="scalar", K=(KA, KB), iface_var=x), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C7(d):
    """FEBio + deal.II. Elasticity, mild contrast: FEBio is the constraint.

    FEBio 4 has no heat module, so a FEBio cell is necessarily mechanical, and
    FEBio has no small-strain material either -- it solves a finite-deformation
    problem with a Newton iteration. The strains here are of order 1e-5, so the
    geometric terms are ~1e-10 relative and the discretisation error dominates by
    seven orders; the contrast is kept mild so the outer Newton loop is not the
    thing under test.
    """
    # WHY NOT A MILDER CONTRAST. It was mu 1:2 first, and that is the regime the
    # shipped Jacobi driver cannot serve in practice: the measured per-iteration
    # contraction was 0.97, so the residual fell from 8.7e-01 to 4.6e-04 in 200
    # iterations and needed roughly 500 more. The driver applies ONE theta to
    # both exchanged blocks, and its two eigenvalues are (1-theta) +/-
    # theta*sqrt(rho); with rho near 1 the best achievable contraction IS
    # sqrt(rho), whatever theta is chosen, so no relaxation setting rescues it.
    # A contrast further from balanced is a property of the PROBLEM and costs
    # nothing in discrimination -- the transmission conditions are just as
    # binding at 1:5 as at 1:2.
    lam, muA, muB = sp.Integer(500), sp.Integer(250), sp.Integer(1250)
    g = Geom(2, 0)
    ua, ub, fa, fb, co = V.vector_straight(d, lam, muA, muB)
    uA, uB = _scale(ua, U_AMP), _scale(ub, U_AMP)
    fA, fB = _scale(fa, U_AMP), _scale(fb, U_AMP)
    s = vector_spec("C7", ["febio", "dealii"], ("dirichlet", "neumann"), g,
                    "elasticity", lam, muA, muB, "mu 1:5, lambda shared",
                    "two-material linear elasticity, shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C8(d):
    """NGSolve + Kratos -- the 1:1000 contrast, carried over from D6.

    ITS ROLE ASSIGNMENT IS FORCED AND THAT WAS MEASURED. With Kratos on the
    Dirichlet side -- the only role its shipped participant supports -- the
    conductance ratio is rho = 714 and the iteration does not converge: executed
    on D6, 60 iterations at the theoretically optimal theta = 0.0014, residual
    stuck at 0.988. So this cell cannot be served by the shipped Kratos
    participant at all; it needs a Neumann-side one, and the task says which role
    Kratos has so that fact is a stated requirement rather than a trap.
    """
    kA, kB = sp.Integer(1), sp.Integer(1000)
    g = Geom(2, 0)
    KA, KB = kA * sp.eye(2), kB * sp.eye(2)
    uA, uB, fA, fB, co = V.straight_scalar(d, KA, KB)
    s = scalar_spec("C8", ["ngsolve", "kratos"], ("dirichlet", "neumann"), g,
                    "diffusion", "thermal conductivity k = 1",
                    "thermal conductivity k = 1000",
                    "-div(k grad u) = f in each subdomain", "1:1000",
                    "two-material conduction, severe contrast",
                    notes=("The conductance ratio is large; a fixed relaxation "
                           "factor that works at unit contrast will not "
                           "converge here."))
    return s, dict(family="scalar", K=(KA, KB), iface_var=x), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C9(d):
    """NGSolve + scikit-fem. Elasticity with a HORIZONTAL interface.

    The elasticity operator is isotropic, so the reflection x <-> y maps a
    solution of the vertical-interface problem to a solution of this one with the
    displacement components exchanged. Nothing in the construction is weakened;
    what changes is that the interface normal is e_y, the normal and shear
    traction components swap roles, and every "sample by y at x = const"
    assumption fails. A is the NEUMANN side here.
    """
    # A is the NEUMANN side here, so A carries the STIFF material: the Jacobi
    # driver diverges for every theta when the Dirichlet side is the stiffer one
    # (measured on this very cell before the moduli were exchanged -- see the
    # conductance-ratio note above).
    lam, muA, muB = sp.Integer(480), sp.Integer(1200), sp.Integer(240)
    g = Geom(2, 1)
    ua, ub, fa, fb, _ = V.vector_straight(d, lam, muA, muB)
    uA = _scale((_swap(ua[1]), _swap(ua[0])), U_AMP)
    uB = _scale((_swap(ub[1]), _swap(ub[0])), U_AMP)
    fA = _scale([_swap(fa[1]), _swap(fa[0])], U_AMP)
    fB = _scale([_swap(fb[1]), _swap(fb[0])], U_AMP)
    s = vector_spec("C9", ["ngsolve", "skfem"], ("neumann", "dirichlet"), g,
                    "elasticity", lam, muA, muB, "mu 5:1, lambda shared",
                    "two-material linear elasticity, horizontal interface")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=y), {"A": uA, "B": uB}, {"A": fA, "B": fB}, (x, y)


def instance_C10(d):
    """Kratos + DUNE. The 3-D cell.

    The interface is a PLANE, so the partner must be sampled on a
    two-dimensional set: the 1-D interpolation every shipped participant uses
    does not apply, and this is the only instance in the set where that is true.
    """
    kA, kB = sp.Integer(1), sp.Integer(4)
    g = Geom(3, 0)
    KA, KB = kA * sp.eye(3), kB * sp.eye(3)
    uA, uB, fA, fB, co = V.straight_scalar(d, KA, KB, dim=3)
    s = scalar_spec("C10", ["kratos", "dune"], ("dirichlet", "neumann"), g,
                    "diffusion", "thermal conductivity k = 1",
                    "thermal conductivity k = 4",
                    "-div(k grad u) = f in each subdomain", "1:4",
                    "two-material conduction in 3-D, planar interface")
    return s, dict(family="scalar", K=(KA, KB), iface_var=x), \
        {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C11(d):
    """FEBio + scikit-fem. Elasticity."""
    lam, muA, muB = sp.Integer(450), sp.Integer(225), sp.Integer(900)
    g = Geom(2, 0)
    ua, ub, fa, fb, co = V.vector_straight(d, lam, muA, muB)
    uA, uB = _scale(ua, U_AMP), _scale(ub, U_AMP)
    fA, fB = _scale(fa, U_AMP), _scale(fb, U_AMP)
    s = vector_spec("C11", ["febio", "skfem"], ("dirichlet", "neumann"), g,
                    "elasticity", lam, muA, muB, "mu 1:4, lambda shared",
                    "two-material linear elasticity, shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


def instance_C12(d):
    """DUNE + FEBio. Elasticity, and the reason a DUNE vector participant exists.

    Neither code in this pair ships a participant that can serve it: FEBio has no
    heat module so the cell cannot be scalar, and DUNE ships only a scalar one.
    The DUNE elasticity participant written for this cell is in walkers/w_dune.py.
    """
    lam, muA, muB = sp.Integer(400), sp.Integer(200), sp.Integer(500)
    g = Geom(2, 0)
    ua, ub, fa, fb, co = V.vector_straight(d, lam, muA, muB)
    uA, uB = _scale(ua, U_AMP), _scale(ub, U_AMP)
    fA, fB = _scale(fa, U_AMP), _scale(fb, U_AMP)
    s = vector_spec("C12", ["dune", "febio"], ("dirichlet", "neumann"), g,
                    "elasticity", lam, muA, muB, "mu 1:5/2, lambda shared",
                    "two-material linear elasticity, shear-modulus jump")
    return s, dict(family="vector", mat=((lam, muA), (lam, muB)),
                   iface_var=x), {"A": uA, "B": uB}, {"A": fA, "B": fB}, co


BUILDERS = [instance_C1, instance_C2, instance_C3, instance_C4, instance_C5,
            instance_C6, instance_C7, instance_C8, instance_C9, instance_C10,
            instance_C11, instance_C12]


# ──────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────
def _boxes(spec):
    out = []
    for ext in (spec["extent_a"], spec["extent_b"]):
        box = []
        for lo, hi in ext:
            pad = 0.05 * (hi - lo)
            box.append((lo + pad, hi - pad))
        out.append(box)
    return out


def numeric_vector_jump(uA, uB, coords, matA, matB, iface_var, iface_val,
                        n=60, h=1e-4, extra_normal=None):
    """Displacement and traction jumps by finite differences, either axis.

    ``build_coupled_v2.numeric_vector_jump`` hard-codes the interface at
    ``x = iv`` and varies ``y``; the horizontal-interface instance needs the
    transpose, and a check that silently evaluates on the wrong line reports a
    clean zero for a broken construction. ``extra_normal`` adds a term to the
    traction (the thermoelastic ``-beta T I``) so the check is run on the stress
    the problem actually transmits.
    """
    import numpy as np
    rng = np.random.default_rng(23)
    idx = list(coords).index(iface_var)
    sA = C.cauchy_stress(uA, coords, *matA)
    sB = C.cauchy_stress(uB, coords, *matB)
    if extra_normal is not None:
        eA, eB = extra_normal
        sA = sA + eA * sp.eye(len(coords))
        sB = sB + eB * sp.eye(len(coords))
    fu = [sp.lambdify(coords, c, "math") for c in uA]
    gu = [sp.lambdify(coords, c, "math") for c in uB]
    fa = [sp.lambdify(coords, sA[i, idx], "math") for i in range(len(coords))]
    fb = [sp.lambdify(coords, sB[i, idx], "math") for i in range(len(coords))]
    du, su, dq, sq = [], [], [], []
    for _ in range(n):
        pt = [float(iface_val) if m == idx else float(rng.uniform(0.05, 0.95))
              for m in range(len(coords))]
        for a, b in zip(fu, gu):
            va, vb = a(*pt), b(*pt)
            du.append(abs(va - vb))
            su.append(abs(va))
        for a, b in zip(fa, fb):
            va, vb = a(*pt), b(*pt)
            dq.append(abs(va - vb))
            sq.append(abs(va))
    return (max(du) / max(max(su), 1e-12), max(dq) / max(max(sq), 1e-12))


def _fmt_source(spec, sources):
    """The published source term, per subdomain and per component."""
    coords = ", ".join(spec["coords"])
    if isinstance(sources, dict) and all(isinstance(k, tuple) for k in sources):
        names = {(0, 0): "on (0,1/2)x(0,1/2)", (1, 0): "on (1/2,1)x(0,1/2)",
                 (1, 1): "on (1/2,1)x(1/2,1)", (0, 1): "on (0,1/2)x(1/2,1)"}
        return f"f({coords}) =\n" + "\n".join(
            f"               {names[k]}: {sp.sstr(sources[k])}"
            for k in sorted(sources))
    comps = spec.get("components")
    if comps and len(comps) > 1:
        names = {"T": "f_T", "ux": "f_x", "uy": "f_y", "uz": "f_z"}
        out = []
        for side in ("A", "B"):
            for c, e in zip(comps, sources[side]):
                out.append(f"               {names.get(c, 'f_' + c)} = "
                           f"{sp.sstr(e)}   (subdomain {side})")
        return f"f({coords}) =\n" + "\n".join(out)
    return (f"f({coords}) = in subdomain A: {sp.sstr(sources['A'])}\n"
            f"               in subdomain B: {sp.sstr(sources['B'])}")


def _outer_trace_check(spec, fields, coords, iface_var, skip=False):
    if skip:
        return C.Check("outer_boundary_trace_zero", "PASS",
                       "checked cell-by-cell in the notched construction")
    bad = []
    for side, ext in (("A", spec["extent_a"]), ("B", spec["extent_b"])):
        for i, cvar in enumerate(coords):
            for val in ext[i]:
                if cvar is iface_var and abs(val - float(XI)) < 1e-12:
                    continue                    # the interface, not the outside
                comps = fields[side] if isinstance(fields[side], (list, tuple)) \
                    else [fields[side]]
                for comp in comps:
                    e = sp.sympify(comp).subs(cvar, sp.nsimplify(val))
                    if sp.simplify(e.subs(t, sp.Symbol("t"))) != 0:
                        bad.append(f"{side}:{cvar}={val}")
    return C.Check("outer_boundary_trace_zero", "PASS" if not bad else "FAIL",
                   "; ".join(sorted(set(bad))) or "all outer traces vanish")


def build_one(fn, seed=None):
    d = V.Draw(seed)
    spec, info, fields, sources, coords = fn(d)
    spec["draw_seed"] = d.seed
    checks, numeric = [], {}
    iface_var = info.get("iface_var")
    fam = info["family"]
    gate_exact = None
    boxes = _boxes(spec)

    if fam == "vector":
        matA, matB = info["mat"]
        for side, mat in (("A", matA), ("B", matB)):
            s = C.cauchy_stress(fields[side], coords, *mat)
            for i in range(len(coords)):
                r = sp.simplify(-sum(sp.diff(s[i, j], coords[j])
                                     for j in range(len(coords)))
                                - sources[side][i])
                checks.append(C.Check(f"strong_form_{side}{i}",
                                      "PASS" if r == 0 else "FAIL", sp.sstr(r)))
        rep = C.check_vector_transmission(fields["A"], fields["B"], coords,
                                          matA, matB, iface_var, XI)
        numeric["jump_u"], numeric["jump_traction"] = numeric_vector_jump(
            fields["A"], fields["B"], coords, matA, matB, iface_var, XI)
        exact = V._exact_strings(fields)
        src = {k: [sp.sstr(c) for c in v] for k, v in sources.items()}
        checks.append(_outer_trace_check(spec, fields, coords, iface_var))

    elif fam == "thermoelastic":
        KA, KB = info["K"]
        matA, matB = info["mat"]
        beta = info["beta"]
        for i, (side, K, mat) in enumerate((("A", KA, matA), ("B", KB, matB))):
            T = fields[side][0]
            u = fields[side][1:]
            r = sp.simplify(C.poisson_source(T, coords, K) - sources[side][0])
            checks.append(C.Check(f"strong_form_T_{side}",
                                  "PASS" if r == 0 else "FAIL", sp.sstr(r)))
            s = C.cauchy_stress(u, coords, *mat) - beta * T * sp.eye(len(coords))
            for k in range(len(coords)):
                r = sp.simplify(-sum(sp.diff(s[k, j], coords[j])
                                     for j in range(len(coords)))
                                - sources[side][1 + k])
                checks.append(C.Check(f"strong_form_u{k}_{side}",
                                      "PASS" if r == 0 else "FAIL", sp.sstr(r)))
            numeric[f"residual_T_{side}"] = V.numeric_residual(
                T, sources[side][0], coords, K, box=boxes[i])
        repT = C.check_scalar_transmission(fields["A"][0], fields["B"][0],
                                           KA, KB, coords, iface_var, XI)
        repU = C.check_vector_transmission(fields["A"][1:], fields["B"][1:],
                                           coords, matA, matB, iface_var, XI)
        numeric["jump_T"], numeric["jump_q"] = V.numeric_interface_jump(
            fields["A"][0], fields["B"][0], KA, KB, coords, iface_var, XI)
        numeric["jump_u"], numeric["jump_traction"] = numeric_vector_jump(
            fields["A"][1:], fields["B"][1:], coords, matA, matB, iface_var, XI,
            extra_normal=(-beta * fields["A"][0], -beta * fields["B"][0]))

        class _R:
            ok = repT.ok and repU.ok

            @staticmethod
            def summary():
                return f"T: {repT.summary()} | u: {repU.summary()}"
        rep = _R()
        exact = V._exact_strings(fields)
        src = {k: [sp.sstr(c) for c in v] for k, v in sources.items()}
        checks.append(_outer_trace_check(spec, fields, coords, iface_var))

    elif fam == "notched":
        cells, kv = info["cells"], info["kv"]
        for key, u in cells.items():
            K = kv[key] * sp.eye(2)
            checks.append(C.verify_strong_form(u, sources[key], coords, K))
            numeric[f"residual_cell{key[0]}{key[1]}"] = V.numeric_residual(
                u, sources[key], coords, K)
        half = sp.Rational(1, 2)
        legs = [("leg1_x", cells[(0, 0)], cells[(1, 0)], kv[(0, 0)] * sp.eye(2),
                 kv[(1, 0)] * sp.eye(2), x, half),
                ("leg2_y", cells[(1, 0)], cells[(1, 1)], kv[(1, 0)] * sp.eye(2),
                 kv[(1, 1)] * sp.eye(2), y, half)]
        reports = []
        for name, ua, ub, Ka, Kb, var, val in legs:
            r = C.check_scalar_transmission(ua, ub, Ka, Kb, coords, var, val)
            reports.append((name, r))
            ju, jq = V.numeric_interface_jump(ua, ub, Ka, Kb, coords, var, val)
            numeric[f"{name}_jump_u"], numeric[f"{name}_jump_q"] = ju, jq

        class _R:
            ok = all(r.ok for _, r in reports)

            @staticmethod
            def summary():
                return "; ".join(f"{n}: {r.summary()}" for n, r in reports)
        rep = _R()
        # The key must carry ONE expression per SUBDOMAIN, not per material cell.
        exact = {"A": sp.sstr(info["exA"]), "B": sp.sstr(info["exB"])}
        src = {f"cell{k[0]}{k[1]}": sp.sstr(v) for k, v in sources.items()}
        # The leak gate gets the four CELL polynomials, not the Piecewise that
        # packages three of them. They are the same solution, and the gate's
        # symbolic comparisons (simplify, cancel, is_proportional) do not
        # terminate in reasonable time against a Piecewise -- a gate that has to
        # be skipped for slowness is a gate that is not run.
        gate_exact = {f"cell{k[0]}{k[1]}": sp.sstr(v)
                      for k, v in info["cells"].items()}
        checks.append(C.Check("outer_boundary_trace_zero", "PASS",
                              "the potential carries (eta - eta_n)(zeta - "
                              "zeta_n) and vanishes on every outer face and "
                              "both notch faces by construction"))

    else:                                                          # scalar
        KA, KB = info["K"]
        react = info.get("reaction", (sp.Integer(0), sp.Integer(0)))
        transient = info.get("transient", False)
        for i, (side, K, c) in enumerate((("A", KA, react[0]),
                                          ("B", KB, react[1]))):
            lhs = C.poisson_source(fields[side], coords, K)
            if transient:
                lhs = lhs + sp.diff(fields[side], t)
            elif c:
                lhs = lhs + c * fields[side]
            r = sp.simplify(lhs - sources[side])
            checks.append(C.Check(f"strong_form_{side}",
                                  "PASS" if r == 0 else "FAIL", sp.sstr(r)))
            box = boxes[i] + ([(0.02, 0.25)] if transient else [])
            numeric[f"residual_{side}"] = V.numeric_residual(
                fields[side], sources[side], coords, K, box=box,
                reaction=(0 if transient else c), transient=transient)
        rep = C.check_scalar_transmission(fields["A"], fields["B"], KA, KB,
                                          coords, iface_var, XI)
        numeric["jump_u"], numeric["jump_q"] = V.numeric_interface_jump(
            fields["A"], fields["B"], KA, KB, coords, iface_var, XI,
            transient=transient)
        if transient:
            # THE KEY MUST HOLD THE FIELD AT t_end, NOT THE SPACE-TIME FIELD.
            #
            # The task asks for the solution at t = t_end and the grader
            # lambdifies key["exact_solution"] over the SPATIAL coordinates
            # only. A stored expression still carrying `t` produces a function
            # with an unbound global: rms_error catches the NameError and
            # returns None, and grade_run reports INVALID_SUBMISSION, "error not
            # computable" -- before any comparison against truth. A correct
            # submission cannot pass. The transient instance in the old set
            # (D8) has exactly this defect; it is fixed here rather than
            # inherited.
            te = sp.nsimplify(info["t_end"])
            fields = {k: sp.expand(sp.sympify(v).subs(t, te))
                      for k, v in fields.items()}
        exact = V._exact_strings(fields)
        src = {k: sp.sstr(v) for k, v in sources.items()}
        checks.append(_outer_trace_check(spec, fields, coords, iface_var))

    gate_exact = gate_exact or exact
    spec["source_public"] = src          # public: it is printed in task.txt
    task = build_task(spec, _fmt_source(spec, sources))
    gate = scan(task, {"exact_solution": gate_exact, "source_term": src},
                spec["id"])

    sym_ok = all(c.ok for c in checks)
    num_ok = all(v < 1e-6 for v in numeric.values())
    return dict(spec=spec, task=task, exact=exact, source=src, checks=checks,
                numeric=numeric, transmission=rep.summary(), gate=gate,
                ok=(sym_ok and rep.ok and num_ok and gate.clean),
                sym_ok=sym_ok, trans_ok=rep.ok, num_ok=num_ok)


# ──────────────────────────────────────────────────────────────────────
def verify_balance(specs) -> list:
    """Re-derive the balance claims from the built specs. Never assert them."""
    from collections import Counter
    pairs, roles = Counter(), {}
    for s in specs:
        a, b = s["codes"]
        pairs[a] += 1
        pairs[b] += 1
        for side, code in (("A", a), ("B", b)):
            roles.setdefault(code, Counter())[s["roles"][side]] += 1
    bad = []
    for code, n in sorted(pairs.items()):
        if n != 3:
            bad.append(f"{code} appears in {n} pairs, not 3")
    if len(pairs) != 8:
        bad.append(f"{len(pairs)} backends appear, not 8: {sorted(pairs)}")
    tot = Counter()
    for code, rc in roles.items():
        tot += rc
        nd, nn = rc.get("dirichlet", 0), rc.get("neumann", 0)
        if {nd, nn} != {1, 2}:
            bad.append(f"{code} is {nd} Dirichlet / {nn} Neumann, not 2/1 or 1/2")
    if tot.get("dirichlet") != 12 or tot.get("neumann") != 12:
        bad.append(f"role split is {dict(tot)}, not 12/12")
    for code in ("fenics", "dealii"):
        if roles.get(code, {}).get("neumann", 0) != 2:
            bad.append(f"{code} must take the Neumann role twice")
    for code in ("4C", "dune", "febio"):
        if roles.get(code, {}).get("neumann", 0) > 1:
            bad.append(f"{code} must take the Neumann role at most once")
    for s in specs:
        if not s.get("iface_graded_band"):
            bad.append(f"{s['id']} has no interface-end exclusion band")
        if not s.get("material_contrast"):
            bad.append(f"{s['id']} declares no material contrast")
        if "NDOF = <integer>" not in s.get("_task", "NDOF = <integer>"):
            bad.append(f"{s['id']} task text carries no NDOF contract line")
    fams = {s["physics_family"] for s in specs}
    for need in ("thermo_mechanical", "conjugate_heat_transfer"):
        if need not in fams:
            bad.append(f"no instance carries the {need} family")
    nvec = sum(1 for s in specs if s.get("components") == ["ux", "uy"])
    if nvec < 3:
        bad.append(f"only {nvec} elasticity pairs, fewer than 3")
    return bad


def write_key(kdir: Path, s: dict, r: dict):
    kdir.mkdir(parents=True, exist_ok=True)
    (kdir / "key.json").write_text(json.dumps({
        "id": s["id"], "kind": "coupled", "codes": s["codes"],
        "dim": s["dim"], "coords": s["coords"],
        "components": s.get("components", ["u"]),
        "exact_solution": r["exact"], "source_term": r["source"],
        "extent_a": s["extent_a"], "extent_b": s["extent_b"],
        "probe_M": s["probe_M"], "mesh_N": s["mesh_N"],
        "theoretical_order": s["theoretical_order"], "tol": s["tol"],
        "band": s["band"], "evidence_grade": s["evidence_grade"],
        # Without this the grader's magnitude check silently degrades to
        # rate-only, and a rate can be manufactured from the source term with no
        # solver involved. It is a property of the answer, so it belongs in the
        # sealed key and nowhere else.
        "exact_rms": V._exact_rms_on_probe_grid(r, s),
        "evidence_grade_reason": s["evidence_grade_reason"],
        "material_contrast": s["material_contrast"],
        "roles": s["roles"], "physics_family": s["physics_family"],
        "pooled": s.get("pooled", True),
        "draw_seed": s["draw_seed"],
        "verification": {
            "symbolic": [f"{c.name}={c.verdict}" for c in r["checks"]],
            "transmission": r["transmission"],
            "numeric_relative": r["numeric"],
        },
        "qoi": s["qoi"], "graded_against": s["graded_against"],
    }, indent=2, default=str), encoding="utf-8")


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
    args = ap.parse_args()

    built, failed = [], []
    for fn in BUILDERS:
        pid = fn.__name__.split("_")[-1]
        if args.only and pid not in args.only:
            continue
        r = build_one(fn, args.seed)
        s = r["spec"]
        s["_task"] = r["task"]
        worst = max(r["numeric"].values()) if r["numeric"] else 0.0
        print(f"[{s['id']}] {'+'.join(s['codes'])}  "
              f"A={s['roles']['A'][:4]} B={s['roles']['B'][:4]}  "
              f"{s['physics_family']}: {s['physics']}")
        print(f"      symbolic={'OK' if r['sym_ok'] else 'FAIL'}  "
              f"numeric_worst_rel={worst:.2e}  ndof_clause="
              f"{'yes' if 'NDOF = <integer>' in r['task'] else 'NO'}")
        print(f"      transmission={r['transmission']}")
        print(f"      leak gate={'CLEAN' if r['gate'].clean else 'LEAK'} "
              f"{[f.rule for f in r['gate'].findings] or ''}")
        for f in r["gate"].findings:
            if f.severity in ("CRITICAL", "HIGH"):
                print(f"        [{f.severity}] {f.detail[:160]}")
        for c in r["checks"]:
            if not c.ok:
                print(f"        CHECK {c.name}={c.verdict}: {c.detail[:160]}")
        (built if r["ok"] else failed).append(r)

    print(f"\n{len(built)}/{len(built) + len(failed)} instances built, "
          f"verified and blind.")
    if failed:
        print("FAILED:", [r["spec"]["id"] for r in failed])
        return 1

    if not args.only:
        bad = verify_balance([r["spec"] for r in built])
        print("balance:", "OK (8 backends x 3 pairs, 12 Dirichlet / 12 Neumann)"
              if not bad else "BROKEN")
        for b in bad:
            print("   -", b)
        if bad:
            return 1

    if not args.apply:
        print("(dry run -- pass --apply to write tasks and keys)")
        return 0

    problems = Path(args.problems_root) if args.problems_root else HERE / "problems"
    keys = Path(args.keys_root) if args.keys_root else Path(os.environ.get(
        "OPENPASO_BLIND_KEYS",
        "/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys"))

    # A REBUILD MUST NOT DESTROY A SPENT INSTANCE OR ITS ANSWER KEY.
    #
    # This wrote problems/<id> and keys/<id> with a FIXED id list, so running
    # it again to draw the evaluation set would have overwritten the
    # development tasks AND their sealed keys in place — the graded rounds
    # would lose the very answers they were graded against, and the new
    # instances would carry ids the phase gate already lists as spent.
    # Refuse, and name the alternative.
    clash = [r["spec"]["id"] for r in built
             if (problems / r["spec"]["id"]).exists()
             or (keys / r["spec"]["id"]).exists()]
    if clash and not args.overwrite_spent:
        print(f"REFUSING: {len(clash)} instance(s) already exist and would be "
              f"overwritten: {clash[:8]}{' ...' if len(clash) > 8 else ''}")
        print("  A drawn instance is evidence: its task, its key and the runs "
              "graded against it belong together.")
        print("  For a NEW draw use --problems-root/--keys-root pointing at a "
              "fresh directory (evaluation instances live apart from "
              "development ones).")
        print("  To genuinely replace these, pass --overwrite-spent and be "
              "sure the keys are backed up.")
        return 1

    for r in built:
        s = r["spec"]
        s.pop("_task", None)
        pdir = problems / s["id"]
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "task.txt").write_text(r["task"], encoding="utf-8")
        public = {k: v for k, v in s.items() if k != "draw_seed"}
        (pdir / "spec_public.json").write_text(
            json.dumps(public, indent=2, default=str))
        write_key(keys / s["id"], s, r)
        print(f"wrote problems/{s['id']}/ and keys/{s['id']}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
