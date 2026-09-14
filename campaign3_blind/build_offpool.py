#!/usr/bin/env python
"""The two OFF-POOL coupled cells: C13 (FEM-DSMC, grade 3) and C14 (FSI, grade 2).

Why these sit outside the pooled twelve
---------------------------------------
The pooled set's arithmetic is exact -- 12 pairs x 2 slots = 24 = 8 backends x 3
appearances -- and its evidence grade is uniformly 1 (exact manufactured
solution). These two cells cannot be grade 1: a DSMC gas has no closed-form
coupled solution, and the FSI problem's standard is an independent re-solve, not
an exact field. DESIGN.md Amendment 2 s4 forbids pooling grades, and grader v2
enforces it by TYPE (BandOnlyResult and ReferenceResult have no observed_order
attribute to average). So C13 is graded band-only (grade 3) and C14 against a
reference (grade 2), each stamped `pooled: false`.

Both are generated here, never hand-written, for the same reason as the twelve:
the task text and the grader read one contract. The grading modes and key
schemas are grader v2's (`grading/loading.py`, `grading/outcomes.py`):

  grade 3:  "grading": "band-only",
            "qoi": {"result_line", "band": [lo, hi]},
            "identity": {"description", "lhs_line", "rhs_line", "rtol"}
  grade 2:  "grading": "reference",
            "reference": {"result_line", "value", "rtol"}

C13 -- FEniCSx + SPARTA, conjugate heat transfer with a rarefied gas
-------------------------------------------------------------------
The band on the QoI is PRE-REGISTERED from slip and free-molecular theory; the
derivation is `c13_theory()` below and is written verbatim into the sealed key.
It was computed and frozen BEFORE the path walk ran, and the walk's own numbers
never touch the band. The conservation identity is the interface energy
balance: the flux the FEM solid conducts equals the flux the DSMC gas deposits,
within a tolerance that covers the DSMC sampling noise.

C14 -- 4C + FEniCSx, steady FSI
-------------------------------
The QoI standard is the root of the coupled interface system found by
Newton-Krylov over the same subproblem solves (matrix-free GMRES, no relaxation
parameter, cannot converge to a non-root) -- grade 2's exact claim: it proves
the SPLIT found the root, and says nothing an independent second code would.
The sealed value and its provenance are in the key.

Run:  python campaign3_blind/build_offpool.py [--apply] [--only C13 C14]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(os.environ.get("OPENPASO_REPO",
                           "/home/alexander/Schreibtisch/ofa-balanced"))
sys.path.insert(0, str(REPO / "src"))

from blind_eval.leakgate import scan                        # noqa: E402

INTEGRITY = """
INTEGRITY (absolute): every number you write must come from ACTUALLY RUNNING
the named solvers. Do not compute values by hand, by formula, or from memory,
and do not stub or mock either solver. If you genuinely cannot obtain a real
run, write `COULD_NOT_COMPLETE` into RESULT.txt and explain why. Inventing
numbers is a failure, worse than an honest incomplete.
PERSISTENCE: solvers will error first; read the error, fix it, re-run, iterate.
"""

EXEC_LOG = """
(3) EXECUTION LOG. Write one file PER PARTICIPANT
    run_level1_A.log     (written by the code that solved side A)
    run_level1_B.log     (written by the code that solved side B)
containing BOTH of the following:
  (a) THE CONSOLE OUTPUT THAT SIDE'S SOLVER ITSELF PRODUCED, captured verbatim
      -- redirect the run into the file, for example
          <run command for that side>  > run_level1_<side>.log 2>&1
      or raise the code's log/verbosity level so its own messages land there.
      Do not retype, summarise or paraphrase it: it must be the text that code
      emitted.
  (b) the line
          NDOF = <integer>
      where <integer> is, for a finite element side, the number of degrees of
      freedom of your discretisation, and for a DSMC side, the number of grid
      CELLS of your simulation volume.
A side without (b) counts as not run at all. A side whose log carries no output
from its own named code cannot be credited to that code. The two files must
therefore carry the output of two DIFFERENT codes.

(4) COUPLING HISTORY. Write
    residual_level1.csv
with a header line and one row per coupling iteration:
    iteration, interface_residual
listing the relative interface mismatch at every iteration of your partitioned
scheme, from the first to the last. A coupled run that cannot produce this did
not couple.
"""


# ══════════════════════════════════════════════════════════════════════
# C13 — the theory, the band, and the cell
# ══════════════════════════════════════════════════════════════════════
# Physical constants and the gas model (SPARTA's own ar.species / ar.vss):
KB = 1.380649e-23          # J/K
M_AR = 6.63e-26            # kg      (ar.species)
D_REF = 4.17e-10           # m       (ar.vss VSS/VHS reference diameter)
OMEGA = 0.81               # -       (ar.vss viscosity-temperature exponent)
T_REF = 273.0              # K       (ar.vss reference temperature)
GAMMA = 5.0 / 3.0          # monatomic
PR = 2.0 / 3.0             # monatomic Prandtl number

# The instance:
T_HOT, T_COLD = 400.0, 200.0       # K: outer solid face / outer gas wall
LS, LG, HH = 1.0e-3, 1.0e-3, 6.0e-4   # m: solid depth, gas depth, height
KS = 0.02                          # W/(m K): solid conductivity
N_MEAN = 1.3327e22                 # 1/m^3: mean gas number density (fill)
ZETA_SMOL = ((2 - 1) / 1) * (2 * GAMMA / (GAMMA + 1)) / PR   # = 1.875
ZETA_MAX = 2.5                     # conservative jump-coefficient upper edge
BAND_PAD = 0.01                    # absolute pad for DSMC statistics


def mu_vhs(T):
    """VHS viscosity from the ar.vss parameters (Bird 1994, eq. 4.62/4.63)."""
    mu_ref = (15.0 * math.sqrt(math.pi * M_AR * KB * T_REF)
              / (2.0 * math.pi * D_REF ** 2 * (5 - 2 * OMEGA)
                 * (7 - 2 * OMEGA)))
    return mu_ref * (T / T_REF) ** OMEGA


def k_gas(T):
    """Monatomic: k = (15/4) (kB/m) mu, exactly Pr = 2/3."""
    return 3.75 * (KB / M_AR) * mu_vhs(T)


def lam_vhs(T, n):
    """VHS mean free path (Bird 1994, eq. 4.65)."""
    return 1.0 / (math.sqrt(2.0) * math.pi * D_REF ** 2 * n
                  * (T_REF / T) ** (OMEGA - 0.5))


def c13_theory():
    """The pre-registered band, from slip and free-molecular theory.

    THE MODEL. Steady 1-D conjugate conduction. Solid: q = KS (T_hot - T_i)/LS.
    Gas: pressure is uniform in a closed steady gas at rest, so
    n(x) = p/(kB T(x)); the bulk carries q = (1/LG) * int k(T) dT between the
    two GAS edge temperatures (the conductivity integral, exact for continuum
    1-D with k(T) ~ T^omega); and at each wall the gas temperature differs from
    the wall temperature by the Smoluchowski jump
        T_gas,wall - T_wall = zeta * q / k(T_gas,wall),
        zeta = ((2-sigma)/sigma) * (2 gamma/(gamma+1)) * lambda / Pr
    which for full accommodation (sigma = 1, SPARTA's diffuse walls) and a
    monatomic gas is zeta = 1.875 lambda. The total particle count fixes p
    through the harmonic-mean temperature: p = N_MEAN kB LG / int dx/T(x),
    with T(x) from k(T) dT/dx = -q.

    THE BAND EDGES, and why the truth must lie between them:
      * LOWER edge of theta: zeta = 0 (continuum, no jump). Rarefaction can
        only ADD interfacial resistance -- the temperature jump is
        non-negative for any accommodation -- so the gas-side resistance is
        bounded below by the no-jump value and theta = R_gas/(R_solid + R_gas)
        from below.
      * UPPER edge: zeta = 2.5 lambda. Published jump coefficients for a
        monatomic gas with full accommodation span 1.87 (Smoluchowski) to
        ~2.2 (refined kinetic theory, e.g. variational solutions of the
        linearised Boltzmann equation); 2.5 covers that spread plus the
        residual Knudsen-layer effects at Kn ~ 0.1, which are O(Kn^2) in the
        bulk quantities.
      * The free-molecular limit is computed as a cross-check that the regime
        is far from FM (its flux exceeds the continuum flux by a factor ~4 at
        these densities, so the transport is collision-dominated and the slip
        family is the right theory; in the FM regime that ratio approaches 1
        from above and then inverts).
        [Kennard: q_fm = alpha/(2-alpha) * (gamma+1)/(gamma-1) * p cbar/(8 T)
        * (T1 - T2), evaluated at the mean state.]
      * BAND_PAD = 0.01 absolute on each edge covers the DSMC sampling noise
        of the QoI, which is a MEAN temperature ratio and self-averages.

    Everything here is closed-form or a 1-D quadrature; no simulation output
    of any kind enters. Frozen before the path walk ran.
    """
    from scipy.optimize import brentq
    from scipy.integrate import quad

    def solve_theta(zmult):
        def kint(t1, t2):
            return quad(k_gas, t1, t2)[0]

        def resid(Ti):
            q = KS * (T_HOT - Ti) / LS

            def edge(Twall, sign):
                # gas edge temperature at a wall: T_g = T_wall +/- zeta q / k
                def f(Tg):
                    n_wall = P[0] / (KB * Tg) if P else N_MEAN
                    z = zmult * lam_vhs(Tg, n_wall)
                    return Tg - (Twall + sign * z * q / k_gas(Tg))
                lo, hi = 150.0, 450.0
                return brentq(f, lo, hi)

            # inner fixed point for the uniform pressure
            P[:] = [N_MEAN * KB * 300.0]
            for _ in range(60):
                Tgh = edge(Ti, -1.0)      # gas is COOLER than the hot wall
                Tgc = edge(T_COLD, +1.0)  # gas is HOTTER than the cold wall
                if Tgh <= Tgc:
                    return q - 0.0        # degenerate; steer brentq away
                qgas = kint(Tgc, Tgh) / LG

                # T(x) from k dT/dx = -q  ->  x(T) = int k/q; then
                # int dx/T = int k/(q T) dT
                denom = quad(lambda T: k_gas(T) / (qgas * T), Tgc, Tgh)[0]
                pnew = N_MEAN * KB * LG / denom
                if abs(pnew - P[0]) < 1e-10 * P[0]:
                    P[0] = pnew
                    break
                P[0] = pnew
            return q - qgas

        P = [N_MEAN * KB * 300.0]
        Ti = brentq(resid, T_COLD + 1.0, T_HOT - 1.0, xtol=1e-10)
        q = KS * (T_HOT - Ti) / LS
        return Ti, q, P[0]

    Ti_lo, q_lo, p_lo = solve_theta(0.0)          # no jump: theta minimum
    Ti_hi, q_hi, p_hi = solve_theta(ZETA_MAX)     # conservative jump: maximum
    th_lo = (Ti_lo - T_COLD) / (T_HOT - T_COLD)
    th_hi = (Ti_hi - T_COLD) / (T_HOT - T_COLD)

    # diagnostics for the derivation record
    lam300 = lam_vhs(300.0, N_MEAN)
    kn = lam300 / LG
    cbar = math.sqrt(8 * KB * 300.0 / (math.pi * M_AR))
    p300 = N_MEAN * KB * 300.0
    q_fm = ((GAMMA + 1) / (GAMMA - 1)) * p300 * cbar / (8 * 300.0) \
        * (T_HOT - T_COLD)
    return {
        "theta_lower_nojump": th_lo, "theta_upper_jump2p5": th_hi,
        "band": [round(th_lo - BAND_PAD, 4), round(th_hi + BAND_PAD, 4)],
        "interface_T_lower": Ti_lo, "interface_T_upper": Ti_hi,
        "flux_lower_W_m2": q_hi, "flux_upper_W_m2": q_lo,
        "lambda_300K_m": lam300, "Kn_300K": kn,
        "k_gas_300K": k_gas(300.0), "mu_300K": mu_vhs(300.0),
        "pressure_Pa_approx": p_lo,
        "q_free_molecular_W_m2": q_fm,
        "fm_to_continuum_ratio": q_fm / q_lo,
        "zeta_smoluchowski_over_lambda": ZETA_SMOL,
        "zeta_upper_over_lambda": ZETA_MAX,
        "derivation": c13_theory.__doc__,
    }


def build_C13():
    th = c13_theory()
    lo, hi = th["band"]
    task = f"""Solve the following CONJUGATE HEAT TRANSFER problem as a COUPLED
simulation using TWO codes: FEniCSx (dolfinx) for the solid on side A and
SPARTA (the DSMC code; write an input script and run the spa_serial binary)
for the rarefied gas on side B.

GEOMETRY (2-D, SI units):
  side A (SOLID): the rectangle (0, {LS:g}) x (0, {HH:g})  [metres]
  side B (GAS):   the rectangle ({LS:g}, {LS + LG:g}) x (0, {HH:g})
  INTERFACE: the line x = {LS:g}, shared by both.

SOLID (side A): steady heat conduction, -div(ks grad T) = 0, with
  ks = {KS:g} W/(m K). No volumetric source.
  x = 0:      T = {T_HOT:g} K (Dirichlet)
  y = 0, y = {HH:g}: insulated (natural)
  x = {LS:g}: the coupling interface (receives the gas heat flux, see below)
  Discretise with Lagrange P1 elements on a uniform mesh with at least 40
  elements across the slab ({LS:g} m) and at least 24 along y.

GAS (side B): argon, simulated by DSMC with SPARTA's standard VSS model
  (the ar.species and ar.vss files shipped with SPARTA: mass 6.63e-26 kg,
  diameter 4.17e-10 m, omega 0.81, Tref 273 K, alpha 1.40).
  Mean number density n = {N_MEAN:g} 1/m^3 (fill the closed box uniformly at
  300 K with this density; there is NO inflow and NO outflow and NO stream
  velocity — this is a closed conduction cell).
  x = {LS + LG:g}: diffuse wall at T = {T_COLD:g} K, full accommodation.
  y boundaries: periodic.
  x = {LS:g}: diffuse wall at the INTERFACE TEMPERATURE imported from the
  solid, full accommodation — this is the coupling interface.
  Numerics: grid cells no larger than one third of the mean free path
  (lambda = {th['lambda_300K_m']:.3e} m at 300 K, so cells of at most
  {th['lambda_300K_m'] / 3:.1e} m; 30 x 18 cells is the minimum), at least 25
  simulated particles per cell, timestep at most 1e-7 s, at least 2000
  timesteps of equilibration before sampling and at least 4000 timesteps of
  flux averaging.

COUPLING SCHEME: a partitioned iteration in which the GAS side is the
Dirichlet-type participant (it receives the interface temperature and returns
the wall heat flux the gas deposits) and the SOLID side is the Neumann-type
participant (it receives that flux as a natural boundary condition on
x = {LS:g} and returns its interface temperature). These roles are prescribed:
SPARTA has no flux boundary condition, so they are the only arrangement its
wall model serves. Iterate until the relative interface mismatch is below
2e-2 OR it stalls at the DSMC statistical noise floor over at least 3
consecutive iterations; report what it reached.

SIGN CONVENTION: the gas-side export is the net energy flux the gas deposits
ON the wall per unit area (positive = the wall gains energy); applied on the
solid as an INFLOW of heat through x = {LS:g}.

REQUIRED OUTPUT (this is how your work is assessed; follow it exactly):

(1) RESULT.txt with exactly these lines:
    THETA_INTERFACE = <(mean interface temperature - {T_COLD:g}) / {T_HOT - T_COLD:g}>
    FLUX_SOLID = <the mean heat flux density through the solid, W/m^2, taken
                  from the solid solve as the total heat entering at x = 0
                  divided by the height {HH:g}>
    FLUX_GAS = <the mean energy flux density the gas deposits on the interface
                wall, W/m^2, from the DSMC surface tally>
    COUPLING_ITERATIONS = <number of coupling iterations>
    INTERFACE_RESIDUAL = <final relative interface mismatch>
    MESH_INDEPENDENCE = <CONVERGED or NOT_CONVERGED, judged from your own runs>
  The mean interface temperature is the average of the solid's interface nodal
  temperatures. FLUX_SOLID and FLUX_GAS are both positive numbers in the
  stated convention and describe the SAME physical flux; they are compared.

(2) There is no probe-grid output for this problem: the assessed quantities
are the RESULT.txt lines above.
{EXEC_LOG}{INTEGRITY}"""
    spec = dict(
        id="C13", kind="coupled", codes=["fenics", "sparta"], dim=2,
        coords=["x", "y"],
        physics="conjugate heat transfer between a conducting solid and a "
                "rarefied argon gas (FEM-DSMC)",
        physics_family="fem_dsmc", pooled=False,
        roles={"A": "neumann", "B": "dirichlet"},
        evidence_grade=3,
        evidence_grade_reason=(
            "band-only: a pre-registered band on a named scalar QoI plus a "
            "conservation identity. Proves the value is where physics says it "
            "must be; proves NOTHING about convergence to the right field. "
            "Never pooled with grade-1 cells."),
        interface_tol="2e-2 relative",
        qoi="THETA_INTERFACE, the dimensionless interface temperature, "
            "banded by slip/free-molecular theory",
        graded_against="pre-registered theoretical band plus the interface "
                       "energy-balance identity",
        notes_public=("the gas is rarefied (Kn ~ 0.1): expect a temperature "
                      "JUMP at both gas walls, and expect the DSMC flux tally "
                      "to carry statistical noise that a coupling tolerance "
                      "cannot be driven below."),
    )
    key = dict(
        id="C13", kind="coupled", codes=["fenics", "sparta"], dim=2,
        grading="band-only", evidence_grade=3, pooled=False,
        qoi={"result_line": "THETA_INTERFACE", "band": [lo, hi]},
        identity={
            "description": "interface energy balance: the flux the FEM solid "
                           "conducts equals the flux the DSMC gas deposits; "
                           "the tolerance covers the DSMC sampling noise",
            "lhs_line": "FLUX_SOLID", "rhs_line": "FLUX_GAS", "rtol": 0.08},
        theory=th,
        band_provenance=(
            "computed by build_offpool.c13_theory() from the VSS gas "
            "parameters and slab geometry alone, BEFORE the path walk ran; "
            "no simulation output of any kind entered the band"),
    )
    return task, spec, key


# ══════════════════════════════════════════════════════════════════════
# C14 — the FSI cell
# ══════════════════════════════════════════════════════════════════════
# The sealed QoI standard, produced by fsi_reference_newtonkrylov.py over the
# pinned participants (provenance in the key): the root of the coupled
# interface system, |R| = 8.42e-15 absolute / 2.16e-13 relative, 11 coupled
# evaluations. The walked partitioned run agreed with it to 1.6e-9 relative.
C14_UY_MIDPOINT = 0.00960226999006514
C14_RTOL = 0.05

C14_GEOM = dict(LX=1.0, HY=0.2, HS=0.05, MU=1.0, RHO=1.0, U_MEAN=1.0,
                E=3.0e6, NU=0.3, NXF=48, NYF=10, NXS=40, NYS=4)


def build_C14():
    g = C14_GEOM
    task = f"""Solve the following steady FLUID-STRUCTURE INTERACTION problem as
a COUPLED simulation using TWO codes: 4C (write a YAML input file and run the
4C binary) for the structure on side A and FEniCSx (dolfinx) for the fluid on
side B.

GEOMETRY (2-D, reference configuration):
  side B (FLUID):     the channel (0, {g['LX']:g}) x (0, {g['HY']:g})
  side A (STRUCTURE): the wall    (0, {g['LX']:g}) x ({g['HY']:g}, {g['HY'] + g['HS']:g})
  INTERFACE: the line y = {g['HY']:g} — the channel's top boundary and the
  wall's bottom (wetted) face.

FLUID (side B): steady incompressible Navier-Stokes,
  rho (u . grad) u = div sigma_f,  div u = 0,
  sigma_f = -p I + mu (grad u + grad u^T),  mu = {g['MU']:g}, rho = {g['RHO']:g}.
  x = 0:   parabolic inflow u = (6 U y ({g['HY']:g} - y)/{g['HY']:g}^2, 0) with
           mean U = {g['U_MEAN']:g}
  x = {g['LX']:g}:  natural (traction-free) outflow
  y = 0:   no-slip wall
  y = {g['HY']:g}: the FSI interface — no-slip on the DEFORMED interface: move
           the fluid mesh by a HARMONIC extension of the imported interface
           displacement (an ALE lift, zero on all other boundaries) and set
           u = 0 on the moved interface.
  Discretise with Taylor-Hood P2/P1 on a {g['NXF']} x {g['NYF']} triangulated
  grid of the reference channel; solve the nonlinear system with Newton.

STRUCTURE (side A): plane-strain linear elasticity,
  E = {g['E']:g}, nu = {g['NU']:g}. No body force.
  x = 0 and x = {g['LX']:g}: clamped (u = 0, both components, whole edge)
  y = {g['HY'] + g['HS']:g}: traction-free
  y = {g['HY']:g}: the FSI interface — carries the imported fluid traction as a
  Neumann load.
  Discretise with 4-node bilinear quadrilaterals on a {g['NXS']} x {g['NYS']}
  grid.

COUPLING SCHEME: a partitioned iteration exchanging the interface displacement
and the interface traction. The FLUID side receives the interface DISPLACEMENT
and imposes it through its ALE mesh motion; the STRUCTURE side receives the
fluid TRACTION and applies it as its interface Neumann load. These roles are
prescribed: solve with them as stated. Iterate until the relative interface
mismatch is below 1e-6.

SIGN CONVENTION (state of the art for getting FSI silently wrong, so it is
prescribed): the traction handed to the structure is t = sigma_f . n_s, where
n_s is the STRUCTURE's outward normal on the interface (pointing INTO the
fluid, n_s = -n_f). The structure applies the handed-in t directly, with no
further sign change. Check: a static fluid at pressure p > 0 under the wall
pushes the wall AWAY from the fluid.

Both interface parametrisations are LAGRANGIAN: exchange data at the REFERENCE
(undeformed) interface positions.

REQUIRED OUTPUT (this is how your work is assessed; follow it exactly):

(1) RESULT.txt with exactly these lines:
    UY_MIDPOINT = <the vertical displacement of the interface at the reference
                   point (x, y) = ({g['LX'] / 2:g}, {g['HY']:g}), from the
                   converged coupled state>
    COUPLING_ITERATIONS = <number of coupling iterations>
    INTERFACE_RESIDUAL = <final relative interface mismatch>
    MESH_INDEPENDENCE = <CONVERGED or NOT_CONVERGED, judged from your own runs>

(2) There is no probe-grid output for this problem: the assessed quantity is
the RESULT.txt line above.
{EXEC_LOG}{INTEGRITY}"""
    spec = dict(
        id="C14", kind="coupled", codes=["4C", "fenics"], dim=2,
        coords=["x", "y"],
        physics="steady fluid-structure interaction: incompressible "
                "Navier-Stokes channel under a clamped elastic wall",
        physics_family="fluid_structure_interaction", pooled=False,
        roles={"A": "neumann", "B": "dirichlet"},
        evidence_grade=2,
        evidence_grade_reason=(
            "reference: the QoI is compared against the root of the coupled "
            "interface system found by Newton-Krylov over the same subproblem "
            "solves. Proves the SPLIT found the root; a defect shared by both "
            "would agree and stay unseen. Never pooled with grade-1 cells."),
        interface_tol="1e-6 relative",
        qoi="UY_MIDPOINT, the interface vertical displacement at x = 1/2",
        graded_against="sealed Newton-Krylov re-solve of the coupled "
                       "interface system",
        geometry=C14_GEOM,
    )
    key = dict(
        id="C14", kind="coupled", codes=["4C", "fenics"], dim=2,
        grading="reference", evidence_grade=2, pooled=False,
        reference={"result_line": "UY_MIDPOINT", "value": C14_UY_MIDPOINT,
                   "rtol": C14_RTOL},
        reference_provenance=dict(
            method="scipy.optimize.root(method='krylov') on the coupled "
                   "interface residual R(y) = B(A(y)) - y, driving the same "
                   "participant solves as subprocesses "
                   "(data/coupling_participants/fsi_reference_newtonkrylov.py)",
            participants=["participant_fsi_fluid_fenics.py (Taylor-Hood "
                          "P2/P1 Navier-Stokes, harmonic ALE lift, "
                          "variationally consistent traction)",
                          "participant_fsi_solid_fourc.py (4C plane-strain "
                          "QUAD4 wall, exact piecewise-linear traction FUNCT)"],
            residual_absolute=8.422e-15, residual_relative=2.157e-13,
            coupled_evaluations=11,
            walk_agreement_relative=1.63e-9,
            rtol_rationale=(
                "0.05 passes legitimate discretisation variation around the "
                "prescribed meshes and fails every wiring error that "
                "motivates the cell: a flipped traction sign reverses the "
                "deflection (200% off), a wrong modulus scales it linearly, "
                "a one-way coupling changes it at the tens-of-percent level"),
        ),
    )
    return task, spec, key


# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    cells = []
    for pid, builder in (("C13", build_C13), ("C14", build_C14)):
        if args.only and pid not in args.only:
            continue
        task, spec, key = builder()
        gate = scan(task, {"exact_solution": None, "source_term": None}, pid)
        ok = gate.clean
        print(f"[{pid}] {'+'.join(spec['codes'])}  grade {key['evidence_grade']}"
              f"  {spec['physics'][:60]}")
        if pid == "C13":
            print(f"      band = {key['qoi']['band']}  "
                  f"(theory: nojump {key['theory']['theta_lower_nojump']:.4f}"
                  f" .. jump2.5 {key['theory']['theta_upper_jump2p5']:.4f}, "
                  f"Kn = {key['theory']['Kn_300K']:.3f}, "
                  f"q_fm/q_cont = {key['theory']['fm_to_continuum_ratio']:.1f})")
        else:
            print(f"      reference {key['reference']['result_line']} = "
                  f"{key['reference']['value']:.6e}  "
                  f"rtol {key['reference']['rtol']}")
        print(f"      leak gate={'CLEAN' if gate.clean else 'LEAK'} "
              f"{[f.rule for f in gate.findings] or ''}")
        for f in gate.findings:
            if f.severity in ("CRITICAL", "HIGH"):
                print(f"        [{f.severity}] {f.detail[:160]}")
                ok = False
        cells.append((pid, task, spec, key, ok))

    bad = [pid for pid, *_, ok in cells if not ok]
    if bad:
        print(f"FAILED: {bad}")
        return 1
    if not args.apply:
        print("(dry run -- pass --apply to write tasks and keys)")
        return 0

    problems = HERE / "problems"
    keys = Path(os.environ.get(
        "OPENPASO_BLIND_KEYS",
        "/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys"))
    for pid, task, spec, key, _ in cells:
        pdir = problems / pid
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / "task.txt").write_text(task, encoding="utf-8")
        (pdir / "spec_public.json").write_text(
            json.dumps(spec, indent=2, default=str))
        kdir = keys / pid
        kdir.mkdir(parents=True, exist_ok=True)
        (kdir / "key.json").write_text(json.dumps(key, indent=2, default=str),
                                       encoding="utf-8")
        print(f"wrote problems/{pid}/ and keys/{pid}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
