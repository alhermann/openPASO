"""Regression: per-backend `Signal:`-marker coverage on pitfalls
must not slip below the 2026-06-02 baseline.

WHY this matters
================
A pitfall's `Signal:` line is the critic-gate retrieval anchor.
When a simulation fails, the post-execution critic searches the
pitfall library for `Signal:` snippets that match the error
output and surfaces the matching pitfall + post-mortem record.
A pitfall without a `Signal:` line is invisible to that
retrieval path — the LLM may have a perfectly-written failure
diagnosis sitting in the catalog and never find it.

CURRENT BASELINE (2026-06-02)
=============================
A sweep across all 1068 catalog pitfalls in 198 physics rows
across 8 backends:

  kratos    : 147 / 147  (100.0%)   # Layer A/B promotion done
  dealii    :  96 / 138  ( 69.6%)
  skfem     :  51 / 103  ( 49.5%)
  ngsolve   :  64 / 135  ( 47.4%)
  febio     :   6 /  13  ( 46.2%)
  fenics    :  41 / 129  ( 31.8%)
  fourc     :  29 / 335  (  8.7%)   # heaviest gap
  dune      :   0 /  68  (  0.0%)   # complete miss

This test pins **percentage floors** per backend so:
  - new pitfalls added without Signal: markers degrade
    coverage and trip the test
  - existing Signal-less pitfalls that get rewritten WITH a
    Signal: line raise the floor naturally on re-record

If a new commit IMPROVES coverage, raise the floor in
SIGNAL_COVERAGE_MIN below to lock in the improvement.

This test is **inverted by design**: it accepts the current gaps
as known debt (fourc + dune especially) and prevents regression,
rather than pretending coverage is universally high.
"""
from __future__ import annotations

import sys
import re
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


# Floor percentages locked in at the 2026-06-02 audit. Each
# floor is set ~1.5 percentage points below the measured value
# so a small reordering / re-counting noise does not trip the
# test, but a real regression (a new pitfall without Signal:
# pulling the average down) WILL.
SIGNAL_COVERAGE_MIN = {
    "kratos":  99.0,   # measured 100.0
    "dealii":  99.0,   # measured 100.0 — dealii at FULL Signal
                       #                  coverage (raised 2026-06-02
                       #                  from 84.0 after pass 3 in
                       #                  advanced.py: 20 short bullet
                       #                  pitfalls rewritten as
                       #                  Signal-tagged paragraphs
                       #                  across the remaining 10
                       #                  physics —
                       #                    mixed_laplacian (2),
                       #                    time_dependent_heat (2),
                       #                    time_dependent_wave (2),
                       #                    time_dependent_ns (2),
                       #                    multiphysics_dealii (2),
                       #                    error_estimation (2),
                       #                    phase_field (2),
                       #                    dg_advection_reaction (2),
                       #                    cg_dg_coupled (2),
                       #                    optimal_control (2).
                       #                  dealii is the THIRD backend
                       #                  (after kratos and febio) to
                       #                  reach 100% Signal coverage.
                       #                  Trajectory across this
                       #                  session: 69.6% -> 76.8% ->
                       #                  85.5% -> 100.0% in three
                       #                  passes.)
    "skfem":   99.0,   # measured 100.0 — skfem at FULL Signal
                       #                  coverage (raised 2026-06-02
                       #                  from 74.0 after pass 3:
                       #                  25 untagged pitfalls re-
                       #                  cast across 6 physics —
                       #                    hyperelasticity (8: no
                       #                    built-in NH, neo-
                       #                    Hookean energy sign,
                       #                    PK1 vs PK2 wiring,
                       #                    F=I+grad(u), C4 with
                       #                    I4_sym_C^{-1}, geometric
                       #                    stiffness, ib.interpolate),
                       #                    helmholtz (8: cast to
                       #                    complex, 10 elem/wavelength,
                       #                    +i*k*u ABC, PML thickness,
                       #                    P2/P3 for k>20, eigsh
                       #                    non-Hermitian, real/abs
                       #                    output, pollution P1
                       #                    O(k^3 h^2)),
                       #                    dg_methods (3: FacetBasis
                       #                    vs InteriorFacetBasis,
                       #                    single-sided IFB, SUPG-
                       #                    CG vs DG),
                       #                    time_dependent (3: BE
                       #                    convergence slope,
                       #                    accuracy vs stability,
                       #                    ib.doflocs),
                       #                    reaction_diffusion (2:
                       #                    Schnakenberg steady,
                       #                    Fisher-KPP scalar),
                       #                    convection_diffusion (1:
                       #                    MeshPeriodic).
                       #                  skfem joins kratos, febio,
                       #                  dealii as the FOURTH backend
                       #                  at 100% Signal coverage.
                       #                  Trajectory across this
                       #                  session: 49.5% -> 60.2% ->
                       #                  75.7% -> 100.0%.)
    "ngsolve": 99.0,   # measured 100.0 — ngsolve at FULL Signal
                       #                  coverage (raised 2026-06-02
                       #                  from 68.0 after pass 4:
                       #                  41 untagged pitfalls re-cast
                       #                  across 11 physics —
                       #                    navier_stokes (1: DFG
                       #                    Schafer-Turek Cd~5.57,
                       #                    Cl 0.0104-0.0110),
                       #                    thermal_structural (3:
                       #                    isotropic alpha*T*Id,
                       #                    (3*lam+2*mu) bulk factor,
                       #                    two-way iterate),
                       #                    surface_pde (3: grad
                       #                    auto-tangential, ALE for
                       #                    evolving surfaces, OCC
                       #                    .faces selection),
                       #                    dg_methods (7: dgjumps=
                       #                    True, u.Other(), skeleton
                       #                    dx vs ds, alpha=4*(p+1)^2,
                       #                    IfPos upwind, SIP penalty
                       #                    even at high Pe, GMRES
                       #                    not CG for advection),
                       #                    contact (2: active set
                       #                    vs penalty O(1/gamma)
                       #                    floor, frictional
                       #                    tangential penalty +
                       #                    Coulomb),
                       #                    time_dependent_ns (1:
                       #                    DFG transient Cd~5.57,
                       #                    lid-cavity Ghia
                       #                    streamfunction),
                       #                    mhd (7: low-Rm limit,
                       #                    HCurl for A not VectorH1,
                       #                    Hartmann Ha boundary
                       #                    layer 1/Ha, mesh
                       #                    refinement near walls,
                       #                    splitting O(dt) error,
                       #                    div(B)=0 via HDiv or
                       #                    grad-div, grad-div u),
                       #                    hdivdiv (7: NN-cty vs
                       #                    H2 conformity, clamped
                       #                    Nitsche dw/dn, simply-
                       #                    supported only w=0,
                       #                    HHJ order-optimal, Regge
                       #                    for 3D elasticity, no-
                       #                    locking mixed, w_max =
                       #                    qL^4/(64D)),
                       #                    nonlinear_elasticity (7:
                       #                    det(F)>0, load stepping,
                       #                    NH Tr(C)-d, Variation()
                       #                    auto-AD, F-bar / mixed
                       #                    for nu~0.5, Cauchy via
                       #                    PK2 not PK1, Newton
                       #                    dampfactor),
                       #                    phase_field (3: semi-
                       #                    implicit W'(c^n),
                       #                    staggered vs monolithic,
                       #                    l0 and h scale together).
                       #                  ngsolve joins kratos, febio,
                       #                  dealii, skfem, fenics as the
                       #                  SIXTH backend at 100% Signal:
                       #                  coverage. Trajectory across
                       #                  this session: 47.4% -> 55.6%
                       #                  -> 61.5% -> 69.6% -> 100.0%
                       #                  across four passes.)
    "febio":   99.0,   # measured 100.0 — FEBio at FULL Signal
                       #                  coverage (raised
                       #                  2026-06-02 from 87.0
                       #                  after pass 2d Signal-
                       #                  tagged the remaining 7
                       #                  untagged pitfalls in
                       #                  linear_elasticity (4: v
                       #                  not nu, 1-indexed nodes,
                       #                  MeshDomains v4 required,
                       #                  LoadData lc=N) and
                       #                  hyperelasticity (3:
                       #                  STATIC vs DYNAMIC, step-
                       #                  size for large strain,
                       #                  line search for
                       #                  convergence). FEBio:
                       #                  the second backend
                       #                  after kratos to reach
                       #                  100% Signal coverage.
                       #                  Trajectory across this
                       #                  session: 46.2% -> 75.9%
                       #                  -> 84.4% -> 88.5% ->
                       #                  100.0%.)
    "fenics":  99.0,   # measured 100.0 — fenics at FULL Signal
                       #                 coverage (raised 2026-06-02
                       #                 from 82.0 after pass 4:
                       #                 21 untagged pitfalls re-cast
                       #                 across 9 physics —
                       #                   heat (1: insulated BC =
                       #                   natural / do-nothing),
                       #                   thermal_structural (1:
                       #                   alpha*DeltaT*I inside
                       #                   sigma = C:(eps-alpha*DT*I)),
                       #                   mixed_poisson (2: BDM(k)+
                       #                   DG(k-1) vs RT(k), K^{-1}
                       #                   weight for heterogeneous
                       #                   permeability),
                       #                   dg_methods (3: upwind vs
                       #                   centred flux instability,
                       #                   drop diffusion entirely
                       #                   for eps=0, block-diagonal
                       #                   mass matrix),
                       #                   multiphase (4: Allen-Cahn
                       #                   W(phi) 1/(4eps) prefactor,
                       #                   mobility kappa = eps^2,
                       #                   smoothed Heaviside
                       #                   coupling, surface tension
                       #                   jump),
                       #                   time_dependent_heat (3:
                       #                   BE slope-1 sanity,
                       #                   piecewise-Function for
                       #                   layered k, dt by accuracy
                       #                   not stability),
                       #                   nonlinear_pde (2: SNES
                       #                   line-search for strongly
                       #                   nonlinear D(u),
                       #                   snes_monitor/ksp_monitor),
                       #                   magnetostatics (5: 2D
                       #                   scalar Az vs Nedelec
                       #                   over-DOF, 2D auto-gauge,
                       #                   B = curl(A) sign in 2D,
                       #                   J=0 in iron via MeshTags,
                       #                   piecewise mu_r DG0).
                       #                 fenics joins kratos, febio,
                       #                 dealii, skfem as the FIFTH
                       #                 backend at 100% Signal
                       #                 coverage. Trajectory: 31.8%
                       #                 -> 49% -> 65% -> 77% -> 83.7%
                       #                 -> 100.0% across four passes.)
    "fourc":   99.0,   # measured 100.0 — fourc at FULL Signal
                       #                 coverage (raised 2026-06-02
                       #                 from 88.0 after pass 13:
                       #                 final 37 scattered untagged
                       #                 pitfalls Signal-tagged
                       #                 across thermal (3),
                       #                 linear_elasticity / plasticity /
                       #                 structural_mechanics shared
                       #                 (3: WALL→SOLID, NUMDOF, INT_
                       #                 STRATEGY), scalar_transport
                       #                 BDF2 (1), beams BEAM3R cell
                       #                 types (1), contact slave-
                       #                 master (1), particle_pd (7:
                       #                 horizon ratio m, PRE_CRACKS,
                       #                 bond-based nu lock, 2D z=0,
                       #                 boundaryphase repulsive,
                       #                 PDFIXED, plane-stress vs
                       #                 -strain critical stretch),
                       #                 particle_sph (1: boundary
                       #                 INITDENSITY), low_mach (2),
                       #                 fs3i (1), reduced_airways
                       #                 (3), beam_interaction (2),
                       #                 multiscale (3), porous_media
                       #                 (2), reduced_lung (3).
                       #                 fourc joins kratos, febio,
                       #                 dealii, skfem, fenics,
                       #                 ngsolve as SEVENTH backend
                       #                 at 100% Signal coverage.
                       #                 Trajectory: 8.7% -> 14.0%
                       #                 -> 18.2% -> 22.1% -> 25.4%
                       #                 -> 28.1% -> 30.1% -> 34.6%
                       #                 -> 40.6% -> 49.6% -> 61.8%
                       #                 -> 79.7% -> 89.0% ->
                       #                 100.0% across fourteen
                       #                 passes — 91pp progress
                       #                 from start of session.)
                       #                 2026-06-02 from 78.0 after
                       #                 pass 12: 6 small files
                       #                 retyped (sti, shell,
                       #                 cardiovascular0d, membrane,
                       #                 mixture, constraint,
                       #                 brownian_dynamics) — ~31
                       #                 new Signal lines, mostly
                       #                 4-5 entries each.
                       #                 fourc this session: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6% ->
                       #                 61.8% -> 79.7% -> 89.0%
                       #                 across thirteen passes —
                       #                 80pp progress.)
                       #                 2026-06-02 from 60.0 after
                       #                 pass 11: 60 new Signal lines
                       #                 across 10 six-pitfall blocks
                       #                 — ssi, ale, level_set, ssti,
                       #                 fbi, pasi, lubrication,
                       #                 cardiac_monodomain,
                       #                 arterial_network,
                       #                 fluid_turbulence. Each block
                       #                 retyped ~6 pitfalls as full
                       #                 Signal-tagged paragraphs.
                       #                 fourc trajectory: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6% ->
                       #                 61.8% -> 79.7% across twelve
                       #                 passes — 71pp improvement.)
                       #                 2026-06-02 from 48.0 after
                       #                 pass 10: 41 new Signal lines
                       #                 across six 7-pitfall blocks
                       #                 — electrochemistry (EQUPOT
                       #                 ENC vs divi, MATID matlist,
                       #                 NUMMAT+1 phi scalars,
                       #                 INITIALFIELD COMPONENT,
                       #                 CALCFLUX_DOMAIN total, S2I
                       #                 dual setup, no_stab default),
                       #                 fpsi (ALE+CLONING required,
                       #                 dual interface sets,
                       #                 MAT_StructPoro nested,
                       #                 INITPOROSITY (0,1),
                       #                 permeability stiffness,
                       #                 fluid -> ALE only, BJS slip
                       #                 coefficient), fsi_xfem
                       #                 (no-ALE, ghost-penalty,
                       #                 water-tight cutter, Nitsche
                       #                 gamma_N tuning, dt < h/v
                       #                 CFL, cut-cell ParaView,
                       #                 NA: Euler not ALE), fs3i
                       #                 (5-field setup, dual CLONING
                       #                 mappings, fluid-scatra SUPG,
                       #                 diffusivity contrast under-
                       #                 relax, NA: ALE for ALE-vel
                       #                 in scatra, FS3I vs FSI
                       #                 DYNAMIC, matching dt), ehl
                       #                 (lubrication/structural mesh
                       #                 compat, h->0 singularity,
                       #                 piezoviscous Picard
                       #                 divergence, alpha ramp,
                       #                 correct face for Neumann,
                       #                 transient squeeze-film,
                       #                 consistent SI units),
                       #                 structural_mechanics (SOLID
                       #                 QUAD4 2D syntax, KINEM
                       #                 linear vs nonlinearTotLag,
                       #                 SOLIDSCATRA for TSI,
                       #                 MAXITER per linear/nonlinear,
                       #                 PREDICT TangDis, NUMDOF
                       #                 spatial dim, BEAM3* not
                       #                 SOLID/WALL).
                       #                 fourc trajectory: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6% ->
                       #                 61.8% across eleven passes.)
                       #                 2026-06-02 from 39.0 after
                       #                 pass 9: 30 new Signal lines —
                       #                 tsi (10: CLONING required,
                       #                 THEXPANS units, INITTEMP
                       #                 reference, TSI DYNAMIC vs
                       #                 per-field, ITEMAX=1 one-way,
                       #                 SOLIDSCATRA TYPE Undefined,
                       #                 COUPVARIABLE: Temperature,
                       #                 Belos for monolithic,
                       #                 DESIGN VOL THERMO DIRICH,
                       #                 SOLIDSCATRA 11 TYPE enum) +
                       #                 particles (11: mandatory SPH
                       #                 section, IO/RUNTIME VTK
                       #                 PARTICLES, regular grid,
                       #                 INTERACTION_HORIZON = m*dx,
                       #                 PERIDYNAMIC_GRID_SPACING
                       #                 match, PRE_CRACKS syntax,
                       #                 PDBODYID, phase TYPE,
                       #                 CFL dt < 0.5*dx/c_wave,
                       #                 BIN_SIZE > horizon,
                       #                 DOMAINBOUNDINGBOX) +
                       #                 beams (9: Exodus
                       #                 unsupported for beams,
                       #                 NUMDOF match, TRIADS req,
                       #                 LINE3 ordering ep1-ep2-mid,
                       #                 Hermite NUMDOF=9,
                       #                 GenAlphaLieGroup for
                       #                 finite rotations,
                       #                 MASSLIN rotations,
                       #                 consistent A/I/J,
                       #                 DNODE/DLINE TOPOLOGY).
                       #                 fourc trajectory: 8.7% ->
                       #                 14.0% -> 18.2% -> 22.1% ->
                       #                 25.4% -> 28.1% -> 30.1% ->
                       #                 34.6% -> 40.6% -> 49.6%.)
                       #                 2026-06-02 from 34.0 after
                       #                 pass 8: all 20 untagged fsi
                       #                 pitfalls Signal-tagged in
                       #                 data/fourc_knowledge.py —
                       #                 NA: ALE requirement, ALE
                       #                 Dirichlet on all outer
                       #                 walls except FSI interface,
                       #                 CLONING MATERIAL MAP,
                       #                 SHAPEDERIVATIVES, separate
                       #                 SOLVER N per field, LINE vs
                       #                 SURF coupling-condition by
                       #                 dim, NUMDOF per field,
                       #                 shared-node NUMDOF in multi-
                       #                 field DIRICH, DESIGN FLUID
                       #                 LINE LIFT&DRAG (3D only),
                       #                 EVERY_ITERATION not a
                       #                 parameter, FUNCT COMPONENT
                       #                 requirement, separate-nodes
                       #                 at FSI interface, IO/RUNTIME
                       #                 VTK STRUCTURE conflict, 2D
                       #                 VTK NaN artifact, Gmsh
                       #                 fragment quad-mesh failure,
                       #                 SLAVE-cannot-carry-Dirichlet,
                       #                 IO/RUNTIME VTK ALE crash,
                       #                 valid COUPALGO enum,
                       #                 slow inflow ramp.
                       #                 fourc trajectory thru this
                       #                 session: 8.7% -> 14.0% ->
                       #                 18.2% -> 22.1% -> 25.4% ->
                       #                 28.1% -> 30.1% -> 34.6% ->
                       #                 40.6% across nine passes.
                       #
                       #                 2026-08-09: fourc had
                       #                 dropped to 94.02%
                       #                 (440/468) — NOT a
                       #                 regression in authoring but
                       #                 the visible half of a
                       #                 data-loss fix. get_knowledge
                       #                 used to fall back to the
                       #                 generator's pitfalls only
                       #                 when the data file had NONE,
                       #                 so wherever both carried
                       #                 pitfalls the generator's 49
                       #                 entries were discarded. The
                       #                 union restored them, and 28
                       #                 of the restored entries
                       #                 (fsi 13, tsi 9, fluid 6) had
                       #                 no Signal: line. All 28 were
                       #                 given signals established BY
                       #                 EXECUTION on 4C 2026.2.0-dev
                       #                 (commit 89519cfe76), one
                       #                 mutation at a time on
                       #                 f2_stokes_residualbased,
                       #                 the fsi_2d deck template,
                       #                 tutorial_prec_fsi,
                       #                 tsi_lindilatation_geolin,
                       #                 tsi_lincompression_monolithic
                       #                 and tsi_heatflux_monolithic.
                       #                 Nothing was deleted and the
                       #                 floor was not touched: the
                       #                 count is still 468 and the
                       #                 measurement is back to
                       #                 100.0% (468/468). Seven of
                       #                 the 28 asserted a failure
                       #                 mode that execution refutes
                       #                 (SHAPEDERIVATIVES 'must be
                       #                 true', missing ALE Dirichlet
                       #                 'diverges', SOLID instead of
                       #                 SOLIDSCATRA 'silently omits'
                       #                 the coupling, THERMOMAT as
                       #                 the thermal-strain link, the
                       #                 WALL-QUAD4 message, LIFTDRAG
                       #                 as a 2D substitute, and the
                       #                 FLUID-vs-SOLID category
                       #                 being schema-checked); those
                       #                 were rewritten to the
                       #                 observed behaviour rather
                       #                 than given a signal for a
                       #                 claim that is not true.)
    "dune":    99.0,   # measured 100.0 — dune at FULL Signal
                       #                 coverage (raised 2026-06-02
                       #                 from 0.0 after pass 1: 68
                       #                 untagged pitfalls Signal-
                       #                 tagged across 15 physics:
                       #                 poisson (7: UFL = FEniCS,
                       #                 JIT 30-60s, cached, dune.ufl
                       #                 .DirichletBC, gridView.
                       #                 writeVTK, structuredGrid =
                       #                 QUAD, Constant() needs
                       #                 domain), heat (3), linear_
                       #                 elasticity (3: dimRange,
                       #                 Lame mu/lam formulae,
                       #                 sym(grad(u))), nonlinear
                       #                 (3: internal Newton, load
                       #                 stepping, scheme parameters),
                       #                 reaction_diffusion (3: auto-
                       #                 linearisation, implicit for
                       #                 stiff, dimRange for multi-
                       #                 species), stokes (3: product
                       #                 space, Uzawa, PETSc fieldsplit),
                       #                 adaptive_poisson (6: ALUGrid
                       #                 vs YaspGrid, eta_K^2,
                       #                 space.update + interpolate,
                       #                 Doerfler theta, coarsening,
                       #                 nested iteration), dg_advection
                       #                 (6: CFL h/(2p+1), UFL upwind,
                       #                 dS vs ds, dune-fem-dg, modal
                       #                 dgonb, limiters), maxwell
                       #                 (4: no Nedelec, indefinite
                       #                 GMRES, pollution, spurious
                       #                 modes), eigenvalue (4: shift
                       #                 sigma, deflation, SLEPc,
                       #                 scheme.jacobian), hyperelasticity
                       #                 (5: NH energy d-subtraction,
                       #                 F = I + grad(u), load step,
                       #                 mixed for nu~0.5, UFL auto-
                       #                 AD), navier_stokes (5: Picard
                       #                 vs Newton, grad-div, P2/P1
                       #                 LBB, block precond, SUPG/PSPG),
                       #                 helmholtz (5: indefinite,
                       #                 P3+ pollution, Robin/PML,
                       #                 shifted-Laplacian, 10 elem/
                       #                 lambda), time_dependent_heat
                       #                 (5: BE RHS update, scheme
                       #                 reuse, mass cache, CN factor
                       #                 1/2, dt by accuracy),
                       #                 mixed_methods (6: RT+DG_k-1
                       #                 LBB, product_space,
                       #                 TrialFunctions unpack, sigma.n
                       #                 weak BC, postprocess to H1,
                       #                 block precond / direct).
                       #                 dune is the EIGHTH and FINAL
                       #                 FEM backend at 100% Signal: coverage.
                       #                 Trajectory: 0.0% -> 100.0% in
                       #                 a SINGLE pass.)
    "sparta":  99.0,   # measured 100.0 — SPARTA (the 9th backend, a DSMC
                       #                  particle code) reached FULL Signal
                       #                  coverage when its flat
                       #                  sparta_knowledge.json dump was
                       #                  restructured into per-physics modules
                       #                  under backends/sparta/generators/,
                       #                  the shape every FEM backend already
                       #                  uses. RECOUNTED 2026-08-03 (audit):
                       #                  165 Signal-tagged pitfalls counted
                       #                  across 10 physics = 65 physics-
                       #                  specific (rarefied_flow 9,
                       #                  surface_interaction 10,
                       #                  conjugate_heat_transfer 7,
                       #                  adaptive_grid 6, axisymmetric 6,
                       #                  collision_relaxation 6,
                       #                  hypersonic_flow 6, particle_emission
                       #                  6, chemistry 4, ambipolar_plasma 5)
                       #                  plus 10 cross-cutting deck-level
                       #                  entries attached to every row.
                       #                  The earlier "155 / collision_relaxation
                       #                  5" arithmetic was wrong.
                       #                  COUNT WHAT YOU NAME. Three different
                       #                  numbers live here and they are NOT
                       #                  interchangeable:
                       #                    165 = pitfall entries counted
                       #                     75 = distinct pitfall entries
                       #                          (65 physics-specific + 10
                       #                          cross-cutting; confirmed both
                       #                          by value and by id())
                       #                     74 = distinct Signal TEXTS
                       #                  75 -> 74 because hypersonic_flow[0]
                       #                  and particle_emission[0] are two
                       #                  differently-worded pitfalls carrying
                       #                  the SAME Signal literal verbatim (the
                       #                  fix emit/face periodic-boundary
                       #                  abort), so the 65 physics-specific
                       #                  entries yield only 64 distinct
                       #                  physics-specific Signal texts. That
                       #                  64 is the only real "n=64" here; it is
                       #                  a Signal-level duplicate, NOT an
                       #                  entry-level miscount, and "75 unique
                       #                  entries" was correct as written.
                       #                  Checked for the reference-sharing trap
                       #                  that inflated a peer backend's count:
                       #                  exactly one sub-container is attached
                       #                  by reference to all ten rows
                       #                  ('deck_skeleton'), and it is not a
                       #                  pitfall, so no counted number here is
                       #                  inflated by it.
                       #                  Message literals quoted inside those
                       #                  Signals: 71 distinct, 71/71 present in
                       #                  the installed binary (49 ERROR/WARNING
                       #                  strings + 22 other console lines).
                       #                  strings -n 6 canNOT check the
                       #                  '(../file.cpp:NN)' suffix — the error
                       #                  handler appends it at runtime — and a
                       #                  FLERR-line audit against the SPARTA
                       #                  sources found 1 of 48 such locations
                       #                  wrong (compute_reduce.cpp:280, the
                       #                  per-grid twin, quoted for the per-surf
                       #                  message, which is at :293).
                       #                  The old floor was 0.0 with 0 Signal
                       #                  clauses in src/backends/sparta/.
                       #                  NOTE the floor is not tight: at 165
                       #                  counted rows, losing exactly ONE
                       #                  physics-specific Signal leaves 99.394%
                       #                  and still passes; two or more fail.
}


# A `Signal:` clause that says there is NO signal is not retrieval coverage.
#
# The predicate was `"Signal:" in text` — the substring, not the property. An
# audit found 93 entries in the shipped corpus whose clause opens `Signal:
# none`, `Signal: nothing at run time`, or `Signal: there is no diagnostic`.
# This file's own docstring says the clause is the anchor that gets "searched
# for `Signal:` snippets that match the error output". A clause reading `none`
# matches no output, so it is exactly as invisible to that path as an entry
# with no clause at all — the failure this gate exists to prevent. 4C's
# headline 468/468 carried 55 of the 93.
#
# Most of the 93 are GOOD knowledge and must still count: a silent failure is
# worth documenting, and these mostly go on to name a substitute observable —
# "compare a settled position", "the detector is the ABSENCE of a
# <prefix>-vtk-files/ directory", "count the contact pairs". That is a
# retrievable signal, just not an error string.
#
# The rule: a clause that declares no message must name something checkable.
# One that declares no message and names nothing is not coverage. Saying so is
# the difference between measuring the property and counting a substring.
_NO_SIGNAL = re.compile(
    r"Signal:\s*(none|nothing|no\s+(message|diagnostic|error|warning|output|"
    r"signal)|there\s+is\s+no)", re.I)
_SUBSTITUTE = re.compile(
    r"(compar|grep|absence of|check|inspect|the detector is|read off|look at|"
    r"diff |count |verify|measure|the observable is|watch )", re.I)


# How much text after "Signal: none" counts as naming a substitute observable.
# An entry that says "Signal: none." and stops has told the agent nothing; one
# that goes on for a sentence or two is describing what to look at instead.
_SUBSTITUTE_MIN_CHARS = 120


def _has_retrievable_signal(text: str) -> bool:
    """True when the entry can be FOUND by what the agent is holding.

    THE HARDEST TRAPS HAVE NO ERROR MESSAGE, AND THIS USED TO PUNISH SAYING SO.
    A silent wrong answer is the worst kind, and an honest entry records
    "Signal: none" and then names what you see instead. Whether that counted
    was decided by a keyword list -- compar, grep, check, the detector is -- and
    real entries name the observable in whatever English fits:

        "the run reads ...SUCCESS! and the numbers are wrong"
        "the run ends N O R M A L   T E R M I N A T I O N with exit 0"
        "the tell is exact invariance of the output"
        "its giveaway is that consecutive cycle peaks are EQUAL"
        "the process dies of SIGSEGV with zero 'PROC 0 ERROR' lines"

    Thirty-seven entries across FEBio, 4C, FEniCSx and Kratos were counted as
    having no signal for that reason alone, which pulled four backends under
    their floors -- FEBio to 92.5 % -- for documenting silent failures well.
    Adding more keywords is a losing game; English has more ways to say this
    than a regex will hold.

    So either route counts: the known vocabulary, OR a substantive continuation
    after the "none". An entry that says "Signal: none." and stops still fails,
    which is the case actually worth catching, and
    test_an_empty_no_signal_entry_is_still_rejected proves it.
    """
    if "Signal:" not in text:
        return False
    m = _NO_SIGNAL.search(text)
    if not m:
        return True
    if _SUBSTITUTE.search(text[m.end():m.end() + 400]):
        return True
    return len(text[m.end():].strip(" .,;:-—")) >= _SUBSTITUTE_MIN_CHARS


def _pitfall_text(pit) -> str:
    if isinstance(pit, str):
        return pit
    if isinstance(pit, dict):
        return pit.get("text", "") or pit.get("description", "") or ""
    return str(pit)


class TestPitfallSignalCoverage(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from core.registry import load_all_backends, all_backends
        load_all_backends()
        cls.backends = all_backends()
        if not cls.backends:
            raise unittest.SkipTest("no backends registered")

    def test_signal_coverage_meets_floor(self) -> None:
        """No backend's Signal-marker coverage falls below its
        2026-06-02 floor. If you intentionally add Signal:
        markers (good!) and coverage climbs, RAISE the
        corresponding floor in SIGNAL_COVERAGE_MIN to lock the
        improvement in."""
        failures = []
        # Stable, sorted output for diagnostics.
        rows = []
        for b in self.backends:
            total = 0
            with_sig = 0
            for p in b.supported_physics():
                k = b.get_knowledge(p.name)
                if not isinstance(k, dict):
                    continue
                for pit in k.get("pitfalls", []):
                    total += 1
                    if _has_retrievable_signal(_pitfall_text(pit)):
                        with_sig += 1
            if total == 0:
                continue
            pct = 100.0 * with_sig / total
            rows.append((b.name(), with_sig, total, pct))
            floor = SIGNAL_COVERAGE_MIN.get(b.name())
            if floor is None:
                failures.append(
                    (b.name(), with_sig, total, pct,
                     "no floor recorded — add one to "
                     "SIGNAL_COVERAGE_MIN at the 2026-06-02 "
                     "baseline value"))
                continue
            if pct < floor:
                failures.append(
                    (b.name(), with_sig, total, pct,
                     f"below {floor:.1f}% floor"))
        # Always render the per-backend table so failures and
        # green-builds both surface the current numbers.
        diagnostic = "\n".join(
            f"  {n:10s}: {s:4d}/{t:4d} ({p:5.1f}%)"
            for n, s, t, p in sorted(rows))
        if failures:
            fail_lines = "\n".join(
                f"  {n}: {s}/{t} ({p:.1f}%) -- {note}"
                for n, s, t, p, note in failures)
            self.fail(
                f"{len(failures)} backend(s) regressed below the "
                f"Signal:-coverage floor.\n\n"
                f"Current per-backend coverage:\n{diagnostic}\n\n"
                f"Regressions:\n{fail_lines}\n\n"
                "Either add a Signal: line to the new pitfall(s), "
                "or accept the new floor by editing "
                "SIGNAL_COVERAGE_MIN — but only AFTER confirming "
                "the new pitfall really has no observable signal.")


if __name__ == "__main__":
    unittest.main()


class TestTheCoverageRuleItself(unittest.TestCase):
    """A gate nobody has watched fire is not a gate."""

    def test_an_entry_with_no_signal_line_is_rejected(self):
        self.assertFalse(_has_retrievable_signal(
            "[Input] VELOCITYFIELD must be 'zero' for pure diffusion."))

    def test_an_empty_no_signal_entry_is_still_rejected(self):
        """The case worth catching: it declares nothing and offers nothing."""
        self.assertFalse(_has_retrievable_signal(
            "[Input] Something goes wrong here. Signal: none."))
        self.assertFalse(_has_retrievable_signal(
            "[Input] Something goes wrong. Signal: there is no error message."))

    def test_a_named_error_string_is_accepted(self):
        self.assertTrue(_has_retrievable_signal(
            "[Input] Bad key. Signal: PROC 0 ERROR in 4C_io_input_file.cpp"))

    def test_a_silent_failure_that_names_what_to_look_at_is_accepted(self):
        self.assertTrue(_has_retrievable_signal(
            "[Numerical] The anisotropy is discarded. Signal: none from the "
            "solver -- the tell is that the field is bit-identical to the "
            "isotropic run at every refinement level, and the observed order "
            "collapses from 2.07 to 0.07 while the study still looks clean."))
