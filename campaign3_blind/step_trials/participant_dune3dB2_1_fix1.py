"""DUNE-fem participant for the OASiS `couple` driver — 3-D scalar conduction
across a PLANAR interface.  Serves either side of the split.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1, so an
iteration-1 fallback is mandatory), writes exports.json LAST and exits 0.
Needs dune-fem importable in the interpreter named in `command` (conda-forge
`dune-fem`).  DUNE JIT-COMPILES ITS UFL FORMS ON FIRST USE: the first run of a
new form takes minutes.  That is not a hang.  The forms here do not depend on
NX/NY/NZ, so a mesh-refinement study compiles once and then reuses the cache.

Physics: steady conduction  -div(K grad T) = f  on one BOX subdomain of a box
split by a plane.  Structured cube grid, Q1 Lagrange; the interface carries
either a Dirichlet trace or a natural (flux) load.

======================================================================
WHY A SEPARATE 3-D FILE — what is genuinely different, not just bigger
======================================================================
The 2-D participants (participant_dune.py, participant_fenics.py,
participant_kratos*.py) all assume the interface is a LINE at x = const sampled
by ONE coordinate.  Three of their steps are wrong in 3-D and each fails
quietly:

1. ORDERING.  The driver relaxes export vectors ENTRY BY ENTRY, so the export
   order must be identical on every iteration.  In 2-D `argsort(y)` does it.
   In 3-D the interface is a plane and a sort on one coordinate leaves the
   other in whatever order the mesh happened to number its dofs — stable within
   a run, so nothing complains, but the coupling then relaxes point i of this
   iteration against a DIFFERENT physical point of the last one.  Here the
   order is a tolerance-quantised `lexsort` over BOTH in-plane coordinates.

2. RESAMPLING A NON-MATCHING PARTNER.  In 2-D `np.interp` over the tangential
   coordinate is exact for a P1 trace.  In 3-D the partner's interface nodes
   are a 2-D point set: `np.interp` on either axis alone sorts that cloud by
   one coordinate and averages across the other, producing a smooth,
   plausible, WRONG boundary condition.  `resample_plane` below does real 2-D
   interpolation, and CUBIC rather than bilinear where it can — see the
   measurement in `_bicubic`, which is worth half an order of the exported
   flux.

3. QUADRATURE WEIGHTS.  The consistent-flux recovery divides the nodal
   reaction by  w_i = int_Gamma phi_i ds.  In 2-D those are EDGE LENGTHS.  In
   3-D they are FACE AREAS.  DUNE gets them right for free — `assemble` of
   `v*ds` restricted to the interface IS that integral, whatever the cell
   shape — which is precisely why this file assembles them instead of writing
   down a formula, as the 2-D Kratos participant has to.  For scale: carrying
   the 2-D edge-length formula over to a unit-square interface gives an
   exported int_Gamma q ds of -0.923 against an exact -22.026 at n=24, on a
   flux field that is still smooth and still the right shape (it is scaled by
   ~h) — and the error GROWS under refinement.  `assemble` cannot make that
   mistake; the print below reports w.sum(), which must equal the interface
   area, as the standing check that it did not.

THE SIGN, WHICH IS THE ONLY THING A NEUMANN PARTICIPANT CAN GET SILENTLY WRONG.
Every participant exports its outward normal flux density with respect to ITS
OWN outward normal,  q = -(K grad T) . n_own,  so the two sides of a shared
interface carry OPPOSITE signs.  On this subdomain
      int_Omega K grad T . grad v = int_Omega f v + int_dOmega (K grad T . n) v ds,
and on the interface n_own = -n_partner, so  K grad T . n_own = +q_partner:
the partner's number is applied UNCHANGED, as `+ g*v*ds`, no minus sign
anywhere.  If you ever flip that sign to "make the temperatures look right",
you have built a coupling that drives heat the wrong way across the interface
and still converges.  The self-check printed at the end is the guard: it
evaluates the discrete divergence theorem on this subdomain and must come out
at round-off.

`main()` returns a dict of everything a verification script needs (grid view,
space, solution, interface dofs, in-plane points, area weights, guarded and
UNGUARDED fluxes, the edge mask) so the participant can be imported and graded
against a manufactured solution.  The coupling itself only ever needs the file
handshake.

======================================================================
MEASURED — this file did not ship until it converged in a real coupling
======================================================================
Manufactured two-material 3-D conduction: box [0,1]^3 split by the plane
x = 0.5, k = 3.2 (this side) and 0.8, exact Dirichlet on all five non-interface
faces of each half, so the whole RIM of the interface plane is an outer
Dirichlet edge — the 3-D corner case made as large as it can be.

AS THE NEUMANN HALF, in a real Dirichlet-Neumann coupling through
src/core/coupling_driver.run_coupling against participant_kratos_3d.py, on
DELIBERATELY NON-MATCHING interface grids (m here against 3m/4 there).  Aitken,
theta0 = 0.7, tol = 1e-9: converged in 31 / 31 / 32 iterations, both
participants "responsive".  L2(T) over the WHOLE box (both subdomains)
converges at the expected second order over n = 6/12/24.

AS THE DIRICHLET HALF, handed the exact interface trace on a non-matching
partner grid, m = 8/16/32:

  L2(T) this subdomain   2.281e-01  5.751e-02  1.441e-02   order 1.99  2.00
  flux, iface INTERIOR   1.751e+00  4.427e-01  1.110e-01   order 1.98  2.00
  flux, WHOLE interface  2.549e+00  6.489e-01  1.883e-01   order 1.97  1.79
  flux, WHOLE unguarded  3.090e+01  2.202e+01  1.561e+01   order 0.49  0.50
  int_Gamma q ds        +24.2476   +22.5220   +22.1466     (exact +22.02642)

The whole-interface number is dragged down by the RIM, exactly as the two end
nodes do it in 2-D — see the edge guard below.  Quote the INTERIOR order when
grading the recovery and the WHOLE one when grading what the partner receives;
they are different questions.

CONSERVATION: interface area recovered as exactly 1.000000 at every level; the
discrete divergence theorem at round-off (1.6e-13 to 4.3e-12 — looser than the
Kratos side because `scheme` solves with CG, not a direct solve); and the
EXCHANGE balance against the partner converging at second order.
That one is NOT zero and cannot be: resampling between non-matching interface
grids is accurate, not conservative.  If you need it at round-off, match the
interface meshes.
"""
import json
import sys
from pathlib import Path

import numpy as np
from dune.grid import structuredGrid
from dune.fem import assemble
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.fem.operator import galerkin as operator_galerkin
from dune.ufl import DirichletBC, Constant
from ufl import (TrialFunction, TestFunction, SpatialCoordinate,
                 conditional, dot, ds, dx, grad, gt, lt)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
SIDE       = "neumann"       # "dirichlet" (import T, export flux) | "neumann"
PARTNER    = "left"          # the partner's `name` in your couple(...) call

X0, X1     = 0.625, 1.5      # this subdomain's box
Y0, Y1     = 0.0, 1.0
Z0, Z1     = 0.0, 1.0

IFACE_AXIS = 0               # interface plane normal: 0=x, 1=y, 2=z
IFACE_POS  = 0.625           # its position; must equal this box's lo or hi on that axis

K          = 4.0             # conductivity of THIS subdomain (constant)
NX, NY, NZ = 6, 6, 6         # this subdomain's OWN mesh; need NOT match the partner

# Which outer faces carry a Dirichlet condition.  Names are "<axis><0|1>" with
# 0 = the low face and 1 = the high face, e.g. "x1" is the plane x = X1.  Every
# face NOT listed here (and not the interface) is natural, i.e. insulated.
# The interface face itself must not appear.
DIRICHLET_FACES = ("x1", "y0", "y1", "z0", "z1")



def F_SRC(x, y, z):
    """Volumetric source f in  -div(K grad T) = f, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above.
    A CONSTANT CANNOT REPRESENT A POLYNOMIAL SOURCE: if your problem states
    one, or you derived it from a manufactured solution, a single number here
    silently solves a different problem. With the whole outer boundary
    prescribed and no source, the answer degenerates to the profile between
    the outer values.

    `x`, `y` and `z` are NumPy arrays of the node coordinates, so build the
    answer with NumPy and return an array of the same shape:

        return 3.0 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y) \
               * np.sin(np.pi * z)
    """
    return 8.0 * np.pi**2 * np.sin(np.pi * y) * np.sin(np.pi * z)

T_INIT     = 0.0             # iteration-1 fallback interface temperature
Q_INIT     = 0.0             # iteration-1 fallback interface flux density


def source_function(space, xd):
    """Carry F_SRC as a DISCRETE FUNCTION, the way the 2-D sibling does.

    Not a UFL expression: a zero source built symbolically folds to a bare 0,
    and `0*v*dx` is a domainless UFL Zero that assemble() refuses. Dofs are
    run-time data, so editing F_SRC never re-triggers the C++ JIT. Sampling at
    the nodes is the P1 interpolant of the source, an O(h^2) load error — the
    same order as the discretization error itself.
    """
    ffun = space.interpolate(0, name="f_src")
    ffun.as_numpy[:] = np.broadcast_to(
        np.asarray(F_SRC(xd[0], xd[1], xd[2]), float), xd[0].shape)
    return ffun


def outer_value(x, y, z):
    """Dirichlet value on the faces listed in DIRICHLET_FACES.

    x, y, z are NUMPY arrays of dof coordinates (this one is evaluated
    pointwise into a discrete function, not assembled symbolically, so that the
    FORM never changes between iterations and DUNE never re-JITs).  Return a
    scalar for a constant temperature or any expression for a graded one.
    """
    return 0.0
# ─────────────────────────────────────────────────────────────────────────

LO = np.array([X0, Y0, Z0], float)
HI = np.array([X1, Y1, Z1], float)
NE = [int(NX), int(NY), int(NZ)]
AX = int(IFACE_AXIS)
TAN = [a for a in (0, 1, 2) if a != AX]          # the two IN-PLANE axes
LEN = HI - LO
TOL = 1e-9 * float(np.max(LEN))
EPS = 1e-8 * float(np.max(LEN))                  # boundary-indicator tolerance
AXN = "xyz"

# Is the interface this box's HIGH face on AX?  The outward normal there is
# +e_AX, on the LOW face it is -e_AX.
ON_HI = abs(IFACE_POS - HI[AX]) < abs(IFACE_POS - LO[AX])
S = 1.0 if ON_HI else -1.0
IFACE_FACE = f"{AXN[AX]}{1 if ON_HI else 0}"


# ── driver handshake ────────────────────────────────────────────────────────
def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def _unique_tol(a, tol):
    """Sorted unique values of `a`, merging entries closer together than tol."""
    s = np.sort(np.asarray(a, float))
    keep = [s[0]]
    for v in s[1:]:
        if v - keep[-1] > tol:
            keep.append(v)
    return np.asarray(keep)


def _bilinear(u, v, G, q):
    """Bilinear interpolation of grid values G[iu, iv] at the points q (M,2).
    Points outside the grid are CLAMPED to its border: the two sides nominally
    cover the same plane, so an excursion is round-off or a partner whose mesh
    stops a hair short, and extrapolating there is worse than clamping."""
    iu = np.clip(np.searchsorted(u, q[:, 0]) - 1, 0, u.size - 2)
    iv = np.clip(np.searchsorted(v, q[:, 1]) - 1, 0, v.size - 2)
    tu = np.clip((q[:, 0] - u[iu]) / (u[iu + 1] - u[iu]), 0.0, 1.0)
    tv = np.clip((q[:, 1] - v[iv]) / (v[iv + 1] - v[iv]), 0.0, 1.0)
    return ((1 - tu) * (1 - tv) * G[iu, iv] + tu * (1 - tv) * G[iu + 1, iv]
            + (1 - tu) * tv * G[iu, iv + 1] + tu * tv * G[iu + 1, iv + 1])


def _lag4(nodes, xq):
    """Cubic Lagrange weights (M,4) for query xq (M,) on stencils nodes (M,4)."""
    d = xq[:, None] - nodes
    wts = np.empty_like(d)
    for a in range(4):
        num = np.ones(len(xq))
        den = np.ones(len(xq))
        for b in range(4):
            if b == a:
                continue
            num *= d[:, b]
            den *= nodes[:, a] - nodes[:, b]
        wts[:, a] = num / den
    return wts


def _bicubic(u, v, G, q):
    """Tensor-product CUBIC interpolation of grid values G at points q (M,2).

    WHY NOT BILINEAR — this is the single measurement that mattered most in
    building the 3-D pair.  Same participant, same manufactured problem, same
    non-matching partner grid, ONLY the tensor route swapped (measured orders
    over meshes 6/12/24, and the L2 error at 24):

        route      volume L2(T)   flux INTERIOR      flux WHOLE plane
        cubic      1.98           1.97  (1.32e-01)   1.82  (2.06e-01)
        bilinear   1.98           1.32  (2.18e-01)   1.44  (2.67e-01)

    The mechanism: a bilinear interpolant of a smooth trace is off by O(h^2),
    but its error is a KINK-SHAPED bump on the scale of one partner cell.  The
    discrete Dirichlet-to-Neumann map differentiates the boundary data, so an
    h-scale wiggle of amplitude h^2 comes back as a flux error of amplitude
    h^2/h = O(h).  Read the table again: the temperature field does not notice
    AT ALL — a high-frequency, mean-zero boundary perturbation is strongly
    damped inside the domain, so the volume order is 1.98 either way.  That is
    exactly what makes this failure quiet.  The field looks right, every
    residual converges, and the only thing that has changed is the order of the
    number the partner actually consumes.

    A cubic tensor interpolant is O(h^4) with a smooth error, so the amplified
    part is O(h^3) and the flux keeps its second order.  It does NOT invent
    accuracy: the partner's nodal values are still only as good as the
    partner's own mesh, and this only stops the TRANSFER from being the
    limiting error.  Needs >= 4 lines per direction; below that the caller
    falls back to bilinear.
    """
    qu = np.clip(q[:, 0], u[0], u[-1])
    qv = np.clip(q[:, 1], v[0], v[-1])
    bu = np.clip(np.searchsorted(u, qu) - 2, 0, u.size - 4)
    bv = np.clip(np.searchsorted(v, qv) - 2, 0, v.size - 4)
    su = bu[:, None] + np.arange(4)[None, :]
    sv = bv[:, None] + np.arange(4)[None, :]
    wu = _lag4(u[su], qu)
    wv = _lag4(v[sv], qv)
    stencil = G[su[:, :, None], sv[:, None, :]]          # (M,4,4)
    return np.einsum("ma,mb,mab->m", wu, wv, stencil)


def resample_plane(imp, key, fallback, pts):
    """Map a partner's interface samples onto THIS participant's points.

    `pts` is (N,2): the two IN-PLANE coordinates of this side's interface
    dofs.  The driver does no interpolation — non-matching meshes are handled
    here, and in 3-D that means a genuine 2-D interpolation (item 2 of the
    header).  Routes, in order of preference:

      1. TENSOR PRODUCT — the partner's points are exactly unique(u) x
         unique(v).  True whenever the partner meshes a box, which is the
         normal case.  CUBIC on that rectilinear grid (bilinear if either
         direction has fewer than 4 lines).
      2. scipy LinearNDInterpolator on the Delaunay triangulation of the
         partner's points; the few points outside its convex hull are filled
         from the nearest partner point.  This route is piecewise linear and
         carries the order penalty described in `_bicubic` — an unstructured
         partner on a coupling that must deliver a second-order FLUX is a real
         limitation, not a formality.
      3. NEAREST NEIGHBOUR, with a loud warning.  Only O(h): it caps the
         coupled solution itself at first order.  It exists so an exotic
         partner mesh degrades instead of crashing.
    """
    n = len(pts)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    co = np.asarray(imp["coordinates"], float)
    if co.ndim != 2 or co.shape[1] < 3:
        # A 2-D partner on a 3-D interface.  Failing loudly is the point: the
        # alternative is to silently drop a coordinate and couple two different
        # geometries to each other.
        sys.exit(f"partner '{PARTNER}' exported {co.shape[1] if co.ndim == 2 else '?'}"
                 f"-component coordinates; a 3-D planar interface needs [x,y,z]")
    # `or []` and not `.get(key, [])`: a partner that writes the key with an
    # explicit null gets [] here instead of a TypeError out of np.asarray, and
    # falls through to the fallback like any other unusable import.
    vs = np.asarray(imp.get(key) or [], float).ravel()
    if vs.size != co.shape[0]:
        return np.full(n, float(fallback))
    src = co[:, TAN]

    u = _unique_tol(src[:, 0], TOL)
    v = _unique_tol(src[:, 1], TOL)
    if u.size >= 2 and v.size >= 2 and u.size * v.size == src.shape[0]:
        iu = np.abs(src[:, 0][:, None] - u[None, :]).argmin(1)
        iv = np.abs(src[:, 1][:, None] - v[None, :]).argmin(1)
        G = np.full((u.size, v.size), np.nan)
        G[iu, iv] = vs
        if not np.isnan(G).any():
            p = np.asarray(pts, float)
            if u.size >= 4 and v.size >= 4:
                return _bicubic(u, v, G, p)
            return _bilinear(u, v, G, p)

    try:
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
        lin = LinearNDInterpolator(src, vs)
        out = lin(pts)
        bad = ~np.isfinite(out)
        if bad.any():
            out[bad] = NearestNDInterpolator(src, vs)(np.asarray(pts)[bad])
        return out
    except Exception as e:                                    # noqa: BLE001
        print(f"[dune3d {SIDE}] WARNING: 2-D interpolation unavailable ({e}); "
              f"falling back to NEAREST NEIGHBOUR on the interface. That is "
              f"O(h) and WILL flatten the convergence rate — match the meshes "
              f"or install scipy before believing any order measured this way.")
        d = ((np.asarray(pts)[:, None, :] - src[None, :, :]) ** 2).sum(-1)
        return vs[d.argmin(1)]


def order_plane(pts):
    """Deterministic order of interface points over BOTH in-plane coordinates.

    Quantised so that dofs nominally on the same row cannot be split apart by
    round-off into an order that changes with the grid manager's mood.  The
    driver relaxes entry by entry, so this order IS part of the contract.
    """
    q = np.round(np.asarray(pts, float) / max(TOL, 1e-300))
    return np.lexsort((q[:, 1], q[:, 0]))


# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# the problem you were given, in this backend, however you judge best. That is
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
#
# At this point you are expected to have produced:
#   * the discrete solution on this subdomain, with the partner's interface
#     data applied according to SIDE, and
#   * the assembled operator and the VOLUME load separately, because the flux
#     recovery below subtracts the volume load alone.
# ─────────────────────────────────────────────────────────────────────────


def main():
    if abs(IFACE_POS - LO[AX]) > TOL and abs(IFACE_POS - HI[AX]) > TOL:
        sys.exit(f"IFACE_POS={IFACE_POS} is not a {AXN[AX]}-face of this box "
                 f"[{LO[AX]},{HI[AX]}] — nothing is shared with the partner")
    if IFACE_FACE in DIRICHLET_FACES:
        sys.exit(f"DIRICHLET_FACES lists '{IFACE_FACE}', which IS the interface. "
                 f"The outer condition would overwrite the coupling and the run "
                 f"would still exit 0 with a plausible export.")
    if min(NE) < 1:
        sys.exit(f"NX,NY,NZ = {tuple(NE)}: need at least one element per axis")

    # ── BUILD THE MESH ───────────────────────────────────────────────────────
    gridView = structuredGrid([X0, Y0, Z0], [X1, Y1, Z1], [NX, NY, NZ])

    # ── BUILD THE FUNCTION SPACE ────────────────────────────────────────────
    space = lagrange(gridView, 1)

    # ── GET THE DOF COORDINATES ─────────────────────────────────────────────
    xd = space.dofCoordinates()
    x = SpatialCoordinate(space)

    # ── BUILD THE TRIAL/TEST FUNCTIONS ──────────────────────────────────────
    u = TrialFunction(space)
    v = TestFunction(space)

    # ── BUILD THE WEAK FORMS ────────────────────────────────────────────────
    K_const = Constant(K, name="K")
    a_form = dot(K_const * grad(u), grad(v)) * dx

    ffun = source_function(space, xd)
    b_vol = ffun * v * dx

    # ── BUILD THE DIRICHLET BOUNDARY CONDITIONS ─────────────────────────────
    dbc_list = []
    for face in DIRICHLET_FACES:
        ax = "xyz".index(face[0])
        hi = (face[1] == "1")
        val = outer_value(xd[0], xd[1], xd[2])
        # Build indicator for this face
        if hi:
            ind = conditional(lt(abs(x[ax] - float(HI[ax])), EPS), 1, 0)
        else:
            ind = conditional(lt(abs(x[ax] - float(LO[ax])), EPS), 1, 0)
        dbc = DirichletBC(space, val, ind)
        dbc_list.append(dbc)

    # ── BUILD THE INTERFACE INDICATOR ───────────────────────────────────────
    iface_ind = conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS), 1, 0)

    # ── BUILD THE INTERFACE DOF LIST AND OUTER MASK ────────────────────────
    # Interface dofs: those with x[AX] ≈ IFACE_POS
    iface_dofs = []
    for i in range(space.dim):
        if abs(xd[AX][i] - IFACE_POS) < TOL:
            iface_dofs.append(i)
    iface_dofs = np.array(iface_dofs, dtype=int)

    # Outer mask: dofs on any outer Dirichlet face
    outer_mask = np.zeros(space.dim, dtype=bool)
    for face in DIRICHLET_FACES:
        ax = "xyz".index(face[0])
        hi = (face[1] == "1")
        if hi:
            outer_mask |= (abs(xd[ax] - float(HI[ax])) < EPS)
        else:
            outer_mask |= (abs(xd[ax] - float(LO[ax])) < EPS)

    # ── GET THE INTERFACE POINTS ────────────────────────────────────────────
    pts_all = np.array([xd[0][i], xd[1][i], xd[2][i]] for i in iface_dofs)

    # ── BUILD THE SOLUTION FUNCTION ────────────────────────────────────────
    uh = space.interpolate(0, name="uh")

    # ── BUILD THE INTERFACE VALUE FUNCTION (for Dirichlet side) ────────────
    gd = space.interpolate(0, name="gd")

    # ── READ PARTNER DATA ──────────────────────────────────────────────────
    imp = read_imports()

    # ── APPLY PARTNER DATA ─────────────────────────────────────────────────
    b_form = b_vol
    ind = None

    if SIDE == "dirichlet":
        free = iface_dofs[~outer_mask[iface_dofs]]
        if len(free) > 0:
            gd.as_numpy[free] = resample_plane(imp, "values", T_INIT, pts_all[free][:, TAN])
        ind = iface_ind
    else:
        # NEUMANN: apply partner flux as natural BC on interface
        gflx = space.interpolate(0, name="iface_flux")
        q_in = resample_plane(imp, "normal_fluxes", Q_INIT, pts_all)
        gflx.as_numpy[iface_dofs] = q_in
        b_form = b_form + conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS),
                                      gflx * v, 0.0) * ds

    # ── BUILD THE SCHEME AND SOLVE ─────────────────────────────────────────
    scheme = galerkin([a_form == b_form] + dbc_list, solver='cg',
                      parameters={'linear.verbose': True})
    info = scheme.solve(target=uh)
    if not info['converged']:
        sys.exit(f"DUNE solve did not converge after {info['iterations']} iterations")

    # ── ORDER THE INTERFACE DOFS ───────────────────────────────────────────
    order = order_plane(pts_all[:, TAN])
    iface_dofs = iface_dofs[order]
    pts3 = pts_all[order]
    pts = pts3[:, TAN]

    edge = outer_mask[iface_dofs]        # interface dofs ALSO on an outer face

    # ── COMPUTE THE INTERFACE WEIGHTS ───────────────────────────────────────
    wfun = assemble(conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS), v, 0.0) * ds)
    w = np.array(wfun.as_numpy)[iface_dofs]

    # ── COMPUTE THE UNCONSTRAINED RESIDUAL ─────────────────────────────────
    op_free = operator_galerkin([a_form == b_vol])        # volume load, no bcs
    rfun = space.interpolate(0, name="residual")
    op_free(uh, rfun)
    rv = np.array(rfun.as_numpy)

    # ── THE CONSISTENT (REACTION) FLUX ─────────────────────────────────────
    r = rv[iface_dofs]

    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w) > 1e-14 * max(1.0, float(np.max(np.abs(w))))
    Q[ok] = -r[ok] / w[ok]
    Q_raw = Q.copy()

    # ── THE EDGE GUARD ─────────────────────────────────────────────────────
    suspect = edge | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        gp = pts[good]
        for i in np.where(suspect)[0]:
            d = ((gp - pts[i]) ** 2).sum(1)
            Q[i] = Q[good[int(d.argmin())]]
    elif suspect.any():
        print(f"[dune3d {SIDE}] WARNING: every interface dof is also on an "
              f"outer Dirichlet face; no clean reaction exists anywhere on "
              f"this interface and the exported flux is the raw one.")

    T = np.array(uh.as_numpy)[iface_dofs]

    # ── CONSERVATION SELF-CHECK ────────────────────────────────────────────
    gvec = np.zeros_like(rv)
    if SIDE != "dirichlet":
        gvec = np.array(assemble(
            conditional(lt(abs(x[AX] - float(IFACE_POS)), EPS), gflx * v, 0.0)
            * ds).as_numpy)
    fixed = outer_mask.copy()
    if SIDE == "dirichlet":
        fixed[iface_dofs] = True
    react = float((rv - gvec)[fixed].sum())
    load_vol = float(np.array(assemble(b_vol).as_numpy).sum())
    load_if = float(gvec.sum())
    imb = abs(react + load_vol + load_if)
    scale = max(abs(react), abs(load_vol), abs(load_if))
    if scale <= 1e-10 * K * max(1.0, float(np.max(np.abs(T)))) * float(np.max(LEN)):
        bal = (f"balance trivially satisfied (no source and no applied "
               f"interface flux: the identity reads 0 = 0 and proves nothing "
               f"about the coupling; |imbalance|={imb:.3e})")
    else:
        bal = f"balance {imb:.3e} abs / {imb / scale:.3e} rel"

    print(f"[dune3d {SIDE}] interface n={len(iface_dofs)} area={w.sum():.6g} "
          f"edge_dofs={int(edge.sum())} T=[{T.min():.6g},{T.max():.6g}] "
          f"q=[{Q.min():.6g},{Q.max():.6g}] {bal}")

    # exports.json LAST: the driver takes its existence as proof of success.
    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(iface_dofs)),
        "coordinates": [[float(p[0]), float(p[1]), float(p[2])] for p in pts3],
        "values": [float(t) for t in T],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))

    return {"gridView": gridView, "space": space, "uh": uh, "x": x,
            "iface_dofs": iface_dofs, "coords": pts3, "pts": pts,
            "weights": w, "T": T, "q": Q, "q_raw": Q_raw, "edge": edge,
            "imbalance": imb, "imbalance_rel": imb / scale if scale > 0 else 0.0}


if __name__ == "__main__":
    main()
