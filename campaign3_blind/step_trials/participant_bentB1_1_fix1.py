"""FEniCSx (dolfinx) participant for the OASiS `couple` driver.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: steady conduction  -div(K grad T) = F_SRC  on one rectangular
subdomain of a domain split by a straight interface at x = IFACE_X.
Top and bottom edges are natural (zero-flux). The non-interface x-boundary
carries a Dirichlet value T_OUTER.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import dolfinx                # the MODULE, so dolfinx.log.set_log_level(...) resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from dolfinx import fem` alone leaves `dolfinx` undefined
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
SIDE      = "neumann"       # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "left"          # the partner's `name` in your couple(...) call
X0, X1    = 0.5, 1.0        # this subdomain's x-extent
Y0, Y1    = 0.0, 0.5        # this subdomain's y-extent
IFACE_AXIS = "x"            # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                            # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                            # (they are stacked). Everything below follows from it.
IFACE_X   = 0.5             # the shared interface; X0/X1 for axis "x", Y0/Y1 for axis "y"
IFACE_SEGMENTS = ()         # EMPTY: the interface is the single straight line named above. For an
                            # interface that BENDS -- one subdomain's corner cut out of the other --
                            # list its legs in order instead, each ("x"|"y", position, from, to):
                            # [("x", 0.5, 0.0, 0.5), ("y", 0.5, 0.5, 1.0)]
K         = 2.5             # conductivity


def F_SRC(x, y):
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)

    HOW THIS ENTERS FEniCSx (your solve below has to do it): a UFL expression
    carries no NumPy ufuncs, so do NOT call this function on
    ufl.SpatialCoordinate (np.zeros_like(x) does not even raise -- it returns
    a 0-d object array and the source collapses to a constant). Interpolate it
    into a P1 Function on your space instead, f_h.interpolate(lambda X:
    F_SRC(X[0], X[1])), and integrate f_h * v * dx -- the P1 interpolant of the
    source, quadrature error O(h^2), the order of the discretisation.
    """
    return np.zeros_like(x)
T_OUTER   = 0.0             # Dirichlet value on the NON-interface x-boundary
NX, NY    = 16, 16          # this subdomain's OWN mesh; need not match the partner
T_INIT    = 310.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0             # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level; the per-level
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OASIS_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))   # a multi-level call's level keys
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

AX = 0 if IFACE_AXIS == "x" else 1         # the coordinate the interface FIXES
AL = 1 - AX                                # the coordinate that RUNS ALONG it
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)         # this subdomain, across the interface
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)       # this subdomain, along it
OUTER_X = LO if abs(IFACE_X - HI) < abs(IFACE_X - LO) else HI
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S * e_AX


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


TOL_IF = 1e-9 * max(X1 - X0, Y1 - Y0)


def arc_length(pts):
    """Distance along the interface, from its start, for each point of `pts` (an (n, 2) array).

    This is what makes a BENT interface exchangeable: the two sides meet on a curve, and a single
    coordinate is not monotone along two legs. With IFACE_SEGMENTS empty this is the coordinate along
    the single straight line, so the usual case is unchanged."""
    pts = np.atleast_2d(np.asarray(pts, float))
    if not IFACE_SEGMENTS:
        return pts[:, AL]
    s_out = np.full(len(pts), np.nan)
    base = 0.0
    for axis, pos, a, b in IFACE_SEGMENTS:
        ax = 0 if axis == "x" else 1
        al = 1 - ax
        lo, hi = (a, b) if a <= b else (b, a)
        on = ((np.abs(pts[:, ax] - pos) < TOL_IF) & (pts[:, al] >= lo - TOL_IF)
              & (pts[:, al] <= hi + TOL_IF) & np.isnan(s_out))
        s_out[on] = base + np.abs(pts[on, al] - a)
        base += abs(b - a)
    return np.where(np.isnan(s_out), 0.0, s_out)


def sample(imp, key, fallback, where):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here.

    `where` is this side's coordinate along a straight interface, or its (n, 2) interface POINTS for
    a bent one, in which case both sides are matched by distance along IFACE_SEGMENTS."""
    where = np.asarray(where, float)
    bent = where.ndim == 2
    n = len(where)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    pc = np.atleast_2d(np.asarray(imp["coordinates"], float))
    ys = arc_length(pc) if bent else pc[:, AL]
    target = arc_length(where) if bent else where
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(n, float(fallback))
    o = np.argsort(ys)
    return np.interp(target, ys[o], vs[o])


imp = read_imports()

# ── SETUP LOGGING BEFORE MESH CREATION ─────────────────────────────────────
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── BUILD MESH ─────────────────────────────────────────────────────────────
domain = dmesh.create_unit_square(MPI.COMM_WORLD, NX, NY)
domain.geometry.x[:, 0] *= (X1 - X0)
domain.geometry.x[:, 0] += X0
domain.geometry.x[:, 1] *= (Y1 - Y0)
domain.geometry.x[:, 1] += Y0

# ── FUNCTION SPACE ─────────────────────────────────────────────────────────
V = fem.functionspace(domain, ("Lagrange", 1))

# ── MARK BOUNDARY FACETS ───────────────────────────────────────────────────
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, fdim - 1)

# Define markers for different boundary types
corner_tol = 1e-10

def outer_boundary(x):
    """Bottom (y=0) and right (x=1) edges - Dirichlet u=0"""
    tol = 1e-14
    return np.logical_or(np.abs(x[1] - Y0) < tol, np.abs(x[0] - X1) < tol)

def interface_left(x):
    """Left edge x=0.5, but exclude corner at (0.5, 0)"""
    tol = 1e-14
    return np.abs(x[0] - X0) < tol

def interface_top(x):
    """Top edge y=0.5, but exclude corner at (1, 0.5)"""
    tol = 1e-14
    return np.abs(x[1] - Y1) < tol

# Get all boundary facets
facets_all = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.ones_like(x[0], dtype=bool))

# Separate into groups
facets_outer = dmesh.locate_entities_boundary(domain, fdim, outer_boundary)
facets_iface_left = dmesh.locate_entities_boundary(domain, fdim, interface_left)
facets_iface_top = dmesh.locate_entities_boundary(domain, fdim, interface_top)

# Remove corner facets from interface (they belong to outer boundary)
facet_nodes = domain.topology.connectivity(fdim, 0).links
facet_coords = domain.geometry.x

def remove_corners(facet_indices, coords_global):
    """Remove facets touching corners (0.5, 0) or (1, 0.5)"""
    keep = []
    for i, fidx in enumerate(facet_indices):
        fnode = facet_nodes[fidx]
        fcoord = coords_global[fnode]
        # Check if facet touches corner
        touches_corner = False
        for fc in fcoord:
            if (abs(fc[0] - X0) < corner_tol and abs(fc[1] - Y0) < corner_tol):
                touches_corner = True
                break
            if (abs(fc[0] - X1) < corner_tol and abs(fc[1] - Y1) < corner_tol):
                touches_corner = True
                break
        if not touches_corner:
            keep.append(i)
    return facet_indices[keep]

# Combine interface facets
facets_iface_left = remove_corners(facets_iface_left, facet_coords)
facets_iface_top = remove_corners(facets_iface_top, facet_coords)
facets_interface = np.concatenate([facets_iface_left, facets_iface_top]).astype(np.int32)
facets_interface = np.unique(facets_interface)

# Create meshtags
ntags = 3  # 0: exterior, 1: interface, 2: outer Dirichlet
tags_array = np.zeros(len(facets_all), dtype=np.int32)
for f in facets_interface:
    idx = np.where(facets_all == f)[0][0]
    tags_array[idx] = 1
for f in facets_outer:
    idx = np.where(facets_all == f)[0][0]
    tags_array[idx] = 2

meshtags = dmesh.meshtags(domain, fdim, facets_all, tags_array)
ds = ufl.Measure("ds", domain, subdomain_data=meshtags)

# ── SELECT DOFS ────────────────────────────────────────────────────────────
outer_dofs = fem.locate_dofs_topological(V, fdim, facets_outer)
iface_dofs = fem.locate_dofs_topological(V, fdim, facets_interface)

# Sort for consistency
outer_dofs = np.sort(outer_dofs)
iface_dofs = np.sort(iface_dofs)

# ── DIRICHLET BC ON OUTER BOUNDARY ─────────────────────────────────────────
bc_outer = fem.dirichletbc(T_OUTER, outer_dofs, V)

# ── INTERPOLATE IMPORTED FLUX AS NEUMANN DATA ──────────────────────────────
# Get interface DOF coordinates
iface_coords_raw = V.tabulate_dof_coordinates()[iface_dofs]

# For bent interface, we need (x, y) pairs
y_if_bent = iface_coords_raw.copy()  # shape (n, 2)

# Sample the imported flux onto our interface DOFs
if imp is not None:
    imp_flux = sample(imp, "normal_fluxes", Q_INIT, y_if_bent)
else:
    imp_flux = np.full(len(iface_dofs), Q_INIT)

# Create a Function to hold the Neumann data
g_func = fem.Function(V)
g_func.x.array[:] = 0.0
g_func.x.array[iface_dofs] = imp_flux

# ── INTERPOLATE SOURCE TERM ────────────────────────────────────────────────
f_src = fem.Function(V)
f_src.interpolate(lambda X: F_SRC(X[0], X[1]))

# ── WEAK FORM ──────────────────────────────────────────────────────────────
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Bilinear form: integral(K grad u . grad v dx)
a = K * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Linear form: integral(f v dx) - integral(g v ds) on interface
# Note: natural BC enters as -int(g v ds) in standard form
L_vol = f_src * v * ufl.dx

# ── LINEAR PROBLEM ─────────────────────────────────────────────────────────
problem = LinearProblem(a, L_vol, bcs=[bc_outer],
                        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                        petsc_options_prefix='run')

uh = problem.solve()

# ── RECOVER CONSISTENT INTERFACE FLUX ─────────────────────────────────────
# Assemble matrix and volume RHS only (no boundary lifting)
Amat = _fp.assemble_matrix(fem.form(a))
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))
bvec.ghostUpdate()

# Compute residual r = A*u - b_vol
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

# Weight vector w_i = int_Gamma phi_i ds
wvec = _fp.assemble_vector(fem.form(ufl.one() * ds(1)))  # ds(1) is interface
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

# Consistent flux: q_i = -r_i / w_i
Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# Handle suspect nodes (corners or zero weight)
suspect = ~ok
good = np.where(ok)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Interface values
T = uh.x.array[iface_dofs]

print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
_chk_vals = np.asarray(T, float).ravel()
_chk_flux = np.asarray(Q, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was "
                     "exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
        and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 "
                     "against a nonzero imported flux: the imported load never "
                     "entered the assembled system (the facet term / boundary "
                     "condition that integrates it is missing). Fix the "
                     "application; do not couple on")
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

# PER-LEVEL PERSISTENCE: this level's whole field and its interface trace and
# flux, named by LEVEL, never overwritten by the next level (exports.json is).
xy = V.tabulate_dof_coordinates()
with open(f"field_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(xy[:, :2], uh.x.array):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
with open(f"interface_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    for (_px, _py), _t, _q in zip(y_if_bent, T, Q):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_t):.11e},{float(_q):.11e}\n")

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": y_if_bent.tolist(),
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
