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
SIDE      = "neumann"     # this copy takes the OTHER side
PARTNER   = "left"        # the name you gave the first participant
X0, X1    = 0.5, 1.0      # the OTHER subdomain: starts where the first ends
Y0, Y1    = 0.0, 0.5      # this subdomain's y-extent
IFACE_AXIS = "x"          # WHICH straight line the interface is: "x" -> the line x = IFACE_X
                          # (the subdomains sit side by side) | "y" -> the line y = IFACE_X
                          # (they are stacked). Everything below follows from it.
IFACE_X   = 0.5           # the shared interface; X0/X1 for axis "x", Y0/Y1 for axis "y"
IFACE_SEGMENTS = (("x", 0.5, 0.0, 0.5), ("y", 0.5, 0.5, 1.0))  # bent interface: two legs
K         = 2.5           # conductivity


def F_SRC(x, y):
    """Volumetric source, as a function of position."""
    return np.zeros_like(x)
T_OUTER   = 0.0           # Dirichlet value on the outer boundary
NX, NY    = 16, 16        # this subdomain's OWN mesh
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
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

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# ... (explanation continues in the annotated block)
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# ... (explanation continues in the annotated block)

# Set up logging
dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# Create the mesh - unit square then scale
domain = dmesh.create_unit_square(MPI.COMM_WORLD, NX, NY)
# Scale the mesh to match the subdomain (0.5, 1) x (0, 0.5)
x_coords = domain.geometry.x[:, 0]
y_coords = domain.geometry.x[:, 1]
domain.geometry.x[:, 0] = X0 + (X1 - X0) * x_coords
domain.geometry.x[:, 1] = Y0 + (Y1 - Y0) * y_coords

# Create function space
V = fem.functionspace(domain, ("Lagrange", 1))

# Define the weak form
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Bilinear form
a = K * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx

# Linear form (volume source)
f = F_SRC(domain.geometry.x[:, 0], domain.geometry.x[:, 1])
f_func = fem.Function(V)
f_func.interpolate(lambda X: F_SRC(X[0], X[1]))
L_vol = ufl.inner(f_func, v) * ufl.dx

# Identify interface facets
# Interface consists of two legs:
# Leg 1: x = 0.5 for 0 < y < 0.5 (left edge, excluding corners at y=0 and y=0.5)
# Leg 2: y = 0.5 for 0.5 < x < 1 (top edge, excluding corners at x=0.5 and x=1)
# Exclude corners where interface meets outer boundary

def interface_marker_facet(x):
    """Mark interface facets (both legs, excluding corners at outer boundary)."""
    tol = TOL_IF
    # Leg 1: x = 0.5, 0 < y < 0.5 (excluding corners at y=0 and y=0.5)
    leg1 = (np.abs(x[0] - 0.5) < tol) & (x[1] > tol) & (x[1] < 0.5 - tol)
    # Leg 2: y = 0.5, 0.5 < x < 1 (excluding corners at x=0.5 and x=1)
    leg2 = (np.abs(x[1] - 0.5) < tol) & (x[0] > 0.5 + tol) & (x[0] < 1.0 - tol)
    return leg1 | leg2

# Identify outer boundary facets
def outer_boundary_marker_facet(x):
    """Mark outer boundary facets (y = 0 and x = 1, including corners)."""
    tol = TOL_IF
    # Bottom: y = 0
    bottom = x[1] < tol
    # Right: x = 1
    right = x[0] > 1.0 - tol
    return bottom | right

# Create meshtags for interface
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)
interface_facets = dmesh.locate_entities_boundary(domain, fdim, interface_marker_facet)
interface_facets = np.sort(interface_facets)
interface_tags = dmesh.meshtags(domain, fdim, interface_facets, np.ones(len(interface_facets), dtype=np.int32))

# Create meshtags for outer boundary
outer_facets = dmesh.locate_entities_boundary(domain, fdim, outer_boundary_marker_facet)
outer_facets = np.sort(outer_facets)
outer_tags = dmesh.meshtags(domain, fdim, outer_facets, np.ones(len(outer_facets), dtype=np.int32))

# Define boundary measures
ds = ufl.Measure("ds", domain=domain, subdomain_data=interface_tags)
ds_outer = ufl.Measure("ds", domain=domain, subdomain_data=outer_tags)

# Apply Dirichlet boundary conditions on outer boundary
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)
bc = fem.dirichletbc(ufl.Constant(V, 0.0), outer_dofs, V)

# Apply Neumann boundary condition from imported flux on interface
# Sample the imported flux at interface points
imp_flux = sample(imp, "normal_fluxes", Q_INIT, domain.geometry.x[interface_facets[:, 0]])

# Interpolate the flux into a function on the interface
flux_func = fem.Function(V)
flux_func.x.array[:] = 0.0

# Get interface DOFs
iface_dofs = fem.locate_dofs_topological(V, fdim, interface_facets)
flux_func.x.array[iface_dofs] = imp_flux

# Linear form with Neumann boundary condition
L = L_vol + ufl.inner(flux_func, v) * ds(1)

# Solve the linear problem
p = LinearProblem(a, L, bcs=[bc],
                  petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                  petsc_options_prefix='run')
uh = p.solve()

# Get the interface values
T = uh.x.array[iface_dofs]

# Compute the consistent flux recovery
# Assemble the residual r = A u_h - b_vol (no boundary conditions applied)
Amat = _fp.assemble_matrix(fem.form(a))
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

# Assemble the interface weight w_i = int_Gamma phi_i ds
wvec = _fp.assemble_vector(fem.form(ufl.Constant(V, 1.0) * ds(1)))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

# Compute the consistent flux q_i = -r_i / w_i
Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# Handle suspect nodes (where interface meets outer boundary)
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Get interface point coordinates
y_if = np.array([domain.geometry.x[iface_dofs[i], 1] for i in range(len(iface_dofs))])
x_if = np.array([domain.geometry.x[iface_dofs[i], 0] for i in range(len(iface_dofs))])

# Get all DOF coordinates for field export
xy = V.tabulate_dof_coordinates()

print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# EXPORT SELF-CHECK
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

# PER-LEVEL PERSISTENCE
with open(f"field_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(xy[:, :2], uh.x.array):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
with open(f"interface_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    _pts_if = np.column_stack([x_if, y_if])
    for (_px, _py), _t, _q in zip(_pts_if, T, Q):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_t):.11e},{float(_q):.11e}\n")
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": _pts_if.tolist(),
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
