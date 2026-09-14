"""NGSolve participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (VERTEX, BilinearForm, GridFunction, H1, LinearForm, Mesh,
                     NodeId, TaskManager, ds, dx, grad, CoefficientFunction)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
SIDE      = "dirichlet"   # "dirichlet" | "neumann"
PARTNER   = "right"       # name of the partner participant in couple(...)
X0, X1    = 0.0, 0.6      # this subdomain
Y0, Y1    = 0.0, 1.0
IFACE_X   = 0.6           # shared interface (must be X0 or X1)
K         = 1.0           # conductivity


def F_SRC(x, y):
    """Volumetric source, as a function of position."""
    return np.pi**2 * x * np.sin(np.pi * y)

T_OUTER   = 0.0           # Dirichlet value on the NON-interface x-boundary
NX, NY    = 6, 10         # this subdomain's own mesh (netgen maxh derived below)
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None


def sample(imp, key, fallback, y):
    """Interpolate the partner's samples onto this participant's y-coordinates."""
    if not imp or not imp.get("coordinates"):
        return np.full(len(y), float(fallback))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(len(y), float(fallback))
    o = np.argsort(ys)
    return np.interp(y, ys[o], vs[o])


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

# Build mesh
geo = SplineGeometry()
geo.AddRect(pleft=(X0, Y0), pright=(X1, Y1), maxh=MAXH)
geo.SetMeshSize(MAXH)
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

# Create function space
fes = H1(mesh, order=ORDER)

# Define test and trial functions
u, v = fes.TnT()

# Build bilinear form: a(u,v) = int k*grad(u)*grad(v) dx
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
a.Assemble()

# Build volume source linear form
# Sample F_SRC at mesh vertices and interpolate to P1 GridFunction
vertices = np.array([v.point for v in mesh.vertices])
vx = vertices[:, 0]
vy = vertices[:, 1]
f_vals = F_SRC(vx, vy)

# Create source as GridFunction (P1 interpolant)
f_gf = GridFunction(fes)
f_gf.vec[:] = f_vals
f_vol = LinearForm(fes)
f_vol += f_gf * v * dx
f_vol.Assemble()

# Define boundary labels
mesh.SetMeshSize(MAXH)  # ensure consistent labeling
geo.SetMeshSize(MAXH)

# Rebuild mesh with boundary labels
geo2 = SplineGeometry()
geo2.AddRect(pleft=(X0, Y0), pright=(X1, Y1), maxh=MAXH)
geo2.SetMeshSize(MAXH)
mesh2 = Mesh(geo2.GenerateMesh(maxh=MAXH))

# Label boundaries: left=x=X0, right=x=X1, bottom=y=Y0, top=y=Y1
# Use boundary conditions directly via Dirichlet markers
fes2 = H1(mesh2, order=ORDER)
u2, v2 = fes2.TnT()

# Rebuild forms on the properly labeled mesh
a2 = BilinearForm(fes2)
a2 += K * grad(u2) * grad(v2) * dx
a2.Assemble()

f_gf2 = GridFunction(fes2)
vertices2 = np.array([v.point for v in mesh2.vertices])
f_gf2.vec[:] = F_SRC(vertices2[:, 0], vertices2[:, 1])
f_vol2 = LinearForm(fes2)
f_vol2 += f_gf2 * v2 * dx
f_vol2.Assemble()

# Use the second mesh
mesh = mesh2
fes = fes2
a = a2
f_vol = f_vol2
u, v = u2, v2

# Get interface and outer boundary dofs
# Interface is at x = IFACE_X (which is X1 = 0.6, the right boundary)
# Outer Dirichlet is at x = X0 = 0.0 (left), y = Y0 = 0.0 (bottom), y = Y1 = 1.0 (top)

# Find interface vertices (at x = IFACE_X)
iface_v = []
for v in mesh.vertices:
    if abs(v.point.x - IFACE_X) < TOL:
        iface_v.append(v)

# Find outer boundary vertices (x = X0, y = Y0, or y = Y1)
outer_v = []
for v in mesh.vertices:
    if abs(v.point.x - X0) < TOL or abs(v.point.y - Y0) < TOL or abs(v.point.y - Y1) < TOL:
        if abs(v.point.x - IFACE_X) > TOL:  # exclude interface corners
            outer_v.append(v)

# Get interface y-coordinates (sorted)
y_if = np.array([v.point.y for v in iface_v])
iface_dofs = np.array([v.Dofs()[0] for v in iface_v])
outer_dofs = np.array([v.Dofs()[0] for v in outer_v])

# Sort interface by y-coordinate
sort_idx = np.argsort(y_if)
y_if = y_if[sort_idx]
iface_dofs = iface_dofs[sort_idx]

# Read partner's interface values
imp = read_imports()

# Create solution GridFunction
gfu = GridFunction(fes)
gfu.vec[:] = 0.0

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        gfu.vec[int(d)] = float(t)
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)               # P1 trace of the partner's samples
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f += gfun * v * ds("interface")        # APPLY the partner's number unchanged

# Apply outer Dirichlet boundary conditions
# For x = X0 (left), y = Y0 (bottom), y = Y1 (top)
# We need to set these in the linear system

# Build the full linear system without BCs first
A = a.mat
b = f_vol.vec

# Apply Dirichlet BCs on outer boundaries
for v in outer_v:
    d = int(v.Dofs()[0])
    # Zero out row and column
    A.SetRow(d, 0)
    A.SetCol(d, 0)
    A.Set(d, d, 1)
    b.Set(d, T_OUTER)

# Solve
with TaskManager():
    from ngsolve import CG
    gfu.vec.data = CG(A, b, tol=1e-10, maxiter=1000)

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

    # Outward normal flux density q = -(k grad T).n on the interface.
    #
    # ... (explanation continues in the annotated block)
    rvec = f_vol.vec.CreateVector()
    rvec.data = A * gfu.vec - f_vol.vec    # r = A u_h - b_vol, no bc
    fw = LinearForm(fes)
    fw += v * ds("interface")                  # w_i = int_Gamma phi_i ds
    fw.Assemble()

    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]

    # An interface node that ALSO lies on the outer Dirichlet boundary
    # carries the OUTER reaction as well, so its residual is not this
    # ... (explanation continues in the annotated block)
    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
# ... (explanation continues in the annotated block)
_chk_vals = np.asarray([gfu.vec[int(d)] for d in iface_dofs], float).ravel()
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

Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_v)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(gfu.vec[int(d)]) for d in iface_dofs],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     a
#     d
#     f
#     f_vol
#     fes
#     gfu
#     i
#     iface_dofs
#     iface_v
#     outer_dofs
#     v
#     y_if
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
