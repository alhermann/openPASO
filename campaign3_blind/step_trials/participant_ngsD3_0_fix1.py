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
                     NodeId, TaskManager, ds, dx, grad)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
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
NX, NY    = 6, 10         # this subdomain's own mesh (level 1)
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


imp = read_imports()

# ── BUILD MESH AND SPACE ─
geo = SplineGeometry()
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=('bottom', 'right', 'top', 'left'))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = H1(mesh, order=ORDER, dirichlet='left|bottom|top')
v, u = fes.TnT()

# ── ASSEMBLE OPERATOR ─
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
a.Assemble()

# ── ASSEMBLE LOAD (VOLUME SOURCE) ─
# Sample source at vertices, interpolate to P1 GridFunction
verts = mesh.vertices
vx = np.array([v.point[0] for v in verts], float)
vy = np.array([v.point[1] for v in verts], float)
src_vals = F_SRC(vx, vy)

# Create a GridFunction to hold the source
src_gf = GridFunction(fes)
for i, vrt in enumerate(verts):
    src_gf.vec[int(vrt.ndof[0])] = src_vals[i]

f_vol = LinearForm(fes)
f_vol += src_gf * v * dx
f_vol.Assemble()

# ── INTERFACE SETUP ─
iface_v = [v for v in mesh.vertices if abs(v.point[0] - IFACE_X) < TOL]
iface_v = sorted(iface_v, key=lambda v: v.point[1])
y_if = np.array([v.point[1] for v in iface_v], float)

iface_dofs = [v.ndof[0] for v in iface_v]
outer_dofs = [v.ndof[0] for v in mesh.vertices if v.boundary_number() in ('left', 'bottom', 'top')]

gfu = GridFunction(fes)

# ── APPLY IMPORTED INTERFACE VALUES ─
if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    for d, t in zip(iface_dofs, T_if):
        gfu.vec[int(d)] = float(t)
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gfun = GridFunction(fes)
    gfun.vec[:] = 0.0
    for d, q in zip(iface_dofs, q_if):
        gfun.vec[int(d)] = float(q)
    f_vol += gfun * v * ds("right")

with TaskManager():
    a.mat.Inverse("lu")
    gfu.vec.data = a.mat * f_vol.vec

    # Outward normal flux density q = -(k grad T).n on the interface.
    rvec = f_vol.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec
    fw = LinearForm(fes)
    fw += v * ds("right")
    fw.Assemble()

    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]

    # Handle corner nodes
    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

    # ── EXPORT SELF-CHECK ─
    _chk_vals = np.asarray([gfu.vec[int(d)] for d in iface_dofs], float).ravel()
    _chk_flux = np.asarray(Q, float).ravel()
    if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
        raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes")
    _chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
                if Path("imports.json").is_file() else {})
    _chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                                for _d in _chk_imp.values()])
                if _chk_imp else np.zeros(0))
    if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 \
            and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
        raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0")
    if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
            and np.array_equal(_chk_flux, -_chk_qin):
        raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array negated")

    Path("exports.json").write_text(json.dumps({
        "field_name": "temperature",
        "n_points": int(len(iface_v)),
        "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
        "values": [float(gfu.vec[int(d)]) for d in iface_dofs],
        "normal_fluxes": [float(q) for q in Q],
    }, indent=2))
