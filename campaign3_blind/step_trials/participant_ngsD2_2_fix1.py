"""NGSolve participant for the OASiS `couple` driver.

Steady diffusion  -div(k grad u) = f  on one rectangular subdomain.
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
#    Replace ALL of them with your problem's geometry, material and BCs.
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


imp = read_imports()

# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
# ─────────────────────────────────────────────────────────────────────────

geo = SplineGeometry()
p0 = geo.AddPoint(X0, Y0)
p1 = geo.AddPoint(X1, Y0)
p2 = geo.AddPoint(X1, Y1)
p3 = geo.AddPoint(X0, Y1)
geo.Append("spline {%d, %d, %d, %d}" % (p0, p1, p2, p3), bc="domain", maxh=MAXH)
geo.Append("line {%d, %d}" % (p0, p1), bc="bottom")
geo.Append("line {%d, %d}" % (p1, p2), bc="interface")
geo.Append("line {%d, %d}" % (p2, p3), bc="top")
geo.Append("line {%d, %d}" % (p3, p0), bc="left")
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = H1(mesh, order=ORDER)
u, v = fes.TnT()

a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
a.Assemble()

# Source term as P1 GridFunction (sampled at vertices)
pts = np.array([vert.point for vert in mesh.vertices])
vx, vy = pts[:, 0], pts[:, 1]
f_gf = GridFunction(fes)
f_gf.vec[:] = 0.0
vertex_dofs = np.array([mesh.GetVDof(i) for i in range(mesh.vertices.NPts())])
for i, vd in enumerate(vertex_dofs):
    f_gf.vec[int(vd)] = F_SRC(vx[i], vy[i])

f_vol = LinearForm(fes)
f_vol += f_gf * v * dx
f_vol.Assemble()
f = f_vol  # alias for contract compatibility

# Identify interface and outer boundary DOFs
iface_dofs = []
outer_dofs = []
iface_vs = []
for i, vert in enumerate(mesh.vertices):
    pt = vert.point
    ndof = mesh.GetVDof(i)
    if abs(pt[0] - IFACE_X) < TOL:
        iface_dofs.append(ndof)
        iface_vs.append(i)
    elif abs(pt[0] - X0) < TOL or abs(pt[1] - Y0) < TOL or abs(pt[1] - Y1) < TOL:
        outer_dofs.append(ndof)
iface_dofs = np.array(iface_dofs)
outer_dofs = np.array(outer_dofs)
iface_v = [mesh.vertices[i] for i in iface_vs]
y_if = np.array([vert.point[1] for vert in iface_v])

i = 0  # contract variable placeholder
d = 0  # contract variable placeholder

with TaskManager():
    gfu = GridFunction(fes)
    
    # Prepare BC vector
    bc_vec = fes.Vector()
    bc_vec[:] = 0.0
    for d in outer_dofs:
        bc_vec[int(d)] = T_OUTER
        
    if SIDE == "dirichlet":
        T_if = sample(imp, "values", T_INIT, y_if)
        for d, t in zip(iface_dofs, T_if):
            bc_vec[int(d)] = float(t)
    else:
        q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
        gfun = GridFunction(fes)               # P1 trace of the partner's samples
        gfun.vec[:] = 0.0
        for d, q in zip(iface_dofs, q_if):
            gfun.vec[int(d)] = float(q)
        f += gfun * v * ds("interface")        # APPLY the partner's number unchanged
        f.Assemble()
        f_vol = f                              # update volume+boundary load for Neumann

    # Solve reduced system
    free_dofs = fes.FreeDofs()
    gfu.vec.data = a.mat.Inverse(free_dofs, inverse="cholesky") * (f_vol.vec - a.mat * bc_vec) + bc_vec

    # ─────────────────────────────────────────────────────────────────────────
    # OUTWARD FLUX RECOVERY
    # ─────────────────────────────────────────────────────────────────────────
    rvec = f_vol.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec    # r = A u_h - b_vol, no bc
    fw = LinearForm(fes)
    fw += v * ds("interface")                  # w_i = int_Gamma phi_i ds
    fw.Assemble()

    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]

    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for idx in np.where(suspect)[0]:
            Q[idx] = Q[good[np.argmin(np.abs(good - idx))]]

    # ── EXPORT SELF-CHECK ─ keep this block.
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
