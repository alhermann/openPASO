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
SIDE      = "dirichlet"
PARTNER   = "right"
X0, X1    = 0.0, 0.6
Y0, Y1    = 0.0, 1.0
IFACE_X   = 0.6
K         = 1.0


def F_SRC(x, y):
    return np.pi**2 * x * np.sin(np.pi * y)

T_OUTER   = 0.0
NX, NY    = 6, 10
T_INIT    = 0.0
Q_INIT    = 0.0
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)
ORDER = 1

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0
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
geo = SplineGeometry()
p0 = geo.AddPoint((X0, Y0))
p1 = geo.AddPoint((X1, Y0))
p2 = geo.AddPoint((X1, Y1))
p3 = geo.AddPoint((X0, Y1))
geo.AddLine((p0, p1), bcs=["outer"])
geo.AddLine((p1, p2), bcs=["interface"])
geo.AddLine((p2, p3), bcs=["outer"])
geo.AddLine((p3, p0), bcs=["outer"])
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = H1(mesh, order=ORDER)
u, v = fes.TnT()

a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
a.Assemble()

f_vol = LinearForm(fes)
src_gf = GridFunction(fes)
for i, vert in enumerate(mesh.vertices):
    src_gf.vec[i] = F_SRC(vert.point[0], vert.point[1])
f_vol += src_gf * v * dx
f_vol.Assemble()
f = f_vol  # alias for contract compatibility

iface_dofs = []
iface_v = []
y_if = []
for bd in mesh.Boundaries("interface"):
    for i in bd.vertices:
        dof = fes.GetVertexDof(i)
        iface_dofs.append(dof)
        iface_v.append(mesh.vertices[i])
        y_if.append(mesh.vertices[i].point[1])

sort_idx = np.argsort(y_if)
y_if = [y_if[i] for i in sort_idx]
iface_dofs = [iface_dofs[i] for i in sort_idx]
iface_v = [iface_v[i] for i in sort_idx]

outer_dofs = []
for bd in mesh.Boundaries("outer"):
    for i in bd.vertices:
        outer_dofs.append(fes.GetVertexDof(i))
outer_dofs = sorted(list(set(outer_dofs)))

gfu = GridFunction(fes)
# ─────────────────────────────────────────────────────────────────────────

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
    f += gfun * v * ds("interface")

with TaskManager():
# ─────────────────────────────────────────────────────────────────────────
    b = f.vec.CreateVector()
    b.data = f.vec
    for d in outer_dofs:
        b[d] = T_OUTER
        for j in range(len(b)):
            if j != d:
                b[j] -= a.mat[d, j] * T_OUTER
    for d in iface_dofs:
        b[d] = gfu.vec[d]
        for j in range(len(b)):
            if j != d:
                b[j] -= a.mat[d, j] * gfu.vec[d]
    free_dofs = fes.FreeDofs() - set(outer_dofs + iface_dofs)
    a_inv = a.mat.Inverse(free_dofs)
    gfu.vec.data = a_inv * b
# ─────────────────────────────────────────────────────────────────────────

    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f.vec
    fw = LinearForm(fes)
    fw += v * ds("interface")
    fw.Assemble()

    r_if = np.array([rvec[int(d)] for d in iface_dofs], float)
    w_if = np.array([fw.vec[int(d)] for d in iface_dofs], float)
    Q = np.zeros(len(iface_dofs))
    ok = np.abs(w_if) > 1e-14
    Q[ok] = -r_if[ok] / w_if[ok]

    suspect = np.isin(iface_dofs, outer_dofs) | ~ok
    good = np.where(~suspect)[0]
    if len(good):
        for i in np.where(suspect)[0]:
            Q[i] = Q[good[np.argmin(np.abs(good - i))]]

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
