"""NGSolve participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import ngsolve                # the MODULE, so ngsolve.ngsglobals.msg_level = 3 resolves:
                              # a run log that must carry this code's own output needs it,
                              # and `from ngsolve import ...` alone leaves `ngsolve` undefined
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
T_INIT    = 0.0            # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

MAXH  = min((X1 - X0) / NX, (Y1 - Y0) / NY)   # netgen's scalar mesh size
ORDER = 1                                     # H1 order (nodal == vertex dofs)

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

ngsolve.ngsglobals.msg_level = 3

# ── BUILD GEOMETRY AND MESH ───────────────────────────────────────────────
geo = SplineGeometry()
# AddRectangle takes ((x0,y0),(x1,y1)) and bcs=(bottom,right,top,left)
geo.AddRectangle((X0, Y0), (X1, Y1), bcs=('bottom', 'interface', 'top', 'outer'))
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))
# ──────────────────────────────────────────────────────────────────────────

fes = H1(mesh, order=ORDER, dirichlet='outer')
gfu = GridFunction(fes)
u, v = fes.TnT()

# ── ASSEMBLE THE BILINEAR FORM (stiffness) ────────────────────────────────
a = BilinearForm(fes)
a += K * grad(u) * grad(v) * dx
a.Assemble()
# ──────────────────────────────────────────────────────────────────────────

# ── SAMPLE SOURCE AT VERTICES INTO A P1 GridFunction ──────────────────────
pts = np.array([v.point for v in mesh.vertices])
vdof = np.array([fes.GetDofNrs(NodeId(VERTEX, v.nr))[0] for v in mesh.vertices])
f_vol = GridFunction(fes)
f_vol.vec[:] = 0.0
f_vol.vec.FV().NumPy()[vdof] = F_SRC(pts[:, 0], pts[:, 1])
# ──────────────────────────────────────────────────────────────────────────

# ── LINEAR FORM FOR VOLUME SOURCE ────────────────────────────────────────
f = LinearForm(fes)
f += f_vol * v * dx
f.Assemble()
# ──────────────────────────────────────────────────────────────────────────

# ── INTERFACE DOFS AND COORDINATES ────────────────────────────────────────
iface_v = []
for v in mesh.vertices:
    vp = v.point
    if np.abs(vp[0] - IFACE_X) < TOL and np.abs(vp[1] - Y0) > TOL and np.abs(vp[1] - Y1) > TOL:
        iface_v.append(v)
iface_v = sorted(iface_v, key=lambda v: v.point[1])
iface_dofs = np.array([fes.GetDofNrs(NodeId(VERTEX, v.nr))[0] for v in iface_v])
y_if = np.array([v.point[1] for v in iface_v])
# ──────────────────────────────────────────────────────────────────────────

# ── OUTER DOFS (for flux correction at corners) ──────────────────────────
outer_mask = mesh.Boundaries('outer').Mask()
outer_v = set()
for el in mesh.Elements():
    if outer_mask[el.nr]:
        for nv in el.Vertices():
            outer_v.add(nv)
outer_dofs = np.array([fes.GetDofNrs(NodeId(VERTEX, v.nr))[0] for v in outer_v])
# ──────────────────────────────────────────────────────────────────────────

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

with TaskManager():
    # Solve the system
    gfu.vec.Data()[:] = 0.0
    free_dofs = fes.FreeDofs()
    a.mat.Inverse(free_dofs, inverse='sparsecholesky')
    gfu.vec = a.mat.Inverse(free_dofs, inverse='sparsecholesky') * f.vec

    # Outward normal flux density q = -(k grad T).n on the interface.
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f.vec    # r = A u_h - b_vol, no bc
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
