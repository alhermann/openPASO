"""scikit-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm,
                   MeshTri, condense, solve)
from skfem.helpers import dot, grad

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

@LinearForm
def flux_load(v, w):
    return w["g"] * v


@LinearForm
def unit_load(v, w):
    return 1.0 * v


# ── SOLVE ITSELF ─
p = np.zeros((2, (NX+1)*(NY+1)))
t = np.zeros((3, NX*NY*2), dtype=int)
idx = 0
for j in range(NY+1):
    for i in range(NX+1):
        p[0, idx] = X0 + i * (X1 - X0) / NX
        p[1, idx] = Y0 + j * (Y1 - Y0) / NY
        idx += 1
idx = 0
for j in range(NY):
    for i in range(NX):
        tl = j*(NX+1) + i
        tr = tl + 1
        bl = tl + NX + 1
        br = bl + 1
        t[:, idx] = [tl, bl, tr]
        idx += 1
        t[:, idx] = [tr, bl, br]
        idx += 1
mesh = MeshTri(p, t)

basis = Basis(mesh, ElementTriP1())

@BilinearForm
def a(u, v, w):
    return K * dot(grad(u), grad(v))

@LinearForm
def l(v, w):
    return F_SRC(w.x[0], w.x[1]) * v

A = a.assemble(basis)
b = l.assemble(basis)
b_vol = b.copy()

outer_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], X0, atol=1e-6) | 
              np.isclose(x[1], Y0, atol=1e-6) | 
              np.isclose(x[1], Y1, atol=1e-6))
iface_facets = mesh.facets_satisfying(
    lambda x: np.isclose(x[0], IFACE_X, atol=1e-6) & 
              (x[1] > Y0 + 1e-6) & (x[1] < Y1 - 1e-6), 
    boundaries_only=False)

outer_dofs = basis.get_dofs(outer_facets).flatten()
iface_dofs = basis.get_dofs(iface_facets).flatten()
y_if = mesh.p[1, iface_dofs]
fbasis = FacetBasis(mesh, ElementTriP1(), facets=iface_facets)

D = np.concatenate([outer_dofs, iface_dofs])
x0 = basis.zeros()

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    x0[iface_dofs] = T_if
    A_c, b_c = condense(A, b, D=D, x=x0)
    sol = solve(A_c, b_c)
    sol[iface_dofs] = T_if
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()
    gnod[iface_dofs] = q_if
    b_flux = flux_load.assemble(fbasis, g=fbasis.interpolate(gnod))
    sol = solve(*condense(A, b + b_flux, D=outer_dofs))

# Outward normal flux density q = -(k grad T).n on the interface.
r = A @ sol - b_vol
wgt = unit_load.assemble(fbasis)

Q = np.zeros(len(iface_dofs))
ok = np.abs(wgt[iface_dofs]) > 1e-14
Q[ok] = -r[iface_dofs][ok] / wgt[iface_dofs][ok]

suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

_chk_vals = np.asarray(sol[iface_dofs], float).ravel()
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
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(yy)] for yy in y_if],
    "values": [float(t) for t in sol[iface_dofs]],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
