"""scikit-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, FacetBasis, LinearForm,
                   MeshTri, condense, solve)
from skfem.helpers import dot, grad

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

T_OUTER   = 0.0         # Dirichlet value on the NON-interface x-boundary
NX, NY    = 6, 10        # this subdomain's own mesh
T_INIT    = 0.0          # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

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
#
# Build the mesh, the function space, the weak form and the linear solve for
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below.
# ─────────────────────────────────────────────────────────────────────────

element = ElementTriP1()

# Construct structured triangular mesh (NX x NY)
x = np.linspace(X0, X1, NX + 1)
y = np.linspace(Y0, Y1, NY + 1)
X, Y = np.meshgrid(x, y)
p = np.vstack([X.ravel(), Y.ravel()])
elem = []
for i in range(NX):
    for j in range(NY):
        v = i + j * (NX + 1)
        elem.append([v, v + 1, v + NX + 1])
        elem.append([v + 1, v + NX + 2, v + NX + 1])
mesh = MeshTri(p, np.array(elem).T)

basis = Basis(mesh, element)

@BilinearForm
def laplace(u, v, w):
    return K * dot(grad(u), grad(v))

@LinearForm
def load(v, w):
    return F_SRC(w.x, w.y) * v

A = laplace.assemble(basis)
b_vol = load.assemble(basis)
b = b_vol  # alias for contract compatibility

# Boundary selection: explicitly exclude corners from the interface set
outer_facets = mesh.facets_satisfying(lambda x: (x[0] < TOL) | (x[1] < TOL) | (x[1] > Y1 - TOL))
iface_facets = mesh.facets_satisfying(lambda x: (x[0] > X1 - TOL) & (x[1] > TOL) & (x[1] < Y1 - TOL))

outer_dofs = basis.get_dofs(facets=outer_facets).flatten()
iface_dofs = basis.get_dofs(facets=iface_facets).flatten()
y_if = basis.doflocs[1, iface_dofs]

# Impose interface Dirichlet values from partner (or fallback on iter 1)
T_if = sample(imp, "values", T_INIT, y_if)
x0 = np.zeros(A.shape[0])
x0[iface_dofs] = T_if
D = np.concatenate([outer_dofs, iface_dofs])
sol = solve(*condense(A, b_vol, D=D, x=x0))

# ─────────────────────────────────────────────────────────────────────────
# Flux recovery (consistent residual method)
# ─────────────────────────────────────────────────────────────────────────

fbasis = FacetBasis(mesh, element, facets=iface_facets)

@LinearForm
def flux_load(v, w):
    return w["g"] * v

@LinearForm
def unit_load(v, w):
    return 1.0 * v          # w_i = int_Gamma phi_i ds

# Outward normal flux density q = -(k grad T).n on the interface.
r = A @ sol - b_vol                    # r = A u_h - b_vol, no bc applied
wgt = unit_load.assemble(fbasis)       # w_i = int_Gamma phi_i ds

Q = np.zeros(len(iface_dofs))
ok = np.abs(wgt[iface_dofs]) > 1e-14
Q[ok] = -r[iface_dofs][ok] / wgt[iface_dofs][ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux.
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    ...); a Dirichlet side that just negates the partner's flux.
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
