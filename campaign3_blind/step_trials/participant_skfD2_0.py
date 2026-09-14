"""scikit-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
import logging
from pathlib import Path

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
    return (np.pi**2) * x * np.sin(np.pi * y)
T_OUTER   = 0.0           # Dirichlet value on the NON-interface x-boundary
NX, NY    = 6, 10         # this subdomain's own mesh
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

ON_RIGHT = abs(IFACE_X - X1) < abs(IFACE_X - X0)   # interface is this side's x-max?
OUTER_X = X0 if ON_RIGHT else X1
S = 1.0 if ON_RIGHT else -1.0              # outward normal at interface = S * e_x
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

logging.basicConfig(level=logging.INFO)

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

# Build mesh
mesh = MeshTri.init_rect((X0, X1), (Y0, Y1), (NX, NY))

# Basis functions
basis = Basis(mesh, ElementTriP1())
fbasis = FacetBasis(mesh, ElementTriP1())

# Weak forms
@BilinearForm
def laplace(u, v, w):
    return dot(K * grad(u), grad(v))

@LinearForm
def source_load(v, w):
    return F_SRC(w['x'][0], w['y'][0]) * v

# Assemble matrices
A = laplace.assemble(basis=basis)
b_vol = source_load.assemble(basis=basis)

# Find interface facets (at x = IFACE_X)
iface_facets = mesh.facets_satisfying(lambda x: np.abs(x[0] - IFACE_X) < TOL)
iface_dofs = basis.get_dofs(facets=iface_facets).flatten()
y_if = mesh.p[0, iface_dofs]

# Find outer boundary facets (x = OUTER_X, y = Y0, y = Y1)
outer_facets = mesh.facets_satisfying(
    lambda x: (np.abs(x[0] - OUTER_X) < TOL) | 
              (np.abs(x[1] - Y0) < TOL) | 
              (np.abs(x[1] - Y1) < TOL)
)
outer_dofs = basis.get_dofs(facets=outer_facets).flatten()

# Apply interface Dirichlet from partner (or fallback)
T_if = sample(imp, "values", T_INIT, y_if)

# All Dirichlet DOFs
D = np.concatenate([outer_dofs, iface_dofs])

# Prepare RHS with Dirichlet values
b = b_vol.copy()
sol_D = np.zeros(len(D))
sol_D[:len(outer_dofs)] = T_OUTER
sol_D[len(outer_dofs):] = T_if

# Condense and solve
A_red, b_red = condense(A, b, D=D, x=sol_D)
sol_inner = solve(*A_red, b_red)
sol = basis.zeros()
sol[~np.isin(range(len(sol)), D)] = sol_inner
sol[D] = sol_D

# ─────────────────────────────────────────────────────────────────────────

# Outward normal flux density q = -(k grad T).n on the interface.
r = A @ sol - b_vol                    # r = A u_h - b_vol, no bc applied
wgt = fbasis.integral("1").dot(fbasis.dofs[:, 0])[iface_dofs]  # facet weights

# Alternative: compute tributary lengths manually for robustness
h_trib = np.zeros(len(iface_dofs))
for i, dof in enumerate(iface_dofs):
    # Sum half-lengths of adjacent facets on interface
    incident = np.where(iface_facets == mesh.t[:, :3].shape[0])[0]
    pass  # Use simple spacing approximation

# Simpler: use node spacing along interface
y_sorted = np.sort(y_if)
dy = np.diff(y_sorted)
h_trib_simple = np.zeros_like(y_if)
order_map = np.argsort(y_if)
h_trib_simple[order_map[0]] = dy[0]
h_trib_simple[order_map[-1]] = dy[-1]
h_trib_simple[order_map[1:-1]] = 0.5 * (dy[:-1] + dy[1:])

Q = np.zeros(len(iface_dofs))
ok = h_trib_simple > 1e-14
Q[ok] = -r[iface_dofs][ok] / h_trib_simple[ok]

# Handle corner/suspect nodes by interpolation
suspect = ~ok
good = np.where(ok)[0]
if len(good) > 0 and len(suspect) > 0:
    for i in np.where(suspect)[0]:
        nearest = good[np.argmin(np.abs(good - i))]
        Q[i] = Q[nearest]

# ── EXPORT SELF-CHECK ─ keep this block.
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

print(f"Side A complete: {len(iface_dofs)} interface DOFs, "
      f"flux range [{Q.min():.6e}, {Q.max():.6e}]")
