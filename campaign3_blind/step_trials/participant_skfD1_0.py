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
NX, NY    = 6, 10         # this subdomain's own mesh
T_INIT    = 0.0            # iteration-1 fallback interface temperature
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
# recovery below. Those are this tool's own interface, not your method.
# ─────────────────────────────────────────────────────────────────────────

# Build the mesh
mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))

basis = Basis(mesh, ElementTriP1())
fbasis = FacetBasis(mesh, ElementTriP1())

# Identify facets by location
x_min = np.abs(mesh.p[0, :] - X0) < TOL
x_max = np.abs(mesh.p[0, :] - X1) < TOL
y_min = np.abs(mesh.p[1, :] - Y0) < TOL
y_max = np.abs(mesh.p[1, :] - Y1) < TOL

outer_facets = np.where(x_min | y_min | y_max)[0]
iface_facets = np.where(x_max)[0]

iface_dofs = fbasis.find_dofs(iface_facets)
outer_dofs = fbasis.find_dofs(outer_facets)

# Weak forms
@BilinearForm
def stiffness(v, w, _):
    return K * dot(grad(v), grad(w))

@LinearForm
def load(v, w):
    return F_SRC(*w['coord']) * v

A = stiffness.assemble(basis)
b_vol = load.assemble(basis)

# Apply interface data according to SIDE
if SIDE == "dirichlet":
    y_if = mesh.p[1, iface_dofs]
    T_if = sample(imp, "values", T_INIT, y_if)
    sol = basis.zeros()
    sol[iface_dofs] = T_if
    D = np.concatenate([outer_dofs, iface_dofs])
else:
    y_if = mesh.p[1, iface_dofs]
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()
    gnod[iface_dofs] = q_if
    b = b_vol.copy()
    
    @LinearForm
    def flux_load(v, w):
        return w["g"] * v
    
    b += flux_load.assemble(fbasis, g=fbasis.interpolate(gnod))
    D = outer_dofs

# Condense and solve
A_red, b_red = condense(A, b_vol, D=D, x0=basis.zeros())
sol = solve(A_red, b_red)
sol = basis.prolong(sol, D, basis.zeros())

# ─────────────────────────────────────────────────────────────────────────
# Flux recovery (consistent nodal flux via residual)
# ─────────────────────────────────────────────────────────────────────────

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
#    looks like it converged); a Dirichlet side that just copies back the
#    partner's array without solving anything.
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
