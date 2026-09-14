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
NX, NY    = 6, 10         # this subdomain's own mesh
T_INIT    = 0.0           # iteration-1 fallback interface temperature
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


@BilinearForm
def laplace(u, v, w):
    return K * dot(grad(u), grad(v)) * w['measure']


@LinearForm
def source(v, w):
    return F_SRC(w['x'][0], w['x'][1]) * v * w['measure']


@LinearForm
def flux_load(v, w):
    return w["g"] * v


@LinearForm
def unit_load(v, w):
    return 1.0 * v          # w_i = int_Gamma phi_i ds


# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
# Build the mesh, the function space, the weak form and the linear solve for
# ordinary finite-element work and OASiS has no business dictating it.
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below.
# ─────────────────────────────────────────────────────────────────────────

mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1),
                           np.linspace(Y0, Y1, NY + 1))

basis = Basis(mesh, ElementTriP1(), doforder='natural')
fbasis = FacetBasis(mesh, ElementTriP1(), doforder='natural')

A = laplace.assemble(basis=basis)
b_vol = source.assemble(basis=basis)

# Find interface facets (x = IFACE_X within tolerance)
iface_facets = mesh.facets_satisfying(lambda x: np.abs(x[0] - IFACE_X) < TOL,
                                      boundaries_only=True)
iface_dofs = fbasis.find_dofs(iface_facets)

# Find outer boundary DOFs (non-interface boundaries where u = T_OUTER)
outer_facets = mesh.facets_satisfying(
    lambda x: (np.abs(x[0] - OUTER_X) < TOL) & (np.abs(x[1] - Y0) > TOL) & (np.abs(x[1] - Y1) > TOL),
    boundaries_only=True
)
bottom_facets = mesh.facets_satisfying(lambda x: np.abs(x[1] - Y0) < TOL, boundaries_only=True)
top_facets = mesh.facets_satisfying(lambda x: np.abs(x[1] - Y1) < TOL, boundaries_only=True)
outer_dofs = np.concatenate([fbasis.find_dofs(outer_facets),
                             fbasis.find_dofs(bottom_facets),
                             fbasis.find_dofs(top_facets)])
outer_dofs = np.unique(outer_dofs)

# Get interface node coordinates
y_if = mesh.p[1, iface_dofs].copy()

# Read imports and apply interface data
imp = read_imports()

if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    sol = basis.zeros()
    sol[iface_dofs] = T_if
    D = np.concatenate([outer_dofs, iface_dofs])
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()
    gnod[iface_dofs] = q_if
    b = b_vol + flux_load.assemble(fbasis, g=fbasis.interpolate(gnod))
    D = outer_dofs

# Apply outer Dirichlet conditions
sol[outer_dofs] = T_OUTER

# Condense and solve
A_c, b_c = condense(A, b_vol, D=D, x0=sol)
sol_c = solve(A_c, b_c)
sol = basis.project(sol_c, D=D, x0=sol)

# ─────────────────────────────────────────────────────────────────────────
# Flux recovery (DO NOT MODIFY)
# Outward normal flux density q = -(k grad T).n on the interface.
# The residual r = A @ sol - b_vol at constrained interface DOFs gives
# the consistent nodal flux without differentiation.
# ─────────────────────────────────────────────────────────────────────────

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
#    looks converged); a Dirichlet side that just copies back the partner's flux
#    instead of recovering from its own solution.
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
# (Dirichlet role only: a Neumann side's consistent recovery of a CONSTANT
#  applied flux can legitimately reproduce it to the last bit.)
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
