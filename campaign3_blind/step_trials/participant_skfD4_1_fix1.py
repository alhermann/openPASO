"""scikit-fem participant for the OASiS `couple` driver.

Steady heat conduction  -div(k grad T) = f  on one rectangular subdomain.
CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.
"""
import json
from pathlib import Path

import logging                # scikit-fem logs through it (logging.basicConfig(level=logging.INFO)),
                              # and its output goes to STDERR -- redirect with 2>&1 or lose it
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
    """Volumetric source, as a function of position.

    Returns zero as shipped, which is a PLACEHOLDER like every number above
    and is almost never what your problem wants. THIS KNOB USED TO BE A SCALAR
    CONSTANT, AND A CONSTANT CANNOT REPRESENT A SOURCE THAT VARIES WITH
    POSITION: the source of a manufactured solution is a POLYNOMIAL in x and y,
    and no single number is that polynomial. Left at zero the temperature is
    harmonic, the outer Dirichlet values are the only data left in the problem,
    and the answer degenerates to the 1-D profile between them — the interface
    flux is one constant along the whole interface, and it is identically zero
    when the two subdomains carry the same outer value. The coupling will
    converge beautifully to that, and it is not the problem you were given.

    If your problem states a source, or gives you a manufactured solution whose
    source term you derived, put it here. `x` and `y` are NumPy arrays, so
    build the answer with NumPy and return ONE array of the same shape (write
    `0.0 * x + c` for a genuine constant, never a bare `c`):

        # -div(K grad T) for the manufactured T = x**3 * y**2
        return -K * (6.0 * x * y**2 + 2.0 * x**3)
    """
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
# ... (explanation continues in the annotated block)
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# ... (explanation continues in the annotated block)


# === BUILD MESH AND BASIS ===
mesh = MeshTri.init_tensor(np.linspace(X0, X1, NX + 1), np.linspace(Y0, Y1, NY + 1))
element = ElementTriP1
basis = Basis(mesh, element)
fbasis = FacetBasis(mesh, element)

# === WEAK FORMS ===
@BilinearForm
def laplace(v, w, _):
    return dot(K * grad(v), grad(w))

@LinearForm
def source(v, w):
    return F_SRC(w.x[0], w.x[1]) * v

# === ASSEMBLE SYSTEM ===
A = asm(laplace, basis)
b_vol = asm(source, basis)

# === IDENTIFY DOFS ===
# Outer boundary: x=0, y=0, y=1 (excluding interface corners)
outer_facets = np.where((mesh.p[0] <= OUTER_X + TOL) & 
                        ((mesh.p[1] <= TOL) | (mesh.p[1] >= 1.0 - TOL)))[0]
outer_dofs = basis.get_dofs(facets=outer_facets).flatten()

# Interface: x=X1 (right edge), excluding corners that touch outer boundary
iface_facets = np.where((mesh.p[0] >= X1 - TOL))[0]
# Remove corners (nodes where y=0 or y=1)
corner_mask = np.zeros(len(iface_facets), dtype=bool)
for i, fidx in enumerate(iface_facets):
    node_idx = mesh.facets[:, fidx][0]
    y_coord = mesh.p[1, node_idx]
    if y_coord <= TOL or y_coord >= 1.0 - TOL:
        corner_mask[i] = True
iface_facets_clean = iface_facets[~corner_mask]
iface_dofs = basis.get_dofs(facets=iface_facets_clean).flatten()

# Get y-coordinates of interface nodes (sorted)
y_if = mesh.p[1, iface_dofs].copy()
sort_idx = np.argsort(y_if)
iface_dofs = iface_dofs[sort_idx]
y_if = y_if[sort_idx]

# === APPLY INTERFACE DATA ===
if SIDE == "dirichlet":
    T_if = sample(imp, "values", T_INIT, y_if)
    D = np.concatenate([outer_dofs, iface_dofs])
    # Create solution array with all Dirichlet values
    sol = basis.zeros()
    sol[outer_dofs] = T_OUTER
    sol[iface_dofs] = T_if
    # Condense and solve for remaining DOFs
    A_c, b_c = condense(A, b_vol, D=D, x=sol)
    sol_free = solve(A_c, b_c)
    # Fill free DOFs
    free_dofs = np.setdiff1d(basis.dofs['u'].flatten(), D)
    sol[free_dofs] = sol_free
else:
    q_if = sample(imp, "normal_fluxes", Q_INIT, y_if)
    gnod = basis.zeros()
    gnod[iface_dofs] = q_if
    @LinearForm
    def flux_load(v, w):
        return w["g"] * v
    
    b = b_vol + asm(flux_load, fbasis, g=fbasis.interpolate(gnod))
    D = outer_dofs
    A_c, b_c = condense(A, b, D=D, x=basis.zeros())
    sol = solve(A_c, b_c)
    sol[outer_dofs] = T_OUTER

# === RECOVER CONSISTENT OUTWARD FLUX ===
r = A @ sol - b_vol                    # r = A u_h - b_vol, no bc applied
wgt = asm(LinearForm(lambda v, w: 1.0 * v), fbasis)       # w_i = int_Gamma phi_i ds

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

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     A
#     b
#     b_vol
#     basis
#     fbasis
#     iface_dofs
#     outer_dofs
#     sol
#     y_if
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
