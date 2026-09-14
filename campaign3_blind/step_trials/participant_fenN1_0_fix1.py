"""FEniCSx (dolfinx) participant for the OASiS `couple` driver.

CONTRACT (do not change): runs in its work_dir with no arguments, reads
imports.json (written every iteration; it is `{}` on iteration 1), writes
exports.json LAST.

Physics: steady conduction  -div(K grad T) = F_SRC  on one rectangular
subdomain of a domain split by a straight interface at x = IFACE_X.
Top and bottom edges are natural (zero-flux). The non-interface x-boundary
carries a Dirichlet value T_OUTER.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
#    As shipped this is the LEFT / Dirichlet side; the payload that served
# ... (explanation continues in the annotated block)
SIDE      = "neumann"     # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "left"        # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.4      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 5.0           # conductivity


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

    HOW THIS ENTERS FEniCSx (your solve below has to do it): a UFL expression
    carries no NumPy ufuncs, so do NOT call this function on
    ufl.SpatialCoordinate (np.zeros_like(x) does not even raise -- it returns
    a 0-d object array and the source collapses to a constant). Interpolate it
    into a P1 Function on your space instead, f_h.interpolate(lambda X:
    F_SRC(X[0], X[1])), and integrate f_h * v * dx -- the P1 interpolant of the
    source, quadrature error O(h^2), the order of the discretisation.
    """
    return 5.0 * np.pi**2 * (1.4 - x) * np.sin(np.pi * y)
T_OUTER   = 0.0           # Dirichlet value on the NON-interface x-boundary
NX, NY    = 8, 10         # this subdomain's OWN mesh; need not match the partner
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level; the per-level
# ... (explanation continues in the annotated block)
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OASIS_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))   # a multi-level call's level keys
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

OUTER_X = X0 if IFACE_X == X1 else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S*e_x


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1,
    so the caller must fall back to an initial guess."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


def sample(imp, key, fallback, y):
    """Map the partner's samples onto THIS participant's interface points.
    The driver does no interpolation — non-matching meshes are handled here."""
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

# Create mesh
msh = dmesh.create_box(MPI.COMM_WORLD, 
                       [[X0, Y0, 0.0], [X1, Y1, 0.0]], 
                       [NX, NY, 1], 
                       dmesh.CellType.triangle)

# Function space
V = fem.functionspace(msh, ("Lagrange", 1))

# Spatial coordinate
x_coords = ufl.SpatialCoordinate(msh)

# Source term interpolated into Function
f_src = fem.Function(V)
f_src.interpolate(lambda X: F_SRC(X[0], X[1]))

# Trial/test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Find interface facets (at x ≈ IFACE_X)
tdim = msh.topology.dim
fdim = tdim - 1
msh.topology.create_connectivity(tdim, fdim)
facets = msh.topology.index_map(fdim).size_local
facet_coords = msh.geometry.x[msh.topology.connectivity(fdim, tdim).connectivity()]
iface_facets = []
for i in range(facets):
    facet_nodes = msh.topology.connectivity(fdim, tdim).connectivity()[i]
    facet_xs = facet_coords[facet_nodes, 0]
    if np.allclose(facet_xs, IFACE_X, atol=1e-12):
        iface_facets.append(i)

iface_facets = np.sort(np.array(iface_facets))

# Find outer boundary facets (x = OUTER_X, y = Y0, y = Y1)
outer_facets = []
for i in range(facets):
    if i in iface_facets:
        continue
    facet_nodes = msh.topology.connectivity(fdim, tdim).connectivity()[i]
    facet_xs = facet_coords[facet_nodes, 0]
    facet_ys = facet_coords[facet_nodes, 1]
    if np.allclose(facet_xs, OUTER_X, atol=1e-12) or \
       np.allclose(facet_ys, Y0, atol=1e-12) or \
       np.allclose(facet_ys, Y1, atol=1e-12):
        outer_facets.append(i)

outer_facets = np.sort(np.array(outer_facets))

# Create mesh tags
from dolfinx.mesh import meshtags
iface_tags = meshtags(msh, fdim, iface_facets, 1)
outer_tags = meshtags(msh, fdim, outer_facets, 2)

# Boundary measures
ds_if = ufl.Measure("ds", domain=msh, subdomain_data=iface_tags, subdomain_id=1)
ds_outer = ufl.Measure("ds", domain=msh, subdomain_data=outer_tags, subdomain_id=2)
dx = ufl.dx(msh)

# Dirichlet BC on outer boundary
bc_dofs, bc_indices = fem.locate_dofs_topological((V, 0), fdim, outer_facets)
bc = fem.dirichletbc(default_scalar_type(T_OUTER), bc_dofs, V)

# Weak form: a(u,v) = K*grad(u)*grad(v)
a = K * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

# Volume source term
L_vol = f_src * v * dx

# Import flux from partner (Neumann side)
if imp and imp.get("normal_fluxes"):
    q_in = np.asarray(imp["normal_fluxes"], float)
    q_coords = np.array([[c[1] for c in imp["coordinates"]]], float).flatten()
else:
    q_in = np.full(NY + 1, float(Q_INIT))
    q_coords = np.linspace(Y0, Y1, NY + 1)

# Sample imported flux at interface DOFs
xy = V.tabulate_dof_coordinates().reshape(-1, 2)
iface_mask = np.abs(xy[:, 0] - IFACE_X) < 1e-12
iface_dofs = np.where(iface_mask)[0]
y_if = xy[iface_mask, 1]
q_at_iface = sample(imp, "normal_fluxes", Q_INIT, y_if)

# Create Function for Neumann BC
g_func = fem.Function(V)
g_func.x.array[:] = 0.0
g_func.x.array[iface_dofs] = q_at_iface

# Interface load term (Neumann)
L_neumann = g_func * v * ds_if

# Total linear form
L = L_vol + L_neumann

# Assemble and solve
problem = LinearProblem(a, L, bcs=[bc],
                        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                        petsc_options_prefix='run')
uh = problem.solve()

# Get DOF coordinates
xy = V.tabulate_dof_coordinates().reshape(-1, 2)

# Identify interface DOFs
iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-12)[0]
y_if = xy[iface_dofs, 1]

# Identify outer boundary DOFs
outer_mask = (np.abs(xy[:, 0] - OUTER_X) < 1e-12) | \
             (np.abs(xy[:, 1] - Y0) < 1e-12) | \
             (np.abs(xy[:, 1] - Y1) < 1e-12)
outer_dofs = np.where(outer_mask)[0]

# Flux recovery: residual without BC lifting
Amat = _fp.assemble_matrix(fem.form(a))              # no bcs= on purpose
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))          # no lifting, no set_bc
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

wvec = _fp.assemble_vector(fem.form(w_ * ds_if))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux
# ... (explanation continues in the annotated block)
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

T = uh.x.array[iface_dofs]
print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# EACH SIDE COMPUTES ITS OWN FLUX. NEVER WRITE THE PARTNER'S NEGATED.
#
# ... (explanation continues in the annotated block)
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
# ... (explanation continues in the annotated block)
_chk_vals = np.asarray(T, float).ravel()
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

# PER-LEVEL PERSISTENCE: this level's whole field and its interface trace and
# flux, named by LEVEL, never overwritten by the next level (exports.json is).
with open(f"field_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(xy[:, :2], uh.x.array):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
with open(f"interface_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    for _y, _t, _q in zip(y_if, T, Q):
        _f.write(f"{float(IFACE_X):.11e},{float(_y):.11e},{float(_t):.11e},{float(_q):.11e}\n")
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     L_vol
#     a
#     ds_if
#     iface_dofs
#     outer_dofs
#     uh
#     w_
#     xy
#     y_if
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
