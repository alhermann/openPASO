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

dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# Build mesh
mesh = dmesh.create_rectangle(MPI.COMM_WORLD, 
                               np.array([[X0, Y0], [X1, Y1]]), 
                               (NX, NY), 
                               dmesh.CellType.triangle)
V = fem.functionspace(mesh, ("Lagrange", 1))

# Tag interface and outer boundaries
import basix
import dolfinx.mesh
from dolfinx.mesh import locate_entities_boundary

# Interface: x = IFACE_X
def interface_marker(x):
    return np.isclose(x[0], IFACE_X, atol=1e-14)

# Outer boundary: x = OUTER_X or y = Y0 or y = Y1
def outer_marker(x):
    return np.isclose(x[0], OUTER_X, atol=1e-14) | np.isclose(x[1], Y0, atol=1e-14) | np.isclose(x[1], Y1, atol=1e-14)

# Get facet indices
interface_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, interface_marker)
outer_facets = locate_entities_boundary(mesh, mesh.topology.dim - 1, outer_marker)

# Create meshtags
from dolfinx.mesh import meshtags
import numpy as np
facetdim = mesh.topology.dim - 1
mesh.topology.create_connectivity(facetdim, mesh.topology.dim)
ifacet_tags = meshtags(mesh, facetdim, interface_facets, 1)
outer_tags = meshtags(mesh, facetdim, outer_facets, 1)

# Define measures
dx = ufl.dx(mesh)
ds_if = ufl.ds(mesh, subdomain_data=ifacet_tags)
ds_outer = ufl.ds(mesh, subdomain_data=outer_tags)

# Function space and trial/test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Interpolate source term
f_h = fem.Function(V)
f_h.interpolate(lambda X: F_SRC(X[0], X[1]))

# Weak form: a(u,v) = L(v)
# -div(K grad u) = f  =>  K grad u . grad v dx = f v dx + g v ds (Neumann)
a = K * ufl.dot(ufl.grad(u), ufl.grad(v)) * dx

# Volume load
L_vol = f_h * v * dx

# Read imported flux (for Neumann side)
if SIDE == "neumann":
    # Interface y-coordinates at mesh nodes on interface
    # Get interface DOFs first
    iface_dofs = V.subset_dofs(ifacet_tags.find(1))
    # Get coordinates of interface nodes
    iface_coords = V.tabulate_dof_coordinates()[iface_dofs]
    y_if = iface_coords[:, 1]
    # Sort interface DOFs by y-coordinate for consistent ordering
    y_order = np.argsort(y_if)
    iface_dofs_sorted = iface_dofs[y_order]
    y_if_sorted = y_if[y_order]
    
    # Sample imported flux at interface points
    q_in = sample(imp, "normal_fluxes", Q_INIT, y_if_sorted)
    # Create function for Neumann BC
    g_h = fem.Function(V)
    g_h.x.array[iface_dofs_sorted] = q_in
    
    # Neumann term: g v ds (partner's flux is applied as natural BC)
    # The partner exports their outward normal flux; we apply it unchanged
    # as the natural boundary condition on our interface
    L_n = g_h * v * ds_if
    L = L_vol + L_n
else:
    L = L_vol
    L_n = None

# Dirichlet BC on outer boundary
from dolfinx.fem import dirichletbc
outer_dofs = V.subset_dofs(outer_tags.find(1))
bc = dirichletbc(fem.Constant(mesh, T_OUTER), outer_dofs, V)

# Solve
p = LinearProblem(a, L, bcs=[bc],
                  petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                  petsc_options_prefix='run')
uh = p.solve()

# Get interface DOFs and coordinates (sorted by y)
iface_dofs = V.subset_dofs(ifacet_tags.find(1))
iface_coords = V.tabulate_dof_coordinates()[iface_dofs]
y_if = iface_coords[:, 1]
y_order = np.argsort(y_if)
iface_dofs = iface_dofs[y_order]
y_if = y_if[y_order]

# Get outer DOFs
outer_dofs = V.subset_dofs(outer_tags.find(1))

# Get all node coordinates for field export
xy = V.tabulate_dof_coordinates()

# Consistent flux recovery
# q = -(K grad u - f)/h with h the node's tributary length
# Equivalent to: q_i = -r_i / w_i where r = A u - b_vol (no BC applied)
# and w_i = int_Gamma phi_i ds

# Assemble matrix without BCs
Amat = _fp.assemble_matrix(fem.form(a))
Amat.assemble()

# Assemble volume load vector
bvec = _fp.assemble_vector(fem.form(L_vol))
bvec.ghostUpdate()

# Compute residual: r = A u - b_vol
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

# Compute interface weight: w_i = int_Gamma phi_i ds
wvec = _fp.assemble_vector(fem.form(ufl.Constant(mesh, 1.0) * v * ds_if))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

# Compute consistent flux: q_i = -r_i / w_i
Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# An interface node that ALSO lies on the outer Dirichlet boundary carries
# the OUTER reaction as well, so its residual is not this interface's flux
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
