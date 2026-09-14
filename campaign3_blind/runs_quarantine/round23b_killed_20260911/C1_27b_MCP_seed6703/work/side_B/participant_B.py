"""FEniCSx (dolfinx) THERMO-ELASTIC NEUMANN participant for the OASiS `couple` driver.

Steady thermoelasticity on subdomain B of a domain split by interface at x = 0.625,
plane strain, small strain:
    -div(k grad T) = f_T
    -div(sigma_tot(u, T)) = f_u,   sigma_tot = 2 mu eps(u) + lam tr(eps(u)) I - beta T I

NEUMANN side: imports [qn, qx, qy] from partner, exports [T, ux, uy]
"""
import json
import sys
from pathlib import Path

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, log, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

# Set logging level BEFORE creating mesh
log.set_log_level(log.LogLevel.INFO)

# ── PROBLEM PARAMETERS FOR SUBDOMAIN B ───────────────────────────────────────
SIDE      = "neumann"     # Neumann side: import flux/traction, export field
PARTNER   = "A"           # Partner name in couple() call
X0, X1    = 0.625, 1.5    # Subdomain B x-extent
Y0, Y1    = 0.0, 1.0      # Subdomain B y-extent
IFACE_X   = 0.625         # Interface is at LEFT edge of subdomain B
K         = 3.0           # Conductivity in B
LAM, MU   = 600.0, 1600.0 # Lame parameters in B
BETA      = 1.0           # Thermal stress coefficient

# Source terms for subdomain B (converted to Python/NumPy expressions)
def F_T(x, y):
    """Heat source f_T(x, y) for subdomain B."""
    return (-2*x**3*y/3 + 8*x**3/45 - 8*x**2*y/3 + 4*x**2/9 
            - 2*x*y**3/3 + 8*x*y**2/15 + 251*x*y/120 - 41*x/90 
            - 8*y**3/9 + 4*y**2/9 + 829*y/144 - 11/12)

def F_U(x, y):
    """Body force (f_x, f_y) for subdomain B."""
    fx = (365273*x**2*y**3/5985 + 273076*x**2*y**2/4275 - 3555593*x**2*y/29925 
          + 15192*x**2/665 - 119754589*x*y**3/251370 - 359593607*x*y**2/718200 
          + 4672588891*x*y/5027400 - 13314341*x/74480 + 2532*y**5/175 
          + 633*y**4/25 + 4520447879*y**3/20109600 + 894533323*y**2/2298240 
          - 2001017755*y/3217536 + 28506033/238336)
    fy = (x**3*y**2/9 - 8*x**3*y/135 - x**3/135 - 1720331*x**2*y**2/1764 
          - 1032893*x**2*y/1512 + 13423129*x**2/21168 + 27852*x*y**4/665 
          + 27852*x*y**3/475 + 9525905951*x*y**2/6703200 + 269425655*x*y/229824 
          - 872177333*x/846720 - 1436809*y**4/13965 - 1436809*y**3/9975 
          + 976524757*y**2/4468800 - 1508594569*y/5362560 + 13360741/112896)
    return fx, fy

T_OUTER   = 0.0           # T on outer boundary (non-interface)
UX_OUTER, UY_OUTER = 0.0, 0.0   # u on outer boundary
T_INIT, UX_INIT, UY_INIT = 0.0, 0.0, 0.0   # Iteration-1 fallback field
Q_INIT    = (0.0, 0.0, 0.0)                # Iteration-1 fallback flux/traction

# ── PER-LEVEL CONFIG ─────────────────────────────────────────────────────────
LEVEL = 1
NX, NY = 7, 8  # Default for level 1 (h=1/8)
if Path("config.json").is_file():
    try:
        cfg = json.loads(Path("config.json").read_text() or "{}")
        LEVEL = int(cfg.get("level", LEVEL))
        NX = int(cfg.get("nx", NX))
        NY = int(cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

OUTER_X = X1 if IFACE_X == X0 else X0  # Outer x-boundary (opposite interface)
S = -1.0 if IFACE_X < OUTER_X else 1.0  # Outward normal at interface = S*e_x

# ── IMPORT HANDSHAKE ─────────────────────────────────────────────────────────
def read_imports():
    """imports.json is {partner_name: InterfaceData}; {} on iteration 1."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text() or "{}").get(PARTNER) or None
    except json.JSONDecodeError:
        return None

def sample(imp, key, fallback, y):
    """Map partner's samples onto THIS side's interface points, component by component."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    return np.column_stack([np.interp(y, ys[o], vs[o, c]) for c in range(fb.size)])

imp = read_imports()

# ── HOLE 1: MESH AND SPACES ──────────────────────────────────────────────────
domain = dmesh.create_rectangle(MPI.COMM_WORLD, 
                                 [[X0, Y0], [X1, Y1]], 
                                 [NX, NY], 
                                 dmesh.CellType.triangle)

ST = fem.functionspace(domain, ("Lagrange", 1))
SU = fem.functionspace(domain, ("Lagrange", 1, (2,)))

# ── INTERFACE AND OUTER DOF SETS ─────────────────────────────────────────────
fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

xyT = ST.tabulate_dof_coordinates()
iface_T = np.where(np.abs(xyT[:, 0] - IFACE_X) < 1e-10)[0]
iface_T = iface_T[np.argsort(xyT[iface_T, 1])]
y_if = xyT[iface_T, 1]

outer_T = np.where((np.abs(xyT[:, 0] - OUTER_X) < 1e-10) |
                   (np.abs(xyT[:, 1] - Y0) < 1e-10) |
                   (np.abs(xyT[:, 1] - Y1) < 1e-10))[0]

xyU = SU.tabulate_dof_coordinates()
iface_U = np.where(np.abs(xyU[:, 0] - IFACE_X) < 1e-10)[0]
iface_U = iface_U[np.argsort(xyU[iface_U, 1])]

outer_U = np.where((np.abs(xyU[:, 0] - OUTER_X) < 1e-10) |
                   (np.abs(xyU[:, 1] - Y0) < 1e-10) |
                   (np.abs(xyU[:, 1] - Y1) < 1e-10))[0]

if len(iface_T) == 0:
    sys.exit(f"no interface DOFs at x={IFACE_X}")

# Interface corners belong to outer boundary
corner = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_T = iface_T[~corner]
iface_bc_U = iface_U[~corner]

# Mark interface facet
facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], IFACE_X))
tags = dmesh.meshtags(domain, fdim, np.sort(facets), np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)

tT, vT = ufl.TrialFunction(ST), ufl.TestFunction(ST)
uu, vu = ufl.TrialFunction(SU), ufl.TestFunction(SU)

# ── INTERFACE ROLE (NEUMANN) ─────────────────────────────────────────────────
bcs_if_T, bcs_if_U = [], []
q = sample(imp, "normal_fluxes", Q_INIT, y_if)

gq = fem.Function(ST)
gq.x.array[:] = 0.0
gq.x.array[iface_T] = q[:, 0]

gt = fem.Function(SU)
gt.x.array[:] = 0.0
gt.x.array[2 * iface_U] = q[:, 1]
gt.x.array[2 * iface_U + 1] = q[:, 2]

L_T_if = gq * vT * ds_if
L_U_if = ufl.inner(gt, vu) * ds_if

# ── HOLE 2: FORMS, SOURCES, OUTER BCs, SOLVES ─────────────────────────────────
# Heat equation: -div(k grad T) = f_T
aT = K * ufl.inner(ufl.grad(tT), ufl.grad(vT)) * ufl.dx

# Body force as Function
fT_h = fem.Function(ST)
coords = ST.tabulate_dof_coordinates()
fT_vals = F_T(coords[:, 0], coords[:, 1])
fT_h.x.array[:] = fT_vals
L_T_vol = fT_h * vT * ufl.dx

# Plane strain elasticity with thermal stress
# sigma_tot = 2*mu*eps(u) + lam*tr(eps(u))*I - beta*T*I
eps = lambda w: ufl.sym(ufl.grad(w))
Id = ufl.Identity(2)

au = 2*MU * ufl.inner(eps(uu), eps(vu)) * ufl.dx + LAM * ufl.div(uu) * ufl.div(vu) * ufl.dx

# Body force as VectorFunction
fU_h = fem.Function(SU)
fx_vals, fy_vals = F_U(coords[:, 0], coords[:, 1])
fU_h.x.array[:] = np.vstack([fx_vals, fy_vals]).flatten()
L_U_vol_force = ufl.inner(fU_h, vu) * ufl.dx

# Outer Dirichlet BCs - use Functions for vector BCs
bc_T_outer = fem.dirichletbc(fem.Constant(domain, T_OUTER), outer_T.astype(np.int32), ST)

# For vector space, create a Function with the constant values
gU = fem.Function(SU)
gU.x.array[::2] = UX_OUTER  # x-component
gU.x.array[1::2] = UY_OUTER  # y-component
bc_U_outer = fem.dirichletbc(gU, outer_U.astype(np.int32))

bcs_T = [bc_T_outer]
bcs_U = [bc_U_outer]

# Solve heat problem
prob_T = LinearProblem(aT, L_T_vol + L_T_if, bcs=bcs_T + bcs_if_T,
                       petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                       petsc_options_prefix='thermal')
Th = prob_T.solve()

# Elasticity with thermal stress term: -div(sigma_el) - div(-beta*T*I) = f
# Weak form: int(2*mu*eps(u):eps(v) + lam*div(u)*div(v)) dx = int(f.v) dx + int(beta*T*div(v)) dx
L_U_thermal = BETA * Th * ufl.div(vu) * ufl.dx

prob_U = LinearProblem(au, L_U_vol_force + L_U_thermal + L_U_if, bcs=bcs_U + bcs_if_U,
                       petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                       petsc_options_prefix='elastic')
Uh = prob_U.solve()

# ── INTERFACE DATA CHECK ─────────────────────────────────────────────────────
if SIDE == "dirichlet" and len(iface_bc_T):
    _gap = float(np.abs(Th.x.array[iface_bc_T] - gq.x.array[iface_bc_T]).max())
    if _gap > 1e-9 * max(1.0, float(np.abs(gq.x.array).max())):
        raise SystemExit(f"interface trace mismatch: gap {_gap:.3e}")

# ── CONSISTENT OUTWARD FLUX RECOVERY ─────────────────────────────────────────
def _residual(a_form, L_vol, sol, V):
    A = _fp.assemble_matrix(fem.form(a_form))
    A.assemble()
    b = _fp.assemble_vector(fem.form(L_vol))
    b.ghostUpdate()
    r = A.createVecLeft()
    A.mult(sol.x.petsc_vec, r)
    r.axpy(-1.0, b)
    return r.array.copy()

rT = _residual(aT, L_T_vol, Th, ST)
rU = _residual(au, L_U_vol_force + L_U_thermal, Uh, SU)

wT = _fp.assemble_vector(fem.form(vT * ds_if))
wT.ghostUpdate()

ones = fem.Constant(domain, np.ones(2, dtype=default_scalar_type))
wU = _fp.assemble_vector(fem.form(ufl.inner(ones, vu) * ds_if))
wU.ghostUpdate()

wiT = wT.array[iface_T]
idxU = np.column_stack([2 * iface_U, 2 * iface_U + 1])
wiU = wU.array[idxU]

Q = np.zeros((len(iface_T), 3))
okT = np.abs(wiT) > 1e-14
Q[okT, 0] = -rT[iface_T][okT] / wiT[okT]
okU = np.abs(wiU) > 1e-14
Q[:, 1:][okU] = -rU[idxU][okU] / wiU[okU]

# Handle corners by nearest interior node
suspect = np.isin(iface_T, outer_T) | ~okT | ~okU.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

V = np.column_stack([Th.x.array[iface_T], Uh.x.array[2 * iface_U], Uh.x.array[2 * iface_U + 1]])

print(f"[fenics {SIDE} thermoelastic] interface n={len(V)} "
      f"T=[{V[:,0].min():.6g},{V[:,0].max():.6g}] ux=[{V[:,1].min():.6g},{V[:,1].max():.6g}] "
      f"uy=[{V[:,2].min():.6g},{V[:,2].max():.6g}] qn=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}]")

ndof_T = ST.dofmap.index_map.size_global
ndof_U = SU.dofmap.index_map.size_global
print(f"NDOF = {ndof_T + ndof_U}")

# ── EXPORT SELF-CHECK ───────────────────────────────────────────────────────
if not (np.isfinite(V).all() and np.isfinite(Q).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite values or fluxes")

_qin = np.asarray((imp or {}).get("normal_fluxes") or [], float)
if SIDE == "neumann" and _qin.size and np.abs(_qin).max() > 0 and np.abs(Q).max() < 1e-9 * np.abs(_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: recovered flux ~0 against nonzero imported flux")

# ── PER-LEVEL OUTPUT FILES ───────────────────────────────────────────────────
xyT_all = ST.tabulate_dof_coordinates()
with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,T,ux,uy\n")
    for i, (px, py) in enumerate(xyT_all[:, :2]):
        f.write(f"{float(px):.11e},{float(py):.11e},{float(Th.x.array[i]):.11e},"
                f"{float(Uh.x.array[2*i]):.11e},{float(Uh.x.array[2*i+1]):.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for y, (t, ux, uy), (qn, qx, qy) in zip(y_if, V, Q):
        f.write(f"{float(IFACE_X):.11e},{float(y):.11e},{float(t):.11e},{float(ux):.11e},"
                f"{float(uy):.11e},{float(qn):.11e},{float(-qx):.11e},{float(-qy):.11e}\n")

# ── exports.json LAST ────────────────────────────────────────────────────────
Path("exports.json").write_text(json.dumps({
    "field_name": "thermoelastic",
    "n_points": int(len(iface_T)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(a_) for a_ in row] for row in V],
    "normal_fluxes": [[float(a_) for a_ in row] for row in Q],
}, indent=2))

print(f"FEniCSx Neumann participant completed: LEVEL={LEVEL}, nx={NX}, ny={NY}")
