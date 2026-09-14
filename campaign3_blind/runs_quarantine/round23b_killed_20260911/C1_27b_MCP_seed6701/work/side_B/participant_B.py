"""FEniCSx (dolfinx) THERMO-ELASTIC participant for the OASiS `couple` driver.
Neumann side: imports [qn, qx, qy], exports [T, ux, uy]
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

log.set_log_level(log.LogLevel.INFO)

# Read config
CFG = json.loads(Path("config.json").read_text())
NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = (float(CFG["x0"]), float(CFG["x1"]),
                  float(CFG["y0"]), float(CFG["y1"]))
KV = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IF = CFG.get("iface", "left")
SIDE = CFG.get("side", "neumann")
LEVEL = int(CFG.get("level", 1))

# Source terms for subdomain B (Python syntax)
def F_T(x):
    xx, yy = x[0], x[1]
    return -2*xx**3*yy/3 + 8*xx**3/45 - 8*xx**2*yy/3 + 4*xx**2/9 - 2*xx*yy**3/3 + 8*xx*yy**2/15 + 251*xx*yy/120 - 41*xx/90 - 8*yy**3/9 + 4*yy**2/9 + 829*yy/144 - 11/12

def F_U(x):
    xx, yy = x[0], x[1]
    fx = 365273*xx**2*yy**3/5985 + 273076*xx**2*yy**2/4275 - 3555593*xx**2*yy/29925 + 15192*xx**2/665 - 119754589*xx*yy**3/251370 - 359593607*xx*yy**2/718200 + 4672588891*xx*yy/5027400 - 13314341*xx/74480 + 2532*yy**5/175 + 633*yy**4/25 + 4520447879*yy**3/20109600 + 894533323*yy**2/2298240 - 2001017755*yy/3217536 + 28506033/238336
    fy = xx**3*yy**2/9 - 8*xx**3*yy/135 - xx**3/135 - 1720331*xx**2*yy**2/1764 - 1032893*xx**2*yy/1512 + 13423129*xx**2/21168 + 27852*xx*yy**4/665 + 27852*xx*yy**3/475 + 9525905951*xx*yy**2/6703200 + 269425655*xx*yy/229824 - 872177333*xx/846720 - 1436809*yy**4/13965 - 1436809*yy**3/9975 + 976524757*yy**2/4468800 - 1508594569*yy/5362560 + 13360741/112896
    return np.vstack([fx, fy])

PARTNER = "A"
IFACE_X = X0 if IF == "left" else X1
OUTER_X = X1 if IF == "left" else X0
S = -1.0 if IF == "left" else 1.0  # outward normal at interface

# Create mesh
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]], [NX, NY], dmesh.CellType.triangle)
ST = fem.functionspace(domain, ("Lagrange", 1))
SU = fem.functionspace(domain, ("Lagrange", 1, (2,)))

# Interface and outer DOFs
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

# Exclude corners from interface BCs
corner = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_T = iface_T[~corner]
iface_bc_U = iface_U[~corner]

# Interface facet marker
facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], IFACE_X))
tags = dmesh.meshtags(domain, fdim, np.sort(facets), np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)

# Trial/test functions
tT, vT = ufl.TrialFunction(ST), ufl.TestFunction(ST)
uu, vu = ufl.TrialFunction(SU), ufl.TestFunction(SU)

# Read imports - THIS IS THE KEY PART
imp = {}
if Path("imports.json").is_file():
    try:
        imp_raw = json.loads(Path("imports.json").read_text() or "{}")
        imp = imp_raw.get(PARTNER, {})
    except json.JSONDecodeError:
        imp = {}

print(f"[DEBUG] imports.json exists: {Path('imports.json').is_file()}")
print(f"[DEBUG] imp keys: {imp.keys() if imp else 'empty'}")

def sample(imp, key, fallback, y):
    """Interpolate partner's data onto this side's interface points."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        print(f"[DEBUG] No coordinates in imports, using fallback {fb}")
        return np.tile(fb, (len(y), 1))
    ys = np.array([c[1] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != ys.size or vs.shape[1] != fb.size:
        print(f"[DEBUG] Shape mismatch: vs={vs.shape}, expected ({ys.size}, {fb.size})")
        return np.tile(fb, (len(y), 1))
    o = np.argsort(ys)
    result = np.column_stack([np.interp(y, ys[o], vs[o, c]) for c in range(fb.size)])
    print(f"[DEBUG] Sampled {key}: min={result.min()}, max={result.max()}")
    return result

# Neumann role: apply imported flux/traction as natural BC
# IMPORTANT: The partner's outward flux is our inward load, so we apply it UNCHANGED
q = sample(imp, "normal_fluxes", (0.0, 0.0, 0.0), y_if)
print(f"[DEBUG] Imported normal_fluxes shape: {q.shape}, values: {q[:3]}")

gq = fem.Function(ST)
gq.x.array[:] = 0.0
gq.x.array[iface_T] = q[:, 0]

gt = fem.Function(SU)
gt.x.array[:] = 0.0
gt.x.array[2 * iface_U] = q[:, 1]
gt.x.array[2 * iface_U + 1] = q[:, 2]

# Apply as Neumann BC (natural boundary condition)
# For heat: L += gq * vT * ds_if (this adds the flux contribution)
# For elasticity: L += inner(gt, vu) * ds_if (this adds the traction contribution)
L_T_if = gq * vT * ds_if
L_U_if = ufl.inner(gt, vu) * ds_if

# Outer Dirichlet BCs
zero_T = fem.Constant(domain, np.float64(0.0))
zero_U = fem.Constant(domain, np.array([0.0, 0.0], dtype=np.float64))
bcs_T = [fem.dirichletbc(zero_T, outer_T.astype(np.int32), ST)]
bcs_U = [fem.dirichletbc(zero_U, outer_U.astype(np.int32), SU)]

# Heat source
fT_h = fem.Function(ST)
fT_h.interpolate(lambda x: F_T(x))
L_T_vol = fT_h * vT * ufl.dx

# Body force
fU_h = fem.Function(SU)
fU_h.interpolate(lambda x: F_U(x))

# Bilinear forms
aT = KV * ufl.inner(ufl.grad(tT), ufl.grad(vT)) * ufl.dx
eps_u = ufl.sym(ufl.grad(uu))
eps_v = ufl.sym(ufl.grad(vu))
au = 2 * MU * ufl.inner(eps_u, eps_v) * ufl.dx + LAM * ufl.tr(eps_u) * ufl.tr(eps_v) * ufl.dx

# Solve heat equation
pT = LinearProblem(aT, L_T_vol + L_T_if, bcs=bcs_T,
                   petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                   petsc_options_prefix='heat')
Th = pT.solve()

# Elasticity with thermal stress
L_U_vol = ufl.inner(fU_h, vu) * ufl.dx + BETA * Th * ufl.div(vu) * ufl.dx
pU = LinearProblem(au, L_U_vol + L_U_if, bcs=bcs_U,
                   petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                   petsc_options_prefix='elastic')
Uh = pU.solve()

# Consistent flux recovery
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
rU = _residual(au, L_U_vol, Uh, SU)
wT = _fp.assemble_vector(fem.form(ufl.TestFunction(ST) * ds_if))
wT.ghostUpdate()
ones = fem.Constant(domain, np.ones(2, dtype=default_scalar_type))
wU = _fp.assemble_vector(fem.form(ufl.inner(ones, ufl.TestFunction(SU)) * ds_if))
wU.ghostUpdate()

wiT = wT.array[iface_T]
idxU = np.column_stack([2 * iface_U, 2 * iface_U + 1])
wiU = wU.array[idxU]

Q = np.zeros((len(iface_T), 3))
okT = np.abs(wiT) > 1e-14
Q[okT, 0] = -rT[iface_T][okT] / wiT[okT]
okU = np.abs(wiU) > 1e-14
Q[:, 1:][okU] = -rU[idxU][okU] / wiU[okU]

V = np.column_stack([Th.x.array[iface_T], Uh.x.array[2 * iface_U], Uh.x.array[2 * iface_U + 1]])

print(f"[fenics {SIDE} thermoelastic] interface n={len(V)}")
print(f"T range: [{V[:,0].min():.6g}, {V[:,0].max():.6g}]")
print(f"ux range: [{V[:,1].min():.6g}, {V[:,1].max():.6g}]")
print(f"uy range: [{V[:,2].min():.6g}, {V[:,2].max():.6g}]")
print(f"NDOF = {ST.dofmap.index_map.size_global + 2 * SU.dofmap.index_map.size_global}")

# Export self-check
if not (np.isfinite(V).all() and np.isfinite(Q).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite values or fluxes")

_qin = np.asarray((imp or {}).get("normal_fluxes") or [], float)
if SIDE == "neumann" and _qin.size and np.abs(_qin).max() > 0 and np.abs(Q).max() < 1e-9 * np.abs(_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: recovered flux ~0 against nonzero imported flux")

# Per-level persistence
xyT_all = ST.tabulate_dof_coordinates()
with open(f"field_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy\n")
    for _i, (_px, _py) in enumerate(xyT_all[:, :2]):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(Th.x.array[_i]):.11e},"
                 f"{float(Uh.x.array[2 * _i]):.11e},{float(Uh.x.array[2 * _i + 1]):.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for _y, (_t, _ux, _uy), (_qn, _qx, _qy) in zip(y_if, V, Q):
        _f.write(f"{float(IFACE_X):.11e},{float(_y):.11e},{float(_t):.11e},{float(_ux):.11e},"
                 f"{float(_uy):.11e},{float(_qn):.11e},{float(-_qx):.11e},{float(-_qy):.11e}\n")

Path("exports.json").write_text(json.dumps({
    "field_name": "thermoelastic",
    "n_points": int(len(iface_T)),
    "coordinates": [[float(IFACE_X), float(y)] for y in y_if],
    "values": [[float(a_) for a_ in row] for row in V],
    "normal_fluxes": [[float(a_) for a_ in row] for row in Q],
}, indent=2))
