"""FEniCSx (dolfinx) participant B for OASiS couple driver.

Steady thermoelasticity on ONE rectangular subdomain: rectangle (0.8, 1.6) x (0, 1).
Plane strain, small strain. BOTH fields across ONE interface at x = 0.8.
NEUMANN side: imports partner's flux/traction, exports values + own flux.
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

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
SIDE      = "neumann"     # "dirichlet" | "neumann" (this is NEUMANN)
PARTNER   = "A"           # the partner's name in your couple(...) call
X0, X1    = 0.8, 1.6      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
IFACE_X   = 0.8           # the shared interface; must equal X0 or X1
K         = 3.0           # conductivity
LAM, MU   = 400.0, 500.0  # Lame parameters (plane strain)
BETA      = 1.0           # thermal stress coefficient
NX, NY    = 8, 10         # this subdomain's OWN mesh (h = 0.1)
T_INIT, UX_INIT, UY_INIT = 0.0, 0.0, 0.0   # iteration-1 fallback field
Q_INIT    = (0.0, 0.0, 0.0)                # iteration-1 fallback flux/traction
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served)
LEVEL = 1
if Path("config.json").is_file():
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}")
        LEVEL = int(_cfg.get("level", LEVEL))
        NX = int(_cfg.get("nx", NX))
        NY = int(_cfg.get("ny", NY))
    except (ValueError, TypeError, json.JSONDecodeError):
        pass

log.set_log_level(log.LogLevel.INFO)

OUTER_X = X0 if abs(IFACE_X - X1) < abs(IFACE_X - X0) else X1
S = 1.0 if IFACE_X > OUTER_X else -1.0     # outward normal at interface = S*e_x


def F_T(x, y):
    """Heat source f_T(x, y) as NumPy arrays."""
    return (-0.6*np.pi**2*x*(x - 1.6)*(5*x + 2) + 18.0*x - 7.2)*np.sin(np.pi*y)


def F_U(x, y):
    """Body force (f_x, f_y) as two NumPy arrays."""
    fx = (-10.0*np.pi**2*x**2*np.sin(np.pi*y) - 3.0*x**2*np.sin(np.pi*y) 
          + 27.0*np.pi*x**2*np.cos(np.pi*y) + 2.4*x*np.sin(np.pi*y) 
          + 16.0*np.pi**2*x*np.sin(np.pi*y) - 28.8*np.pi*x*np.cos(np.pi*y) 
          + 56.64*np.sin(np.pi*y))
    fy = (-14.0*np.pi**2*x**2*(x - 1.6)*np.sin(np.pi*y) 
          - 0.2*np.pi*x*(x - 1.6)*(5*x + 2)*np.cos(np.pi*y) 
          + 20.0*x*np.sin(np.pi*y) + 18.0*np.pi*x*np.cos(np.pi*y) 
          + 18.0*np.pi*(x - 1.6)*np.cos(np.pi*y) 
          + (10.0*x - 16.0)*np.sin(np.pi*y))
    return fx, fy


T_OUTER   = 0.0           # T on the whole NON-interface boundary
UX_OUTER, UY_OUTER = 0.0, 0.0   # u on the whole NON-interface boundary


def read_imports():
    """imports.json is {partner_name: InterfaceData}; `{}` on iteration 1."""
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

# ── HOLE 1: mesh and spaces ────────────────────────────────────────────────
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [[X0, Y0], [X1, Y1]], [NX, NY], dmesh.CellType.triangle)
ST = fem.functionspace(domain, ("Lagrange", 1))
SU = fem.functionspace(domain, ("Lagrange", 1, (2,)))
# ───────────────────────────────────────────────────────────────────────────

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
    sys.exit(f"no interface DOFs at x={IFACE_X}: this subdomain spans [{X0},{X1}]")

corner = (np.abs(y_if - Y0) < 1e-10) | (np.abs(y_if - Y1) < 1e-10)
iface_bc_T = iface_T[~corner]
iface_bc_U = iface_U[~corner]

facets = dmesh.locate_entities_boundary(domain, fdim, lambda x: np.isclose(x[0], IFACE_X))
tags = dmesh.meshtags(domain, fdim, np.sort(facets), np.full(len(facets), 7, dtype=np.int32))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=tags)(7)

tT, vT = ufl.TrialFunction(ST), ufl.TestFunction(ST)
uu, vu = ufl.TrialFunction(SU), ufl.TestFunction(SU)

# ── THE INTERFACE ROLE (served) ────────────────────────────────────────────
bcs_if_T, bcs_if_U = [], []
L_T_if, L_U_if = 0, 0
if SIDE == "dirichlet":
    g = sample(imp, "values", (T_INIT, UX_INIT, UY_INIT), y_if)
    gT = fem.Function(ST)
    gT.x.array[:] = 0.0
    gT.x.array[iface_T] = g[:, 0]
    gU = fem.Function(SU)
    gU.x.array[:] = 0.0
    gU.x.array[2 * iface_U] = g[:, 1]
    gU.x.array[2 * iface_U + 1] = g[:, 2]
    bcs_if_T = [fem.dirichletbc(gT, iface_bc_T.astype(np.int32))]
    bcs_if_U = [fem.dirichletbc(gU, iface_bc_U.astype(np.int32))]
else:
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

# ── HOLE 2: forms, sources, outer BCs, solves ──────────────────────────────
aT = K * ufl.inner(ufl.grad(tT), ufl.grad(vT)) * ufl.dx

eps_u = ufl.sym(ufl.grad(uu))
eps_v = ufl.sym(ufl.grad(vu))
au = 2*MU*ufl.inner(eps_u, eps_v)*ufl.dx + LAM*ufl.tr(eps_u)*ufl.tr(eps_v)*ufl.dx

fT_h = fem.Function(ST)
fT_h.interpolate(lambda X: F_T(X[0], X[1]))
L_T_vol = fT_h * vT * ufl.dx

fU_h = fem.Function(SU)
fU_h.interpolate(lambda X: np.vstack((F_U(X[0], X[1])[0], F_U(X[0], X[1])[1])))

# Outer Dirichlet conditions
bcs_T = [fem.dirichletbc(fem.Constant(domain, T_OUTER), outer_T.astype(np.int32), ST)]
bcs_U = [fem.dirichletbc(fem.Constant(domain, np.array([UX_OUTER, UY_OUTER])), outer_U.astype(np.int32), SU)]

# Solve heat problem first
pT = LinearProblem(aT, L_T_vol + L_T_if, bcs=bcs_T + bcs_if_T,
                   petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                   petsc_options_prefix='run')
Th = pT.solve()

# Volume load for elasticity: body force PLUS thermal term (both are volume loads)
L_U_vol = ufl.inner(fU_h, vu) * ufl.dx + BETA * Th * ufl.div(vu) * ufl.dx

# Solve elasticity problem
pU = LinearProblem(au, L_U_vol + L_U_if, bcs=bcs_U + bcs_if_U,
                   petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                   petsc_options_prefix='run2')
Uh = pU.solve()
# ───────────────────────────────────────────────────────────────────────────

# ── DID THE INTERFACE DATA ENTER THE SOLVE? (served) ──────────────────────
if SIDE == "dirichlet" and len(iface_bc_T):
    _gap = float(np.abs(Th.x.array[iface_bc_T] - gT.x.array[iface_bc_T]).max())
    _gapU = float(np.abs(Uh.x.array[2 * iface_bc_U] - gU.x.array[2 * iface_bc_U]).max())
    if _gap > 1e-9 * max(1.0, float(np.abs(gT.x.array).max())) or \
            _gapU > 1e-9 * max(1.0, float(np.abs(gU.x.array).max())):
        raise SystemExit("the imposed interface trace is not in the solution")

# ── CONSISTENT OUTWARD FLUX AND TRACTION (served) ─────────────────────────
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
suspect = np.isin(iface_T, outer_T) | ~okT | ~okU.all(axis=1)
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

V = np.column_stack([Th.x.array[iface_T], Uh.x.array[2 * iface_U], Uh.x.array[2 * iface_U + 1]])
print(f"[fenics {SIDE} thermoelastic] interface n={len(V)} "
      f"T=[{V[:,0].min():.6g},{V[:,0].max():.6g}] ux=[{V[:,1].min():.6g},{V[:,1].max():.6g}] "
      f"uy=[{V[:,2].min():.6g},{V[:,2].max():.6g}] qn=[{Q[:,0].min():.6g},{Q[:,0].max():.6g}]")
print(f"NDOF = {ST.dofmap.index_map.size_global + 2 * SU.dofmap.index_map.size_global}")

# ── EXPORT SELF-CHECK ─ keep this block ───────────────────────────────────
if not (np.isfinite(V).all() and np.isfinite(Q).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes")
_qin = np.asarray((imp or {}).get("normal_fluxes") or [], float)
if SIDE == "neumann" and _qin.size and np.abs(_qin).max() > 0 and np.abs(Q).max() < 1e-9 * np.abs(_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: recovered flux ~0 against nonzero imported flux")
if SIDE == "neumann" and _qin.size:
    _qa = sample(imp, "normal_fluxes", Q_INIT, y_if)
    _int = ~suspect
    for c in range(3):
        _sc = np.abs(_qa[_int, c]).max()
        if _sc > 0 and np.abs(Q[_int, c] + _qa[_int, c]).max() / _sc > 0.3:
            raise SystemExit(f"EXPORT SELF-CHECK: component {c} does not match applied load")
if SIDE == "dirichlet" and _qin.shape == Q.shape and Q.size and np.array_equal(Q, -_qin):
    raise SystemExit("EXPORT SELF-CHECK: exported flux is partner's negated bit-for-bit")

# PER-LEVEL PERSISTENCE
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
