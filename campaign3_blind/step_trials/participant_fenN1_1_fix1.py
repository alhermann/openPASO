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
#    As shipped this is the RIGHT / Neumann side.
SIDE      = "neumann"     # "dirichlet" (import T, export flux) | "neumann"
PARTNER   = "left"        # the partner's `name` in your couple(...) call
X0, X1    = 0.6, 1.4      # this subdomain's x-extent
Y0, Y1    = 0.0, 1.0      # this subdomain's y-extent
IFACE_X   = 0.6           # the shared interface; must equal X0 or X1
K         = 5.0           # conductivity


def F_SRC(x, y):
    """Volumetric source, as a function of position."""
    return 5.0 * np.pi**2 * (1.4 - x) * np.sin(np.pi * y)

T_OUTER   = 0.0           # Dirichlet value on the NON-interface x-boundary
NX, NY    = 8, 10         # this subdomain's OWN mesh; need not match the partner
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux
# ─────────────────────────────────────────────────────────────────────────

# ── THE PER-LEVEL RULE (served). A ./config.json {"level": k, "nx": .., "ny": ..}
#    next to this script overrides NX, NY and names the level; the per-level
#    dumps below carry that level so the coarse levels survive the fine ones.
LEVEL = 1
if Path("config.json").is_file() or os.environ.get("OASIS_CONFIG_JSON"):
    try:
        _cfg = json.loads(Path("config.json").read_text() or "{}") if Path("config.json").is_file() else {}
        _cfg.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))
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
# THE SOLVE ITSELF
# ─────────────────────────────────────────────────────────────────────────

from dolfinx import log
log.set_log_level(log.LogLevel.INFO)

# Build mesh
domain = dmesh.create_rectangle(MPI.COMM_WORLD, [np.array([X0, Y0]), np.array([X1, Y1])], [NX, NY], dmesh.CellType.triangle)
domain.topology.create_entities(1)

V = fem.functionspace(domain, ("Lagrange", 1))

# Interpolate source term
f_h = fem.Function(V)
f_h.interpolate(lambda X: F_SRC(X[0], X[1]))

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Weak form: a(u,v) = L(v)
a = K * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L_vol = f_h * v * ufl.dx

# Tag the interface facet (x = IFACE_X)
iface_facets = np.sort(dmesh.locate_entities_boundary(domain, 1, lambda x: np.isclose(x[0], IFACE_X)))
iface_tags = dmesh.meshtags(domain, 1, iface_facets, np.ones(len(iface_facets), dtype=np.int64))
ds_if = ufl.Measure("ds", domain=domain, subdomain_data=iface_tags, subdomain_id=1)

# Get interface DOFs
coords = V.tabulate_dof_coordinates()
iface_dofs = np.where(np.isclose(coords[:, 0], IFACE_X))[0]
y_if = coords[iface_dofs, 1]

# Get outer Dirichlet DOFs (x = OUTER_X, y = Y0, y = Y1)
outer_dofs = np.where(np.isclose(coords[:, 0], OUTER_X) | np.isclose(coords[:, 1], Y0) | np.isclose(coords[:, 1], Y1))[0]

# Dirichlet BC on outer boundary
bc = fem.dirichletbc(T_OUTER, outer_dofs, V)

# Apply imported Neumann flux
if imp and imp.get("normal_fluxes"):
    q_in = sample(imp, "normal_fluxes", Q_INIT, y_if)
    q_func = fem.Function(V)
    q_func.x.array[iface_dofs] = q_in
    L_neumann = q_func * v * ds_if
else:
    L_neumann = ufl.Constant(domain, 0.0) * v * ds_if

# Assemble and solve
problem = LinearProblem(a, L_vol + L_neumann, bcs=[bc],
                        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                        petsc_options_prefix='run')
uh = problem.solve()

# Extract interface values
T = uh.x.array[iface_dofs]

# Compute consistent flux: q = -(K*u - b_vol)/h
# Assemble residual against volume load only (no BC, no Neumann term)
Amat = _fp.assemble_matrix(fem.form(a))
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

# Interface mass weights: w_i = int_Gamma phi_i ds
w_ = ufl.TestFunction(V)
w_form = fem.form(w_ * ds_if)
wvec = _fp.assemble_vector(w_form)
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

# Consistent flux recovery
Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

# Handle corner nodes (also on outer Dirichlet boundary)
suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    the interface check compares against)
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
    for (_px, _py), _u in zip(coords[:, :2], uh.x.array):
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
