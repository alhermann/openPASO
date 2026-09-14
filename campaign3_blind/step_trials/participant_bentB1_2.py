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
import dolfinx                
import ufl
from dolfinx import default_scalar_type, fem, mesh as dmesh
from dolfinx.fem import petsc as _fp
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI

dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)

# ── EDIT THIS BLOCK ─ every number below is an ARBITRARY PLACEHOLDER.
#    Replace ALL of them with your problem's geometry, material and BCs.
SIDE      = "neumann"       # THIS SIDE IS NEUMANN (import flux, export T and flux)
PARTNER   = "left"          # the partner's `name` in your couple(...) call
X0, X1    = 0.5, 1.0        # this subdomain's x-extent (starts at interface)
Y0, Y1    = 0.0, 0.5        # this subdomain's y-extent
IFACE_AXIS = "x"            # primary interface axis (the first leg)
IFACE_X   = 0.5             # the shared interface coordinate
IFACE_SEGMENTS = [
    ("x", 0.5, 0.0, 0.5),   # leg 1: x=0.5, y from 0 to 0.5
    ("y", 0.5, 0.5, 1.0),   # leg 2: y=0.5, x from 0.5 to 1.0
]
K         = 2.5             # conductivity


def F_SRC(x, y):
    """Volumetric source (zero for this subdomain)."""
    return np.zeros_like(x)


T_OUTER   = 0.0             # Dirichlet on outer boundaries (y=0 and x=1)
NX, NY    = 16, 16          # mesh resolution
T_INIT    = 0.0             # initial guess for interface temperature
Q_INIT    = 0.0             # initial guess for interface flux
# ─────────────────────────────────────────────────────────────────────────

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

AX = 0 if IFACE_AXIS == "x" else 1
AL = 1 - AX
LO, HI = (X0, X1) if AX == 0 else (Y0, Y1)
ALO, AHI = (Y0, Y1) if AX == 0 else (X0, X1)
OUTER_X = LO if abs(IFACE_X - HI) < abs(IFACE_X - LO) else HI
S = 1.0 if IFACE_X > OUTER_X else -1.0


def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get(PARTNER) or None
    except json.JSONDecodeError:
        return None


TOL_IF = 1e-9 * max(X1 - X0, Y1 - Y0)


def arc_length(pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    if not IFACE_SEGMENTS:
        return pts[:, AL]
    s_out = np.full(len(pts), np.nan)
    base = 0.0
    for axis, pos, a, b in IFACE_SEGMENTS:
        ax = 0 if axis == "x" else 1
        al = 1 - ax
        lo, hi = (a, b) if a <= b else (b, a)
        on = ((np.abs(pts[:, ax] - pos) < TOL_IF) & (pts[:, al] >= lo - TOL_IF)
              & (pts[:, al] <= hi + TOL_IF) & np.isnan(s_out))
        s_out[on] = base + np.abs(pts[on, al] - a)
        base += abs(b - a)
    return np.where(np.isnan(s_out), 0.0, s_out)


def sample(imp, key, fallback, where):
    where = np.asarray(where, float)
    bent = where.ndim == 2
    n = len(where)
    if not imp or not imp.get("coordinates"):
        return np.full(n, float(fallback))
    pc = np.atleast_2d(np.asarray(imp["coordinates"], float))
    ys = arc_length(pc) if bent else pc[:, AL]
    target = arc_length(where) if bent else where
    vs = np.asarray(imp.get(key, []), float).ravel()
    if vs.size != ys.size:
        return np.full(n, float(fallback))
    o = np.argsort(ys)
    return np.interp(target, ys[o], vs[o])


imp = read_imports()

# ── THE SOLVE ITSELF ─ build mesh, space, weak form and linear solve
domain = dmesh.create_unit_square(MPI.COMM_WORLD, NX, NY)
cell_size_x = (X1 - X0) / NX
cell_size_y = (Y1 - Y0) / NY
geom = domain.geometry.x.copy()
geom[:, 0] = X0 + cell_size_x * geom[:, 0]
geom[:, 1] = Y0 + cell_size_y * geom[:, 1]
domain.geometry.x = geom

V = fem.functionspace(domain, ('Lagrange', 1))
uh = fem.Function(V)

fdim = domain.topology.dim - 1
domain.topology.create_connectivity(fdim, domain.topology.dim)

def interface_marker_left(x):
    tol = 1e-14
    return np.logical_and(
        np.abs(x[0] - 0.5) < tol,
        np.abs(x[1] - 0.5) > tol
    )

def interface_marker_top(x):
    tol = 1e-14
    return np.logical_and(
        np.abs(x[1] - 0.5) < tol,
        np.abs(x[0] - 0.5) > tol
    )

facets_left = dmesh.locate_entities_boundary(domain, fdim, interface_marker_left)
facets_top = dmesh.locate_entities_boundary(domain, fdim, interface_marker_top)
interface_facets = np.unique(np.concatenate([facets_left, facets_top]))
iface_meshtags = dmesh.meshtags(domain, fdim, interface_facets, 1)
ds_if = ufl.Measure("ds", domain, subdomain_data=iface_meshtags)

def outer_bottom(x):
    return np.abs(x[1]) < 1e-14

def outer_right(x):
    return np.abs(x[0] - 1.0) < 1e-14

outer_facets_bottom = dmesh.locate_entities_boundary(domain, fdim, outer_bottom)
outer_facets_right = dmesh.locate_entities_boundary(domain, fdim, outer_right)
outer_facets = np.unique(np.concatenate([outer_facets_bottom, outer_facets_right]))

iface_dofs = fem.locate_dofs_topological(V, fdim, interface_facets)
outer_dofs = fem.locate_dofs_topological(V, fdim, outer_facets)

g_func = fem.Function(V)
if imp and imp.get("normal_fluxes"):
    q_in = np.array(imp["normal_fluxes"])
    g_func.x.array[iface_dofs] = q_in
else:
    g_func.x.array[iface_dofs] = Q_INIT

u, v = fem.TestFunction(V), fem.TestFunction(V)

f_h = fem.Function(V)
f_h.interpolate(lambda X: F_SRC(X[0], X[1]))
L_vol = fem.form(f_h * v * ufl.dx)

a = fem.form(K * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx)
L = L_vol + fem.form(g_func * v * ds_if)

bc_outer = fem.dirichletbc(T_OUTER, outer_dofs, V)

problem = LinearProblem(a, L, bcs=[bc_outer],
                        petsc_options={'ksp_type': 'preonly', 'pc_type': 'lu'},
                        petsc_options_prefix='run')
uh = problem.solve()

xy = V.tabulate_dof_coordinates()
y_if = xy[iface_dofs, :]

Amat = _fp.assemble_matrix(fem.form(a))
Amat.assemble()
bvec = _fp.assemble_vector(fem.form(L_vol))
bvec.ghostUpdate()
r = Amat.createVecLeft()
Amat.mult(uh.x.petsc_vec, r)
r.axpy(-1.0, bvec)

wvec = _fp.assemble_vector(fem.form(v * ds_if))
wvec.ghostUpdate()
wi = wvec.array[iface_dofs]

Q = np.zeros(len(iface_dofs))
ok = np.abs(wi) > 1e-14
Q[ok] = -r.array[iface_dofs][ok] / wi[ok]

suspect = np.isin(iface_dofs, outer_dofs) | ~ok
good = np.where(~suspect)[0]
if len(good):
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

T = uh.x.array[iface_dofs]
print(f"[fenics {SIDE}] interface n={len(T)} "
      f"T=[{T.min():.6g},{T.max():.6g}] q=[{Q.min():.6g},{Q.max():.6g}]")

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
if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size \
        and np.array_equal(_chk_flux, -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's "
                     "array negated, bit for bit: a copy, not a recovery from "
                     "this side's own assembled system")

with open(f"field_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(xy[:, :2], uh.x.array):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_u):.11e}\n")
with open(f"interface_level{LEVEL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    _pts_if = np.column_stack([y_if[:, 0], y_if[:, 1]])
    for (_px, _py), _t, _q in zip(_pts_if, T, Q):
        _f.write(f"{float(_px):.11e},{float(_py):.11e},{float(_t):.11e},{float(_q):.11e}\n")
Path("exports.json").write_text(json.dumps({
    "field_name": "temperature",
    "n_points": int(len(iface_dofs)),
    "coordinates": y_if.tolist(),
    "values": [float(t) for t in T],
    "normal_fluxes": [float(q) for q in Q],
}, indent=2))
