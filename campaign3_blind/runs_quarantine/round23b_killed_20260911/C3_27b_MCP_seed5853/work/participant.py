"""DUNE-fem as the DIRICHLET side of a partitioned coupling (CG P1).

Reads ./config.json {"level":..,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":..,"reaction":..,"source_expr":"<f(x, y) as a Python expression, e.g.
'-10*x**3*y**3/3 + 16*x**2*y/5 - 2'; '0.0' when there is none>","iface":"left|right|bottom|top"}
("source_const": <number> is still accepted for a constant source).
Contract: reads ./imports.json (partner's interface FIELD values at its points),
imposes them as the interface Dirichlet trace, solves -div(k grad u) + c*u = f,
and exports values: [] (the trace is imposed, not owned) plus its OWN consistent
outward flux at the interior interface nodes.

WHAT IS SERVED HERE is the handshake (config + imports + trace mapping), the P1
consistent flux-recovery FORMULA, and the exports schema. THE MESH, THE WEAK
FORM AND THE SOLVE ARE YOURS to write -- see the banner. Get the DUNE-fem API
and a runnable P1 pattern from prepare_simulation(solver='dune',
physics='<your physics>').
"""
import json
from pathlib import Path

import numpy as np

CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV, CV, FV = CFG["k"], CFG.get("reaction", 0.0), CFG.get("source_const", 0.0)
IF = CFG.get("iface", "right")
HX, HY = (X1 - X0) / NX, (Y1 - Y0) / NY
# THE SOURCE COMES FROM config source_expr (the task's f(x, y) as a Python
# expression in x, y: '**' for powers, sin/cos/exp/sqrt/pi allowed), evaluated
# here for the consistent load below and, through src_ufl(x), for YOUR form.
# Measured: a run that carried the task's polynomial source nowhere (config
# source_const 0.0, the same zero in its form) converged at every level to a
# smooth field 1.2e-2 off the answer, order 0.00. A constant 'source_const'
# still works; when both are absent the source is zero.
SRC_EXPR = str(CFG.get("source_expr", "")).strip().replace("^", "**")
import math as _math
_SRC_CODE = compile(SRC_EXPR, "<source_expr>", "eval") if SRC_EXPR and SRC_EXPR not in ("0", "0.0") else None
def F_SRC(px, py):
    """The task's source f at a point, from config (a plain Python function)."""
    if _SRC_CODE is None:
        return float(FV)
    return float(eval(_SRC_CODE, {"__builtins__": {}}, {"x": float(px), "y": float(py), "sin": _math.sin, "cos": _math.cos,
                                                       "exp": _math.exp, "sqrt": _math.sqrt, "pi": _math.pi, "abs": abs}))
def src_ufl(xc):
    """The SAME source as a UFL expression of the spatial coordinate xc = SpatialCoordinate(space),
    for the load of your form (`src_ufl(x) * v * dx`); the recovery below integrates F_SRC."""
    import ufl as _ufl
    if _SRC_CODE is None:
        return float(FV)
    return eval(_SRC_CODE, {"__builtins__": {}}, {"x": xc[0], "y": xc[1], "sin": _ufl.sin, "cos": _ufl.cos,
                                                  "exp": _ufl.exp, "sqrt": _ufl.sqrt, "pi": _math.pi, "abs": abs})

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())
ax = 1 if IF in ("left", "right") else 0
pts_q = []
for _n, d in imp.items():
    co = d.get("coordinates") or []
    va = d.get("values") or []
    if co and va and len(va) == len(co):
        pts_q = sorted(zip([c[ax] for c in co], [float(v) for v in va]))
        break
def trace(t):
    """The partner's field value interpolated onto one of THIS side's interface
    points. The driver does NOT interpolate between meshes -- each participant
    maps the partner's samples onto its own points, here. Empty on iteration 1,
    so fall back to 0.0. Impose trace(coordinate) on the interface edge in the
    solve below."""
    if not pts_q:
        return 0.0
    xs = [p[0] for p in pts_q]; vs = [p[1] for p in pts_q]
    t = min(max(t, xs[0]), xs[-1])
    for a, b, va_, vb in zip(xs, xs[1:], vs, vs[1:]):
        if a <= t <= b:
            w = 0.0 if b == a else (t - a) / (b - a)
            return va_ + w * (vb - va_)
    return vs[-1]
# SIGN CONVENTION: this is the DIRICHLET side -- it IMPORTS the partner's field
# VALUE, imposes it as the interface trace, and exports its OWN outward flux. The
# two sides' fluxes carry OPPOSITE normals; export yours w.r.t. THIS side's
# outward normal and never write the partner's negated number.

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────
# HOLE 1 OF 2: THE MESH AND THE P1 SPACE ARE YOURS AND ARE NOT SERVED HERE.
# Build the SIMPLEX grid of this subdomain (X0..X1, Y0..Y1, NX x NY cells -- the
# P1 recovery below is exact on triangles, and a structuredGrid makes
# quadrilaterals) and the P1 space, from the DUNE-fem API and the measured
# gotchas in prepare_simulation(solver='dune', physics='<your physics>') and
# knowledge(topic='coupling', solver='dune'). Leave behind exactly these names:
#     gridView   the simplex grid view of this subdomain
#     space      the P1 Lagrange space on it
#     x          the SpatialCoordinate of that space
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────

# HOLE 1: Build simplex grid view and P1 space for subdomain A
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
from dune.fem.space import lagrange
from ufl import SpatialCoordinate

gridView = aluConformGrid(cartesianDomain([X0, Y0], [X1, Y1], [NX, NY]))
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)

# ── HANDSHAKE MAPPED ONTO YOUR SPACE (served: the partner's samples on THIS
#    side's interface dofs -- not the solve) ────────────────────────────────
# A discrete function carries the imposed interface datum: its dofs are run-time
# data, so the form text never changes between iterations (no re-JIT). The two
# interface ENDPOINTS keep the outer value (corner rule: they lie on the outer
# Dirichlet boundary, where trace() clamps; imposing the partner trace there is
# O(h)-wrong and caps the whole side at order 1).
_xd = np.array(space.interpolate(x[0], name="_xc").as_numpy)     # every dof's x, in dof order
_yd = np.array(space.interpolate(x[1], name="_yc").as_numpy)
_EPS = 1e-9 * max(X1 - X0, Y1 - Y0)
IF_COORD, IF_VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]
_along = _yd if IF_COORD == 0 else _xd                              # the coordinate that runs along the interface
_across = _xd if IF_COORD == 0 else _yd
_lo, _hi = (Y0, Y1) if IF_COORD == 0 else (X0, X1)
on_iface = np.abs(_across - IF_VAL) < _EPS                          # dof mask of the interface edge
_endpoint = (np.abs(_along - _lo) < _EPS) | (np.abs(_along - _hi) < _EPS)
gtrace = space.interpolate(0, name="gtrace")                        # the imposed interface datum
_gd = gtrace.as_numpy
_gd[:] = 0.0
for _i in np.where(on_iface & ~_endpoint)[0]:
    _gd[_i] = trace(float(_along[_i]))
# THE COEFFICIENTS AS UFL CONSTANTS (served: API plumbing, not the form). A bare
# Python float in a form is folded by UFL: `0.0 * u * v * dx` for a zero reaction
# becomes a domainless Zero and dies with "This integral is missing an
# integration domain" (measured in two of six worker trials). Use these in your
# form, never KV / CV themselves.
from dune.ufl import Constant as _Constant
K_UFL = _Constant(KV, name="k")
C_UFL = _Constant(CV, name="c")
# In your solve below: impose gtrace on the interface edge and the task's outer
# condition on the rest of the boundary. The interface edge as a UFL predicate:
#     conditional(lt(abs(x[IF_COORD] - IF_VAL), _EPS), 1, 0)
# (conditional, lt from ufl; abs is the Python built-in).

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────
# HOLE 2 OF 2: THE WEAK FORM, THE MATERIAL, THE SOURCE, THE BOUNDARY CONDITIONS
# AND THE LINEAR SOLVE ARE YOURS AND ARE NOT SERVED HERE. Write them for the
# problem you were given (-div(k grad u) + c*u = f on this subdomain, with the
# served UFL constants K_UFL and C_UFL as k and c -- never the bare floats),
# impose gtrace on the interface edge and the task's outer condition elsewhere,
# and solve. Leave behind exactly these names:
#     uh       the solved P1 function (uh = space.interpolate(0, name="uh");
#              scheme.solve(target=uh))
#     (F_SRC is ALREADY DEFINED above from config source_expr, and src_ufl(x)
#      is the same f for your form's load: `src_ufl(x) * v * dx` with
#      x = SpatialCoordinate(space). Redefine F_SRC only when the task's source
#      cannot be written as one expression string -- and then keep the form and
#      F_SRC the same f: the recovery below integrates F_SRC and refuses a
#      field whose interior residual against it is not small)
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────

# HOLE 2: Weak form for -div(k grad u) + c*u = f
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx, conditional, lt

u = TrialFunction(space)
v = TestFunction(space)

# Weak form: k*grad(u)*grad(v) + c*u*v = f*v
a = (K_UFL * dot(grad(u), grad(v)) + C_UFL * u * v) * dx
L = src_ufl(x) * v * dx

# Boundary conditions:
# - Interface edge (top for subdomain A): impose gtrace (Dirichlet from partner)
# - Outer boundary: u = 0
# Build indicator for interface edge
iface_indicator = conditional(lt(abs(x[IF_COORD] - IF_VAL), _EPS), 1, 0)
# Build indicator for outer boundary (not on interface)
outer_indicator = (conditional(lt(abs(x[0] - X0), _EPS), 1, 0) +
                   conditional(lt(abs(x[0] - X1), _EPS), 1, 0) +
                   conditional(lt(abs(x[1] - Y0), _EPS), 1, 0) +
                   conditional(lt(abs(x[1] - Y1), _EPS), 1, 0))
# Subtract interface from outer to avoid double-counting corners
outer_indicator = outer_indicator - iface_indicator

# Dirichlet BCs
dbc_interface = DirichletBC(space, gtrace, iface_indicator)
dbc_outer = DirichletBC(space, 0, outer_indicator)

scheme = galerkin([a == L, dbc_interface, dbc_outer], solver="cg", parameters={'linear.verbose': True})

uh = space.interpolate(0, name="uh")
info = scheme.solve(target=uh)
print(f"Solve converged: {info['converged']}, iterations: {info.get('iterations', 'N/A')}")

# ── VERTEX-ORDERED ARRAYS FROM YOUR MESH (served: mesh access, not the solve) ──
_idx = gridView.indexSet
node_coords = np.zeros((gridView.size(2), 2))
for _v in gridView.vertices:
    node_coords[_idx.index(_v)] = _v.geometry.center
elements = [[_idx.subIndex(_e, _i, 2) for _i in range(len(_e.geometry.corners))]
            for _e in gridView.elements]
if any(len(_e) != 3 for _e in elements):
    raise SystemExit("the P1 recovery below needs a SIMPLEX grid (triangles); this grid has "
                     f"cells with {sorted({len(_e) for _e in elements})} corners")
_key = {(round(float(_xd[_i]), 9), round(float(_yd[_i]), 9)): _i for _i in range(len(_xd))}
_d2v = np.array([_key[(round(float(_px), 9), round(float(_py), 9))] for _px, _py in node_coords])
u_vert = np.array(uh.as_numpy, float)[_d2v]                          # the solution in VERTEX order
f_vals = np.array([F_SRC(float(_px), float(_py)) for _px, _py in node_coords], float)
_if_nodes = [_n for _n in range(len(node_coords)) if abs(node_coords[_n][IF_COORD] - IF_VAL) < _EPS]
_if_nodes.sort(key=lambda _n: node_coords[_n][1 - IF_COORD])
interior = _if_nodes[1:-1]                                           # endpoints dropped (corner rule)

# ── CONSISTENT OUTWARD FLUX + EXPORTS -- the served recovery FORMULA ────────
# ONE formula, every backend, both sides. Your solve leaves an assembled operator
# K (the stiffness KV*grad(u).grad(v) plus any reaction CV*u*v) and a VOLUME load
# b_vol -- the P1-CONSISTENT element load (the source in the load as Me @ f_el,
# NOT a lumped nodal f), reaction and source in the load, with NO Dirichlet
# lifting and the constrained rows NOT zeroed. On the FREE interface rows the
# residual r = K u - b_vol then equals the interface functional, so
#
#     q_i = -r_i / w_i        w_i = interface nodal weight  (= h_if, uniform P1)
#
# is the consistent outward flux density at interface node i. It is mesh- and
# material-agnostic -- the same expression the Dirichlet and Neumann sides both
# use, and exactly what the interface check compares; keep the P1 consistent load
# or the recovery loses an order (a projected -k grad(u) on the boundary is only
# order ~1 there).
#
# Formed here over YOUR mesh with the standard P1 element matrices (Ke, Me below
# are the same for every P1 triangle -- they are not tied to any served mesh):
resid = np.zeros(len(node_coords))
_ku_part = np.zeros(len(node_coords))      # the operator side K u + c M u, for the served check below
_bv_part = np.zeros(len(node_coords))      # the served consistent load M f
for el in elements:                       # el = the 3 vertex ids of one triangle
    P = node_coords[list(el)]
    area = 0.5 * abs((P[1,0]-P[0,0])*(P[2,1]-P[0,1]) - (P[1,1]-P[0,1])*(P[2,0]-P[0,0]))
    gr = np.array([[P[1,1]-P[2,1], P[2,0]-P[1,0]],
                   [P[2,1]-P[0,1], P[0,0]-P[2,0]],
                   [P[0,1]-P[1,1], P[1,0]-P[0,0]]]) / (2.0 * area)
    Ke = KV * area * (gr @ gr.T)                       # element stiffness
    Me = area / 12.0 * (np.ones((3, 3)) + np.eye(3))   # CONSISTENT mass, not lumped
    ue = u_vert[list(el)]
    _ku_part[list(el)] += Ke @ ue + CV * (Me @ ue)
    _bv_part[list(el)] += Me @ f_vals[list(el)]
resid = _ku_part - _bv_part
h_if = HY if IF in ("left", "right") else HX
# THE FORM AND THE SERVED LOAD MUST AGREE (served check). On the free interior
# vertices r = K u - b_vol is only the quadrature difference between DUNE's
# integration of the source and the P1-consistent load, a few percent of the
# interface functional at most; a source or coefficient that is in config but
# not in your form (or the reverse) leaves an O(1) interior residual.
_ifset = set(_if_nodes)
_outer_set = {_n for _n, (_px, _py) in enumerate(node_coords)
              if (abs(_px - X0) < _EPS or abs(_px - X1) < _EPS or abs(_py - Y0) < _EPS or abs(_py - Y1) < _EPS)
              and _n not in _ifset}
_free = [_n for _n in range(len(node_coords)) if _n not in _ifset and _n not in _outer_set]
_r_in = max((abs(float(resid[_n])) for _n in _free), default=0.0)
_scale_in = max(max((abs(float(_ku_part[_n])) for _n in _free), default=0.0),
                max((abs(float(_bv_part[_n])) for _n in _free), default=0.0))
if _scale_in > 0 and _r_in > 0.25 * _scale_in:
    raise SystemExit(f"EXPORT SELF-CHECK: on the interior vertices your solution's operator side K u + c M u and the "
                     f"served consistent load M f differ by {_r_in:.3e}, {_r_in / _scale_in:.2f} of their size (a form "
                     f"that integrates the same f, k and c leaves a few percent): your form and config disagree on "
                     f"k ({KV}), the reaction ({CV}) or the source (source_expr={SRC_EXPR!r}) -- the load of your form "
                     f"must be src_ufl(x) * v * dx with the same numbers. Nothing was exported")
q_own = [float(-resid[n] / h_if) for n in interior]    # interior = interface ids[1:-1]
co_out = [[float(node_coords[n][0]), float(node_coords[n][1])] for n in interior]
# exports.json LAST (the driver takes its existence as proof of success).
# values: [] -- the DIRICHLET side does not OWN a value, it IMPOSED one; echoing
# the imposed trace back trips the driver's per-block change checks.
# ── EXPORT SELF-CHECK ─ keep this block. It stops the three exports that look
#    fine and are worthless: a non-finite field; a Neumann side whose imported
#    load never entered the assembled system (it returns the no-load answer and
#    a flux of ~0 against a nonzero partner); and a flux that is the partner's
#    array negated instead of a recovery from THIS side's own system.
# (json, numpy and Path are the imports at the top of this file)
_chk_vals = np.asarray([], float).ravel()
_chk_flux = np.asarray(q_own, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was exported")
_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                             for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))
if False and _chk_qin.size and np.abs(_chk_qin).max() > 0 and (
        np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max()):
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 against a "
                     "nonzero imported flux: the imported load never entered the "
                     "assembled system (the condition that integrates it is missing). "
                     "Fix the application; do not couple on")
if True and _chk_qin.shape == _chk_flux.shape and _chk_flux.size and (
        np.array_equal(_chk_flux, -_chk_qin)):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array "
                     "negated, bit for bit: a copy, not a recovery from this side's "
                     "own assembled system")
json.dump({"field_name": "u", "coordinates": co_out, "values": [],
           "normal_fluxes": q_own, "n_points": len(co_out)},
          open("exports.json", "w"))
# PER-LEVEL PERSISTENCE. Each level writes its own field file (named by the
# config level) so the coarse levels are not overwritten by the finest. Assemble
# the per-level field file for each side, named as your task prescribes, from
# THESE per-level files (interpolated to the task's probe points), never from
# one output the next level overwrites.
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(node_coords, u_vert):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\n")
# and its own interface trace and flux at THIS level's interface vertices
# (exports.json is overwritten by the next level; this file is not).
_uv = {(round(float(_px), 10), round(float(_py), 10)): float(_u) for (_px, _py), _u in zip(node_coords, u_vert)}
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    for (_px, _py), _q in zip(co_out, q_own):
        _f.write(f"{_px:.11e},{_py:.11e},{_uv.get((round(float(_px), 10), round(float(_py), 10)), float('nan')):.11e},{float(_q):.11e}\n")
# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its own (the
# audit and the hand-in read that exact shape); the descriptive line follows.
print(f"NDOF = {len(u_vert)}")
print(f"DUNE Dirichlet participant: NDOF = {len(u_vert)}  "
      f"max|u| = {float(np.abs(u_vert).max()) if len(u_vert) else 0:.6e}")

# ── WHAT YOUR TWO HOLES MUST LEAVE BEHIND ──────────────────────────────────
# The served code above uses these names; the elided blocks have to define
# every one, or the rest will not run:
#
#     hole 1:  gridView (a simplex grid view), space (P1 Lagrange on it),
#              x (SpatialCoordinate(space))
#     hole 2:  uh (the solved P1 function), F_SRC (your source as a Python
#              function of (x, y))
#
# OASiS does not serve the solve, but it will not make you guess which names
# the holes were filling.
