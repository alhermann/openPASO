"""DUNE-fem as the DIRICHLET side of a partitioned coupling (CG P1).

Reads ./config.json {"level":..,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":..,"reaction":..,"source_expr":"<f(x, y) as a Python expression>","iface":"left|right|bottom|top"}
Contract: reads ./imports.json (partner's interface FIELD values at its points),
imposes them as the interface Dirichlet trace, solves -div(k grad u) + c*u = f,
and exports values: [] (the trace is imposed, not owned) plus its OWN consistent
outward flux at the interior interface nodes.
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
# THE SOURCE COMES FROM config source_expr
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
    """The SAME source as a UFL expression of the spatial coordinate xc = SpatialCoordinate(space)."""
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
    """The partner's field value interpolated onto one of THIS side's interface points."""
    if not pts_q:
        return 0.0
    xs = [p[0] for p in pts_q]; vs = [p[1] for p in pts_q]
    t = min(max(t, xs[0]), xs[-1])
    for a, b, va_, vb in zip(xs, xs[1:], vs, vs[1:]):
        if a <= t <= b:
            w = 0.0 if b == a else (t - a) / (b - a)
            return va_ + w * (vb - va_)
    return vs[-1]

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────
# HOLE 1 OF 2: THE MESH AND THE P1 SPACE ARE YOURS
# Build the SIMPLEX grid of this subdomain and the P1 space
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
gridView = aluConformGrid(cartesianDomain([X0, Y0], [X1, Y1], [NX, NY]))
from dune.fem.space import lagrange
space = lagrange(gridView, order=1)
from ufl import SpatialCoordinate
x = SpatialCoordinate(space)
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────

# ── HANDSHAKE MAPPED ONTO YOUR SPACE ───────────────────────────────────────
_xd = np.array(space.interpolate(x[0], name="_xc").as_numpy)
_yd = np.array(space.interpolate(x[1], name="_yc").as_numpy)
_EPS = 1e-9 * max(X1 - X0, Y1 - Y0)
IF_COORD, IF_VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]
_along = _yd if IF_COORD == 0 else _xd
_across = _xd if IF_COORD == 0 else _yd
_lo, _hi = (Y0, Y1) if IF_COORD == 0 else (X0, X1)
on_iface = np.abs(_across - IF_VAL) < _EPS
_endpoint = (np.abs(_along - _lo) < _EPS) | (np.abs(_along - _hi) < _EPS)
gtrace = space.interpolate(0, name="gtrace")
_gd = gtrace.as_numpy
_gd[:] = 0.0
for _i in np.where(on_iface & ~_endpoint)[0]:
    _gd[_i] = trace(float(_along[_i]))

from dune.ufl import Constant as _Constant
K_UFL = _Constant(KV, name="k")
C_UFL = _Constant(CV, name="c")

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────
# HOLE 2 OF 2: THE WEAK FORM, MATERIAL, SOURCE, BOUNDARY CONDITIONS AND SOLVE
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx, conditional, lt

u = TrialFunction(space)
v = TestFunction(space)

# Weak form: -div(k grad u) + c*u = f
# Variational form: k*grad(u)*grad(v) + c*u*v = f*v
a = (K_UFL * dot(grad(u), grad(v)) + C_UFL * u * v) * dx
L = src_ufl(x) * v * dx

# Boundary conditions:
# - Outer boundary: u = 0
# - Interface edge: u = gtrace (imposed from partner)

# Build indicators for boundaries using UFL conditionals
eps_bc = 1e-8 * max(X1 - X0, Y1 - Y0)

# Interface indicator: points on the interface edge
if IF == "top":
    iface_indicator = conditional(lt(abs(x[1] - Y1), eps_bc), 1, 0)
elif IF == "bottom":
    iface_indicator = conditional(lt(abs(x[1] - Y0), eps_bc), 1, 0)
elif IF == "left":
    iface_indicator = conditional(lt(abs(x[0] - X0), eps_bc), 1, 0)
else:  # right
    iface_indicator = conditional(lt(abs(x[0] - X1), eps_bc), 1, 0)

# Outer boundary indicator: all edges except interface
outer_left = conditional(lt(abs(x[0] - X0), eps_bc), 1, 0)
outer_right = conditional(lt(abs(x[0] - X1), eps_bc), 1, 0)
outer_bottom = conditional(lt(abs(x[1] - Y0), eps_bc), 1, 0)
outer_top = conditional(lt(abs(x[1] - Y1), eps_bc), 1, 0)

# For the outer boundary, we need to exclude the interface
if IF == "top":
    # Outer = left + right + bottom
    outer_indicator = outer_left + outer_right + outer_bottom
elif IF == "bottom":
    outer_indicator = outer_left + outer_right + outer_top
elif IF == "left":
    outer_indicator = outer_right + outer_bottom + outer_top
else:  # right
    outer_indicator = outer_left + outer_bottom + outer_top

# Dirichlet BC: u=0 on outer boundary
dbc_outer = DirichletBC(space, 0, outer_indicator)

# Dirichlet BC: u=gtrace on interface edge
dbc_iface = DirichletBC(space, gtrace, iface_indicator)

scheme = galerkin([a == L, dbc_outer, dbc_iface], solver="cg", parameters={"linear.verbose": True})

uh = space.interpolate(0, name="uh")
info = scheme.solve(target=uh)

print(f"Solver converged: {info['converged']}")
print(f"Iterations: {info.get('iterations', 'N/A')}")
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─────────────────────────────────────

# ── VERTEX-ORDERED ARRAYS FROM YOUR MESH ───────────────────────────────────
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
u_vert = np.array(uh.as_numpy, float)[_d2v]
f_vals = np.array([F_SRC(float(_px), float(_py)) for _px, _py in node_coords], float)
_if_nodes = [_n for _n in range(len(node_coords)) if abs(node_coords[_n][IF_COORD] - IF_VAL) < _EPS]
_if_nodes.sort(key=lambda _n: node_coords[_n][1 - IF_COORD])
interior = _if_nodes[1:-1]

# ── CONSISTENT OUTWARD FLUX + EXPORTS ──────────────────────────────────────
resid = np.zeros(len(node_coords))
_ku_part = np.zeros(len(node_coords))
_bv_part = np.zeros(len(node_coords))
for el in elements:
    P = node_coords[list(el)]
    area = 0.5 * abs((P[1,0]-P[0,0])*(P[2,1]-P[0,1]) - (P[1,1]-P[0,1])*(P[2,0]-P[0,0]))
    gr = np.array([[P[1,1]-P[2,1], P[2,0]-P[1,0]],
                   [P[2,1]-P[0,1], P[0,0]-P[2,0]],
                   [P[0,1]-P[1,1], P[1,0]-P[0,0]]]) / (2.0 * area)
    Ke = KV * area * (gr @ gr.T)
    Me = area / 12.0 * (np.ones((3, 3)) + np.eye(3))
    ue = u_vert[list(el)]
    _ku_part[list(el)] += Ke @ ue + CV * (Me @ ue)
    _bv_part[list(el)] += Me @ f_vals[list(el)]
resid = _ku_part - _bv_part
h_if = HY if IF in ("left", "right") else HX

_ifset = set(_if_nodes)
_outer_set = {_n for _n, (_px, _py) in enumerate(node_coords)
              if (abs(_px - X0) < _EPS or abs(_px - X1) < _EPS or abs(_py - Y0) < _EPS or abs(_py - Y1) < _EPS)
              and _n not in _ifset}
_free = [_n for _n in range(len(node_coords)) if _n not in _ifset and _n not in _outer_set]
_r_in = max((abs(float(resid[_n])) for _n in _free), default=0.0)
_scale_in = max(max((abs(float(_ku_part[_n])) for _n in _free), default=0.0),
                max((abs(float(_bv_part[_n])) for _n in _free), default=0.0))
if _scale_in > 0 and _r_in > 0.25 * _scale_in:
    raise SystemExit(f"EXPORT SELF-CHECK: interior residual too large ({_r_in:.3e}), form may disagree with config")

q_own = [float(-resid[n] / h_if) for n in interior]
co_out = [[float(node_coords[n][0]), float(node_coords[n][1])] for n in interior]

_chk_vals = np.asarray([], float).ravel()
_chk_flux = np.asarray(q_own, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes")

json.dump({"field_name": "u", "coordinates": co_out, "values": [],
           "normal_fluxes": q_own, "n_points": len(co_out)},
          open("exports.json", "w"))

_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(node_coords, u_vert):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\n")

_uv = {(round(float(_px), 10), round(float(_py), 10)): float(_u) for (_px, _py), _u in zip(node_coords, u_vert)}
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    for (_px, _py), _q in zip(co_out, q_own):
        _f.write(f"{_px:.11e},{_py:.11e},{_uv.get((round(float(_px), 10), round(float(_py), 10)), float('nan')):.11e},{float(_q):.11e}\n")

print(f"NDOF = {len(u_vert)}")
print(f"DUNE Dirichlet participant: NDOF = {len(u_vert)}  max|u| = {float(np.abs(u_vert).max()) if len(u_vert) else 0:.6e}")
