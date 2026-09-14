"""DUNE-fem as the DIRICHLET side of a partitioned coupling (CG P1).
Reads ./config.json, imposes interface Dirichlet trace from imports.json,
solves -div(k grad u) + c*u = f, exports outward flux at interface.
"""
import json
from pathlib import Path
import numpy as np
import math as _math

CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV, CV = CFG["k"], CFG.get("reaction", 0.0)
IF = CFG.get("iface", "right")
HX, HY = (X1 - X0) / NX, (Y1 - Y0) / NY

SRC_EXPR = str(CFG.get("source_expr", "")).strip().replace("^", "**")
_SRC_CODE = compile(SRC_EXPR, "<source_expr>", "eval") if SRC_EXPR and SRC_EXPR not in ("0", "0.0") else None

def F_SRC(px, py):
    if _SRC_CODE is None:
        return 0.0
    return float(eval(_SRC_CODE, {"__builtins__": {}}, 
                     {"x": float(px), "y": float(py), "sin": _math.sin, "cos": _math.cos,
                      "exp": _math.exp, "sqrt": _math.sqrt, "pi": _math.pi, "abs": abs}))

# Partner's interface samples
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
    if not pts_q:
        return 0.0
    xs = [p[0] for p in pts_q]; vs = [p[1] for p in pts_q]
    t = min(max(t, xs[0]), xs[-1])
    for a, b, va_, vb in zip(xs, xs[1:], vs, vs[1:]):
        if a <= t <= b:
            w = 0.0 if b == a else (t - a) / (b - a)
            return va_ + w * (vb - va_)
    return vs[-1]

# HOLE 1: Mesh and P1 space - use simplex grid (triangles)
from dune.grid import cartesianDomain
from dune.alugrid import aluConformGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC, Constant
from ufl import TrialFunction, TestFunction, dot, grad, dx, conditional, lt, SpatialCoordinate

domain = cartesianDomain([X0, Y0], [X1, Y1], [NX, NY])
gridView = aluConformGrid(domain)
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)

# Handshake mapped onto space
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

K_UFL = Constant(KV, name="k")
C_UFL = Constant(CV, name="c")

# HOLE 2: Weak form and solve
u = TrialFunction(space)
v = TestFunction(space)

# Source as UFL expression
def src_ufl(xc):
    if _SRC_CODE is None:
        return Constant(0.0, name="f")
    return eval(_SRC_CODE, {"__builtins__": {}}, 
               {"x": xc[0], "y": xc[1], "sin": lambda x: x, "cos": lambda x: x,
                "exp": lambda x: x, "sqrt": lambda x: x, "pi": _math.pi, "abs": abs})

a = dot(K_UFL * grad(u), grad(v)) * dx + C_UFL * u * v * dx
L = src_ufl(x) * v * dx

# Outer boundary BC: u=0 on x=0, x=1, y=0 (but NOT y=Y1 which is the interface)
outer_indicator = conditional(lt(abs(x[0] - X0), _EPS), 1, 0) + \
                  conditional(lt(abs(x[0] - X1), _EPS), 1, 0) + \
                  conditional(lt(abs(x[1] - Y0), _EPS), 1, 0)
outer_bc = DirichletBC(space, 0, outer_indicator)

# Interface BC: impose gtrace on the top edge (interface)
iface_indicator = conditional(lt(abs(x[1] - Y1), _EPS), 1, 0)
iface_bc = DirichletBC(space, gtrace, iface_indicator)

uh = space.interpolate(0, name="uh")
scheme = galerkin([a == L, outer_bc, iface_bc], solver='cg')
info = scheme.solve(target=uh)
print(f"DUNE solve converged: {info['converged']}")

# Vertex-ordered arrays
_idx = gridView.indexSet
node_coords = np.zeros((gridView.size(2), 2))
for _v in gridView.vertices:
    node_coords[_idx.index(_v)] = _v.geometry.center
elements = [[_idx.subIndex(_e, _i, 2) for _i in range(len(_e.geometry.corners))]
            for _e in gridView.elements]
_key = {(round(float(_xd[_i]), 9), round(float(_yd[_i]), 9)): _i for _i in range(len(_xd))}
_d2v = np.array([_key[(round(float(_px), 9), round(float(_py), 9))] for _px, _py in node_coords])
u_vert = np.array(uh.as_numpy, float)[_d2v]
f_vals = np.array([F_SRC(float(_px), float(_py)) for _px, _py in node_coords], float)
_if_nodes = [_n for _n in range(len(node_coords)) if abs(node_coords[_n][IF_COORD] - IF_VAL) < _EPS]
_if_nodes.sort(key=lambda _n: node_coords[_n][1 - IF_COORD])
interior = _if_nodes[1:-1]

# Consistent outward flux
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
q_own = [float(-resid[n] / h_if) for n in interior]
co_out = [[float(node_coords[n][0]), float(node_coords[n][1])] for n in interior]

json.dump({"field_name": "u", "coordinates": co_out, "values": [],
           "normal_fluxes": q_own, "n_points": len(co_out)}, open("exports.json", "w"))

_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(node_coords, u_vert):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\n")

_uv = {(round(float(_px), 10), round(float(_py), 10)): float(_u) for (_px, _py), _u in zip(node_coords, u_vert)}
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    for (_px, _py), _q in zip(co_out, q_own):
        _f.write(f"{_px:.11e},{_py:.11e},{_uv.get((round(float(_px), 10), round(float(_py), 10)), 0.0):.11e},{float(_q):.11e}\n")

print(f"NDOF = {len(u_vert)}")
print(f"DUNE Dirichlet participant: NDOF = {len(u_vert)}  max|u| = {float(np.abs(u_vert).max()):.6e}")
