"""NGSolve participant for subdomain A (Neumann side)."""
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

SIDE = "neumann"
PARTNER = "B"
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.0, 0.625
IFACE_Y = 0.625
LAM, MU = 480, 1200
TI_X, TI_Y = 0.0, 0.0

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except:
        return None
    return d.get(PARTNER) or None

def sample_vector(imp, key, fallback, x):
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(x), 1))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != xs.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(x), 1))
    o = np.argsort(xs)
    return np.column_stack([np.interp(x, xs[o], vs[o, c]) for c in range(vs.shape[1])])

import os
mesh_level = int(os.environ.get('MESH_LEVEL', '1'))
NX, NY = {1: (8, 5), 2: (16, 10), 3: (32, 20)}.get(mesh_level, (8, 5))
print(f"Subdomain A: NX={NX}, NY={NY}, level={mesh_level}")

geo = SplineGeometry()
p = [geo.AddPoint(X0, Y0), geo.AddPoint(X1, Y0), geo.AddPoint(X1, Y1), geo.AddPoint(X0, Y1)]
geo.Append(["line", p[0], p[1]], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p[1], p[2]], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p[2], p[3]], leftdomain=1, rightdomain=0, bc="interface")
geo.Append(["line", p[3], p[0]], leftdomain=1, rightdomain=0, bc="left")

mesh = Mesh(geo.GenerateMesh(maxh=min((X1-X0)/NX, (Y1-Y0)/NY)))
fes = VectorH1(mesh, order=1)
u, v = fes.TnT()

a = BilinearForm(fes)
a += InnerProduct(2*MU*Sym(Grad(u)) + LAM*Trace(Sym(Grad(u)))*Id(2), Sym(Grad(v))) * dx
a.Assemble()

f_vol = LinearForm(fes)
fx = (-63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 
      + 216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 - 288*x*y**2/3125 
      + 1992*x*y/15625 + 33*x/3125 - 36*y**2/625 + 54*y/625)
fy = (-108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 
      - 279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 
      + 9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 - 12*y**2/3125 + 51*y/3125 - 42/3125)
f_vol += CoefficientFunction((fx, fy)) * v * dx
f_vol.Assemble()

tol = 1e-9
verts = list(mesh.vertices)
vdof = {i: tuple(int(d) for d in fes.GetDofNrs(NodeId(VERTEX, i))) for i in range(len(verts))}

iface_v, outer_v = [], []
for i, v in enumerate(verts):
    xv, yv = v.point[0], v.point[1]
    on_iface = abs(yv - IFACE_Y) < tol
    on_outer = abs(yv - Y0) < tol or abs(xv - X0) < tol or abs(xv - X1) < tol
    is_corner = on_iface and (abs(xv - X0) < tol or abs(xv - X1) < tol)
    if on_iface and not is_corner:
        iface_v.append(i)
    elif on_outer:
        outer_v.append(i)

iface_v.sort(key=lambda idx: verts[idx].point[0])
iface_x = np.array([verts[idx].point[0] for idx in iface_v])

outer_dofs = np.array(sorted({vdof[i][c] for i in outer_v for c in [0,1]}))
iface_dofs = [(vdof[i][0], vdof[i][1]) for i in iface_v]

imp = read_imports()
gfu = GridFunction(fes)

if SIDE == "neumann":
    t_if = sample_vector(imp, "normal_fluxes", (TI_X, TI_Y), iface_x)
    
    # Apply traction via LinearForm with CF
    tx_cf = CF(lambda x, y: 0.0)
    ty_cf = CF(lambda x, y: 0.0)
    
    # Interpolate traction values onto interface
    f_trac = LinearForm(fes)
    # Use GridFunction to interpolate traction
    gfun = GridFunction(fes)
    for k, (dx, dy) in enumerate(iface_dofs):
        gfun.vec[dx], gfun.vec[dy] = t_if[k, 0], t_if[k, 1]
    
    # Add traction contribution manually
    b_trac = f_vol.vec.CreateVector()
    b_trac.data = 0.0
    # For Neumann BC, add integral(t . v) ds
    # We'll approximate by adding nodal contributions
    for k, (dx, dy) in enumerate(iface_dofs):
        # Simple approximation: add traction times shape function integral
        h_elem = (X1-X0)/NX
        b_trac[dx] += t_if[k, 0] * h_elem / 2
        b_trac[dy] += t_if[k, 1] * h_elem / 2
    
    b_vec = f_vol.vec.CreateVector()
    b_vec.data = f_vol.vec + b_trac
else:
    b_vec = f_vol.vec

with TaskManager():
    inv_a = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')
    gfu.vec.data = inv_a * b_vec

# Traction recovery
rvec = f_vol.vec.CreateVector()
rvec.data = a.mat * gfu.vec - f_vol.vec

fw = LinearForm(fes)
fw += InnerProduct(CF((1.0, 1.0)), v) * ds("interface")
fw.Assemble()

Q = np.zeros((len(iface_v), 2))
for k, (dx, dy) in enumerate(iface_dofs):
    wx, wy = fw.vec[dx], fw.vec[dy]
    rx, ry = rvec[dx], rvec[dy]
    Q[k, 0] = -rx/wx if abs(wx) > 1e-14 else 0
    Q[k, 1] = -ry/wy if abs(wy) > 1e-14 else 0

Path("exports.json").write_text(json.dumps({
    "field_name": "displacement",
    "n_points": len(iface_v),
    "coordinates": [[float(verts[i].point[0]), float(verts[i].point[1])] for i in iface_v],
    "values": [[float(gfu.vec[iface_dofs[k][0]]), float(gfu.vec[iface_dofs[k][1]])] for k in range(len(iface_v))],
    "normal_fluxes": [[float(Q[k, 0]), float(Q[k, 1])] for k in range(len(iface_v))]
}))

with open("run_log.txt", "w") as f:
    f.write(f"NDOF = {fes.ndof}\nElements = {mesh.ne}\nVertices = {mesh.nv}\n")

print(f"Subdomain A done: NDOF={fes.ndof}, iface_pts={len(iface_v)}")
