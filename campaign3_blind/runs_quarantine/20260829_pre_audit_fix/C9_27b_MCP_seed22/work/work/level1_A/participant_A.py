"""NGSolve participant for subdomain A (Neumann side) - coupled elasticity."""
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import *

# Problem parameters
SIDE = "neumann"  # A is Neumann side
PARTNER = "B"
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.0, 0.625
IFACE_Y = 0.625

# Material: lambda=480, mu=1200
LAMBDA = 480.0
MU = 1200.0

LEVEL = 1
if LEVEL == 1:
    NX, NY = 8, 5
elif LEVEL == 2:
    NX, NY = 16, 10
else:
    NX, NY = 32, 20

UI_X, UI_Y = 0.0, 0.0
TI_X, TI_Y = 0.0, 0.0
TOL = 1e-9 * max(X1 - X0, Y1 - Y0)

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None

def sample(imp, key, fallback, x):
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
    return np.column_stack([np.interp(x, xs[o], vs[o, c])
                            for c in range(vs.shape[1])])

imp = read_imports()

# Build mesh
geo = SplineGeometry()
p0 = geo.AddPoint(X0, Y0)
p1 = geo.AddPoint(X1, Y0)
p2 = geo.AddPoint(X1, Y1)
p3 = geo.AddPoint(X0, Y1)
geo.Append(["line", p0, p1], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p1, p2], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p2, p3], leftdomain=1, rightdomain=0, bc="interface")
geo.Append(["line", p3, p0], leftdomain=1, rightdomain=0, bc="left")

MAXH = min((X1 - X0) / NX, (Y1 - Y0) / NY)
mesh = Mesh(geo.GenerateMesh(maxh=MAXH))

fes = VectorH1(mesh, order=1, dirichlet="bottom|left|right")
u, v = fes.TnT()

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def Stress(u):
    return 2 * MU * Strain(u) + LAMBDA * Trace(Strain(u)) * Id(2)

a = BilinearForm(InnerProduct(Stress(u), Strain(v)) * dx)
a.Assemble()

# Body force using specialcf for position-dependent expressions
x_cf = specialcf("x")
y_cf = specialcf("y")

fx = (-63*x_cf**4*y_cf/3125 + 99*x_cf**4/5000 + 168*x_cf**3*y_cf/15625 - 33*x_cf**3/3125 
      + 216*x_cf**2*y_cf**2/625 - 1557*x_cf**2*y_cf/3125 - 99*x_cf**2/5000 
      - 288*x_cf*y_cf**2/3125 + 1992*x_cf*y_cf/15625 + 33*x_cf/3125 
      - 36*y_cf**2/625 + 54*y_cf/625)
fy = (-108*x_cf**5/15625 + 72*x_cf**4/15625 - 18*x_cf**3*y_cf**2/625 + 153*x_cf**3*y_cf/1250 
      - 279*x_cf**3/3125 + 36*x_cf**2*y_cf**2/3125 - 153*x_cf**2*y_cf/3125 + 486*x_cf**2/15625 
      + 9*x_cf*y_cf**2/625 - 153*x_cf*y_cf/2500 + 63*x_cf/1250 
      - 12*y_cf**2/3125 + 51*y_cf/3125 - 42/3125)

f_vol = LinearForm(fes)
f_vol += fx * v[0] * dx + fy * v[1] * dx
f_vol.Assemble()

all_iface_v = [v for v in mesh.vertices if abs(v.point[1] - IFACE_Y) < TOL]
all_iface_v.sort(key=lambda v: v.point[0])
x_if = np.array([v.point[0] for v in all_iface_v])

gfu = GridFunction(fes)
gfun = GridFunction(fes)
gfun.vec[:] = 0.0

if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), x_if)
    for k, vtx in enumerate(all_iface_v):
        if abs(vtx.point[0] - X0) < TOL or abs(vtx.point[0] - X1) < TOL:
            continue
        vd = fes.GetDofNrs(NodeId(VERTEX, vtx.nr))
        gfu.vec[int(vd[0])] = float(u_if[k, 0])
        gfu.vec[int(vd[1])] = float(u_if[k, 1])
else:
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), x_if)
    for k, vtx in enumerate(all_iface_v):
        vd = fes.GetDofNrs(NodeId(VERTEX, vtx.nr))
        gfun.vec[int(vd[0])] = float(t_if[k, 0])
        gfun.vec[int(vd[1])] = float(t_if[k, 1])
    
    f = LinearForm(fes)
    f += InnerProduct(gfun, v) * ds("interface")
    f += fx * v[0] * dx + fy * v[1] * dx
    f.Assemble()

with TaskManager():
    if SIDE == "neumann":
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec
    else:
        gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f_vol.vec

u_export = []
for v in all_iface_v:
    vd = fes.GetDofNrs(NodeId(VERTEX, v.nr))
    ux = float(gfu.vec[int(vd[0])])
    uy = float(gfu.vec[int(vd[1])])
    u_export.append([ux, uy])

if SIDE == "neumann":
    rvec = f.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec
else:
    rvec = f_vol.vec.CreateVector()
    rvec.data = a.mat * gfu.vec - f_vol.vec

fw_u, fw_v = fes.TnT()
fw = LinearForm(fes)
fw += 1.0 * fw_v[0] * ds("interface") + 1.0 * fw_v[1] * ds("interface")
fw.Assemble()

Q = np.zeros((len(all_iface_v), 2))
ok = np.ones((len(all_iface_v), 2), bool)
for k, vtx in enumerate(all_iface_v):
    vd = fes.GetDofNrs(NodeId(VERTEX, vtx.nr))
    for c in (0, 1):
        d = int(vd[c])
        wi = float(fw.vec[d])
        if abs(wi) > 1e-14:
            Q[k, c] = -float(rvec[d]) / wi
        else:
            ok[k, c] = False

suspect = np.zeros(len(all_iface_v), bool)
for k, vtx in enumerate(all_iface_v):
    if abs(vtx.point[0] - X0) < TOL or abs(vtx.point[0] - X1) < TOL:
        suspect[k] = True
    elif not ok[k].all():
        suspect[k] = True

good = np.where(~suspect)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

exports = {
    "field_name": "displacement",
    "n_points": len(all_iface_v),
    "coordinates": [[float(v.point[0]), float(v.point[1])] for v in all_iface_v],
    "values": u_export,
    "normal_fluxes": [[float(Q[k, 0]), float(Q[k, 1])] for k in range(len(all_iface_v))]
}
Path("exports.json").write_text(json.dumps(exports, indent=2))

ndof = fes.ndof
with open(f"run_level{LEVEL}_A.log", "w") as f:
    f.write(f"NDOF = {ndof}\n")

print(f"Subdomain A (Neumann): NDOF={ndof}, interface points={len(all_iface_v)}")
