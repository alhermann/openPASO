"""NGSolve participant for subdomain A (NEUMANN side) - coupled elasticity.

Subdomain A: (0, 1) x (0, 0.625), lambda=480, mu=1200
Interface at y = 0.625 (top edge of A)
Outer boundary: u=0 on all outer edges (bottom, left, right)
Interface: receives traction from B, exports displacement

SIGN CONVENTION: export traction as -(sigma . n_own) where n_own is outward normal.
For subdomain A, n_own at interface = (0, +1) (pointing up, out of A).
Traction = sigma . n = [sigma_xx, sigma_xy; sigma_yx, sigma_yy] . [0, 1] = [sigma_xy, sigma_yy]
Export q_out = -traction = [-sigma_xy, -sigma_yy]
"""
import json
from pathlib import Path
import numpy as np
from netgen.geom2d import SplineGeometry
from ngsolve import (BilinearForm, CF, GridFunction, InnerProduct,
                     LinearForm, Mesh, TaskManager, VectorH1, ds, dx,
                     grad, Sym, Trace, Id, CoefficientFunction, x, y,
                     Project, L2, MatrixValued)

# Problem parameters
SIDE = "neumann"  # A is Neumann side
PARTNER = "B"     # partner is subdomain B
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.0, 0.625
IFACE_Y = 0.625   # interface at top of A

# Material: lambda=480, mu=1200 (plane strain)
LAMBDA = 480.0
MU = 1200.0

# Source term for subdomain A - defined using NGSolve x and y
F_SRC_x = (-63*x**4*y/3125 + 99*x**4/5000 + 168*x**3*y/15625 - 33*x**3/3125 
           + 216*x**2*y**2/625 - 1557*x**2*y/3125 - 99*x**2/5000 
           - 288*x*y**2/3125 + 1992*x*y/15625 + 33*x/3125 
           - 36*y**2/625 + 54*y/625)

F_SRC_y = (-108*x**5/15625 + 72*x**4/15625 - 18*x**3*y**2/625 + 153*x**3*y/1250 
           - 279*x**3/3125 + 36*x**2*y**2/3125 - 153*x**2*y/3125 + 486*x**2/15625 
           + 9*x*y**2/625 - 153*x*y/2500 + 63*x/1250 
           - 12*y**2/3125 + 51*y/3125 - 42/3125)

# Mesh resolution (passed via environment)
import os
LEVEL = int(os.environ.get('LEVEL', 1))
if LEVEL == 1:
    NX, NY = 8, 5  # 0.625 / (1/8) = 5
elif LEVEL == 2:
    NX, NY = 16, 10
else:  # LEVEL == 3
    NX, NY = 32, 20

# Fallback values for iteration 1
UI_X, UI_Y = 0.0, 0.0  # fallback displacement
TI_X, TI_Y = 0.0, 0.0  # fallback traction

def read_imports():
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None

def sample(imp, key, fallback, x_arr):
    """Interpolate partner's vector samples onto this participant's x-coordinates."""
    fb = np.asarray(fallback, float).ravel()
    if not imp or not imp.get("coordinates"):
        return np.tile(fb, (len(x_arr), 1))
    xs = np.array([c[0] for c in imp["coordinates"]], float)
    vs = np.asarray(imp.get(key) or [], float)
    if vs.ndim == 1:
        vs = vs.reshape(-1, 1)
    if vs.shape[0] != xs.size or vs.shape[1] != fb.size:
        return np.tile(fb, (len(x_arr), 1))
    o = np.argsort(xs)
    return np.column_stack([np.interp(x_arr, xs[o], vs[o, c])
                            for c in range(vs.shape[1])])

# Read imports
imp = read_imports()

# Build mesh
geo = SplineGeometry()
p0 = geo.AddPoint(X0, Y0)
p1 = geo.AddPoint(X1, Y0)
p2 = geo.AddPoint(X1, Y1)
p3 = geo.AddPoint(X0, Y1)

# Create edges with boundary names
geo.Append([p0, p1], leftdomain=1, bc="bottom")   # y=0
geo.Append([p1, p2], leftdomain=1, bc="right")    # x=1
geo.Append([p2, p3], leftdomain=1, bc="interface") # y=0.625 (interface)
geo.Append([p3, p0], leftdomain=1, bc="left")     # x=0

maxh = min((X1-X0)/NX, (Y1-Y0)/NY)
mesh = Mesh(geo.GenerateMesh(maxh=maxh))

# Vector FE space - Dirichlet on outer boundary (bottom, left, right)
fes = VectorH1(mesh, order=1, dirichlet="bottom|left|right")
u, v = fes.TnT()

# Strain and stress
def Strain(u):
    return Sym(grad(u))

def Stress(u):
    return 2*MU*Strain(u) + LAMBDA*Trace(Strain(u))*Id(2)

# Bilinear form
a = BilinearForm(fes)
a += InnerProduct(Stress(u), Strain(v)) * dx
a.Assemble()

# Volume load (source term) - use tuple of expressions
f_vol = LinearForm(fes)
f_vol += InnerProduct(CF((F_SRC_x, F_SRC_y)), v) * dx
f_vol.Assemble()

# Full load (will add interface traction for Neumann side)
f = LinearForm(fes)
f += InnerProduct(CF((F_SRC_x, F_SRC_y)), v) * dx
f.Assemble()

# Get interface nodes (on y=0.625, excluding corners)
iface_vertices = []
iface_x = []
for i, vertex in enumerate(mesh.vertices):
    yp = vertex.point[1]
    xp = vertex.point[0]
    if abs(yp - IFACE_Y) < 1e-10:
        iface_vertices.append(i)
        iface_x.append(xp)
iface_vertices = np.array(iface_vertices)
iface_x = np.array(iface_x)

# Get outer boundary vertices (for corner exclusion)
outer_vertices = []
for i, vertex in enumerate(mesh.vertices):
    xp, yp = vertex.point
    is_outer = (abs(yp) < 1e-10) or (abs(xp) < 1e-10) or (abs(xp - 1.0) < 1e-10)
    if is_outer and abs(yp - IFACE_Y) < 1e-10:
        outer_vertices.append(i)
outer_vertices = np.array(outer_vertices)

# Apply interface data
if SIDE == "neumann":
    # Import traction and apply as Neumann BC
    t_if = sample(imp, "normal_fluxes", (TI_X, TI_Y), iface_x)
    gfun = GridFunction(fes)
    gfun.vec[:] = 0.0
    for k, vtx in enumerate(iface_vertices):
        gfun.vec[2*vtx] = float(t_if[k, 0])
        gfun.vec[2*vtx+1] = float(t_if[k, 1])
    # Apply traction: f += InnerProduct(gfun, v) * ds("interface")
    f += InnerProduct(gfun, v) * ds("interface")
    f.Assemble()

# Solve
gfu = GridFunction(fes)
with TaskManager():
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs(), inverse="sparsecholesky") * f.vec

# Compute interface traction by projecting stress to a discontinuous space
# and evaluating at interface nodes
fes_stress = MatrixValued(H1(mesh, order=0), symmetric=True)
stress_gfu = GridFunction(fes_stress)

# Define stress as a function of the solution
def stress_func(x):
    eps = Sym(grad(gfu))(x)
    sig = 2*MU*eps + LAMBDA*Trace(eps)*Id(2)
    return sig

# Project stress
stress_proj = Project(stress_func, fes_stress)

# Extract traction at interface nodes
# For subdomain A, n = (0, 1), so traction = [sigma_xy, sigma_yy]
# Export q_out = -traction = [-sigma_xy, -sigma_yy]
Q = np.zeros((len(iface_vertices), 2))
for k, vtx in enumerate(iface_vertices):
    vertex = mesh.vertices[vtx]
    xv, yv = vertex.point
    
    # Evaluate projected stress at this point
    sig = stress_proj(mesh(xv, yv))
    
    # sig is a 2x2 matrix: [[sig_xx, sig_xy], [sig_xy, sig_yy]]
    sig_xy = sig[0, 1]
    sig_yy = sig[1, 1]
    
    # Traction = sigma . n where n = (0, 1)
    tx = sig_xy
    ty = sig_yy
    
    # Export q_out = -traction
    Q[k, 0] = -tx
    Q[k, 1] = -ty

# Handle corners: use nearest interior node
suspect = np.isin(iface_vertices, outer_vertices)
good = np.where(~suspect)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Export data
exports = {
    "field_name": "displacement",
    "n_points": int(len(iface_vertices)),
    "coordinates": [[float(iface_x[k]), float(IFACE_Y)] for k in range(len(iface_vertices))],
    "values": [[float(gfu.vec[2*iface_vertices[k]]), float(gfu.vec[2*iface_vertices[k]+1])] 
               for k in range(len(iface_vertices))],
    "normal_fluxes": [[float(Q[k, 0]), float(Q[k, 1])] for k in range(len(iface_vertices))]
}

Path("exports.json").write_text(json.dumps(exports, indent=2))

# Write run log
ndof = fes.ndof
with open("run.log", "w") as log:
    log.write(f"NDOF = {ndof}\n")
    log.write(f"Level = {LEVEL}\n")
    log.write(f"Elements = {mesh.ne}\n")
    log.write(f"Interface nodes = {len(iface_vertices)}\n")

print(f"Subdomain A (Neumann): NDOF={ndof}, converged")
print(f"Traction sample: {Q[:3]}")
