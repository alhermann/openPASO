"""scikit-fem participant for subdomain B (Dirichlet side) - coupled elasticity.

Subdomain B: (0,1) x (0.625, 1.5), lambda=480, mu=240
Interface at y = 0.625 (bottom edge of this subdomain)
Role: DIRICHLET - imports displacement from partner, exports traction
"""
import json
from pathlib import Path
import numpy as np
from skfem import *
from skfem.helpers import ddot, sym_grad, trace
from skfem.models.elasticity import linear_elasticity

# ── PROBLEM PARAMETERS ───────────────────────────────────────────────────────
SIDE      = "dirichlet"   # Subdomain B is DIRICHLET side
PARTNER   = "A"           # Partner name (subdomain A)
X0, X1    = 0.0, 1.0      # Subdomain B x-range
Y0, Y1    = 0.625, 1.5    # Subdomain B y-range
IFACE_Y   = 0.625         # Interface at bottom (y = 5/8)
LAM       = 480           # Lame parameter lambda
MU        = 240           # Shear modulus mu

# Source term for subdomain B (polynomial body force)
def F_SRC(x, y):
    """Body force f_x, f_y for subdomain B."""
    fx = (1737*x**4*y/30625 - 9357*x**4/245000 - 4632*x**3*y/153125 
          + 3119*x**3/153125 + 396*x**2*y**2/875 - 57969*x**2*y/61250 
          + 86847*x**2/245000 - 528*x*y**2/4375 + 40962*x*y/153125 
          - 16034*x/153125 - 66*y**2/875 + 519*y/3500 - 369/7000)
    fy = (2316*x**5/153125 - 1544*x**4/153125 + 1158*x**3*y**2/30625 
          + 9201*x**3*y/61250 - 53561*x**3/245000 - 2316*x**2*y**2/153125 
          - 9201*x**2*y/153125 + 59737*x**2/612500 - 579*x*y**2/30625 
          - 9201*x*y/122500 + 9477*x/98000 + 772*y**2/153125 
          + 3067*y/153125 - 3159/122500)
    return fx, fy

# Iteration-1 fallback values
UI_X, UI_Y = 0.0, 0.0     # Fallback interface displacement
TI_X, TI_Y = 0.0, 0.0     # Fallback interface traction
# ────────────────────────────────────────────────────────────────────────────

def read_imports():
    """Read imports.json and return partner data."""
    p = Path("imports.json")
    if not p.is_file():
        return None
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    return d.get(PARTNER) or None

def sample_vector(imp, key, fallback, x):
    """Interpolate partner's vector samples onto this participant's x-coordinates.
    
    Returns (len(x), 2) array with components interpolated separately.
    """
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

# Read mesh level from environment or use default
import os
mesh_level = int(os.environ.get('MESH_LEVEL', '1'))
if mesh_level == 1:
    NX, NY = 8, 7  # h ≈ 1/8 (0.875/7 ≈ 0.125)
elif mesh_level == 2:
    NX, NY = 16, 14  # h ≈ 1/16
else:
    NX, NY = 32, 28  # h ≈ 1/32

print(f"Subdomain B (skfem): NX={NX}, NY={NY}, level={mesh_level}")

# Build mesh using tensor grid
x = np.linspace(X0, X1, NX + 1)
y = np.linspace(Y0, Y1, NY + 1)
m = MeshQuad.init_tensor(x, y)

# Tag boundaries
_tol = 1e-10
m = m.with_boundaries({
    "bottom": lambda x: np.abs(x[1] - Y0) < _tol,  # Interface
    "top": lambda x: np.abs(x[1] - Y1) < _tol,
    "left": lambda x: np.abs(x[0] - X0) < _tol,
    "right": lambda x: np.abs(x[0] - X1) < _tol,
})

# Vector FE space - P1 elements on quads
e = ElementVector(ElementQuad1())
basis = Basis(m, e)

# Bilinear form using built-in elasticity model
K = linear_elasticity(LAM, MU).assemble(basis)

# Volume load (body force)
@LinearForm
def volume_load(v, w):
    x = w['x']
    y = w['y']
    fx, fy = F_SRC(x, y)
    return fx * v[0] + fy * v[1]

b_vol = volume_load.assemble(basis)

# Identify boundary DOFs
# Get nodal DOF mapping: ndof[comp, node_idx] -> global_dof
nd = basis.nodal_dofs  # shape (2, n_nodes)

# Find interface nodes (on y = IFACE_Y, but NOT corners)
interface_nodes = []
outer_nodes = []

for i, (xi, yi) in enumerate(m.p.T):
    on_interface = abs(yi - IFACE_Y) < _tol
    on_top = abs(yi - Y1) < _tol
    on_left = abs(xi - X0) < _tol
    on_right = abs(xi - X1) < _tol
    
    # Corner points belong to outer boundary, not interface
    is_corner = (on_interface and (on_left or on_right))
    
    if on_interface and not is_corner:
        interface_nodes.append(i)
    elif on_top or on_left or on_right:
        outer_nodes.append(i)

# Sort interface nodes by x-coordinate
interface_nodes.sort()
iface_x = m.p[0, interface_nodes]

# Build DOF lists
outer_dofs = set()
for i in outer_nodes:
    outer_dofs.add(nd[0, i])
    outer_dofs.add(nd[1, i])
outer_dofs = np.array(sorted(outer_dofs))

iface_dofs_sorted = [(nd[0, i], nd[1, i]) for i in interface_nodes]

# Read imports
imp = read_imports()

# Apply boundary conditions and solve
sol = basis.zeros()

if SIDE == "dirichlet":
    # Import displacement from partner and apply as Dirichlet BC
    u_if = sample_vector(imp, "values", (UI_X, UI_Y), iface_x)
    
    # Set interface DOFs
    for k, i in enumerate(interface_nodes):
        sol[nd[0, i]] = float(u_if[k, 0])
        sol[nd[1, i]] = float(u_if[k, 1])
    
    # All Dirichlet DOFs: outer + interface
    D = np.concatenate([outer_dofs, [nd[c, i] for i in interface_nodes for c in [0, 1]]])
    b = b_vol.copy()
else:
    # Neumann side would import traction here
    D = outer_dofs
    b = b_vol.copy()

# Condense and solve
A_c, b_c, _ = condense(K, b, D=D)
sol_free = solve(A_c, b_c)
sol[D] = 0.0  # Reset Dirichlet DOFs
idx = 0
for dof in sorted(set(range(K.shape[0])) - set(D)):
    sol[dof] = sol_free[idx]
    idx += 1

# Compute interface traction export: q_out = -(sigma . n_own)
# For subdomain B, outward normal at interface (bottom) is n = (0, -1)
# Use consistent (reaction) flux recovery: r = K*u - b_vol

r = K @ sol - b_vol

# Weight form for nodal averaging
@LinearForm
def unit_load(v, w):
    return 1.0 * v[0] + 1.0 * v[1]

# Assemble weights on interface facets
fbasis = FacetBasis(m, ElementQuad1(), facets=m.facets_satisfying(lambda x: np.abs(x[1] - IFACE_Y) < _tol))
wgt = unit_load.assemble(FacetBasis(m, e, facets=fbasis.facets))

Q = np.zeros((len(interface_nodes), 2))
for k, i in enumerate(interface_nodes):
    dx, dy = nd[0, i], nd[1, i]
    wx, wy = wgt[dx], wgt[dy]
    rx, ry = r[dx], r[dy]
    if abs(wx) > 1e-14:
        Q[k, 0] = -rx / wx
    if abs(wy) > 1e-14:
        Q[k, 1] = -ry / wy

# Export interface data
export_data = {
    "field_name": "displacement",
    "n_points": len(interface_nodes),
    "coordinates": [[float(m.p[0, i]), float(m.p[1, i])] for i in interface_nodes],
    "values": [[float(sol[nd[0, i]]), float(sol[nd[1, i]])] for i in interface_nodes],
    "normal_fluxes": [[float(Q[k, 0]), float(Q[k, 1])] for k in range(len(interface_nodes))]
}

Path("exports.json").write_text(json.dumps(export_data, indent=2))

# Write run log
ndof = K.shape[0]
with open("run_log.txt", "w") as f:
    f.write(f"NDOF = {ndof}\n")
    f.write(f"Elements = {m.ne}\n")
    f.write(f"Vertices = {m.nv}\n")

print(f"Subdomain B complete: NDOF={ndof}, interface points={len(interface_nodes)}")
