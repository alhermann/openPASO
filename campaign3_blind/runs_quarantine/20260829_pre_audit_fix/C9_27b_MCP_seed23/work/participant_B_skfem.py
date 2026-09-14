"""scikit-fem participant for subdomain B (DIRICHLET side) - coupled elasticity.

Subdomain B: (0, 1) x (0.625, 1.5), lambda=480, mu=240
Interface at y = 0.625 (bottom edge of B)
Outer boundary: u=0 on all outer edges (top, left, right)
Interface: receives displacement from A, exports traction

SIGN CONVENTION: export traction as -(sigma . n_own) where n_own is outward normal.
For subdomain B, n_own at interface = (0, -1) (pointing down, out of B).
"""
import json
from pathlib import Path
import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP1, ElementVector,
                   FacetBasis, LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace

# Problem parameters
SIDE = "dirichlet"  # B is Dirichlet side
PARTNER = "A"       # partner is subdomain A
X0, X1 = 0.0, 1.0
Y0, Y1 = 0.625, 1.5
IFACE_Y = 0.625     # interface at bottom of B

# Material: lambda=480, mu=240 (plane strain)
LAMBDA = 480.0
MU = 240.0

# Source term for subdomain B
def F_SRC(x, y):
    """Body force for subdomain B."""
    fx = (1737*x**4*y/30625 - 9357*x**4/245000 - 4632*x**3*y/153125 + 3119*x**3/153125 
          + 396*x**2*y**2/875 - 57969*x**2*y/61250 + 86847*x**2/245000 
          - 528*x*y**2/4375 + 40962*x*y/153125 - 16034*x/153125 
          - 66*y**2/875 + 519*y/3500 - 369/7000)
    fy = (2316*x**5/153125 - 1544*x**4/153125 + 1158*x**3*y**2/30625 + 9201*x**3*y/61250 
          - 53561*x**3/245000 - 2316*x**2*y**2/153125 - 9201*x**2*y/153125 + 59737*x**2/612500 
          - 579*x*y**2/30625 - 9201*x*y/122500 + 9477*x/98000 
          + 772*y**2/153125 + 3067*y/153125 - 3159/122500)
    return fx, fy

# Mesh resolution
import os
LEVEL = int(os.environ.get('LEVEL', 1))
if LEVEL == 1:
    NX, NY = 8, 7  # 0.875 / (1/8) = 7
elif LEVEL == 2:
    NX, NY = 16, 14
else:  # LEVEL == 3
    NX, NY = 32, 28

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

def sample(imp, key, fallback, x):
    """Interpolate partner's vector samples onto this participant's x-coordinates."""
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

# Read imports
imp = read_imports()

# Build mesh - use tensor mesh for structured grid
x = np.linspace(X0, X1, NX + 1)
y = np.linspace(Y0, Y1, NY + 1)
mesh = MeshTri.init_tensor(x, y)

# Tag boundaries
mesh = mesh.with_boundaries({
    "bottom": lambda x: np.isclose(x[1], Y0, atol=1e-10),
    "top": lambda x: np.isclose(x[1], Y1, atol=1e-10),
    "left": lambda x: np.isclose(x[0], X0, atol=1e-10),
    "right": lambda x: np.isclose(x[0], X1, atol=1e-10),
})

# Vector FE space
basis = Basis(mesh, ElementVector(ElementTriP1()))

# Bilinear form for linear elasticity
@BilinearForm
def elasticity(u, v, w):
    """Plane strain elasticity bilinear form."""
    lam = w["lam"]
    mu = w["mu"]
    eps_u = sym_grad(u)
    eps_v = sym_grad(v)
    return 2*mu*ddot(eps_u, eps_v) + lam*trace(eps_u)*trace(eps_v)

K = asm(elasticity, basis, lam=LAMBDA, mu=MU)

# Volume load
@LinearForm
def body_force(v, w):
    """Body force linear form."""
    x_coord = w['x'][0]
    y_coord = w['x'][1]
    fx, fy = F_SRC(x_coord, y_coord)
    return fx * v[0] + fy * v[1]

b_vol = asm(body_force, basis)

# Get interface DOFs (on y=0.625) using basis.get_dofs
bottom_dofs = set(basis.get_dofs("bottom").flatten())

# Find which nodes are on the bottom interface
nodal_dofs = basis.nodal_dofs  # shape (2, n_nodes)
n_nodes = nodal_dofs.shape[1]
iface_nodes_x = [i for i in range(n_nodes) if nodal_dofs[0, i] in bottom_dofs]
iface_nodes_y = [i for i in range(n_nodes) if nodal_dofs[1, i] in bottom_dofs]
iface_nodes = sorted(set(iface_nodes_x) & set(iface_nodes_y))  # Nodes with both components on bottom
iface_nodes = np.array(iface_nodes)

# Get x-coordinates of interface nodes
iface_x = mesh.p[0, iface_nodes]

# Create mapping from node index to DOF indices
iface_dofs = np.column_stack([nodal_dofs[0, iface_nodes], nodal_dofs[1, iface_nodes]])

# Get outer boundary nodes (for corner exclusion)
outer_dofs = set(np.concatenate([
    basis.get_dofs("top").flatten(),
    basis.get_dofs("left").flatten(),
    basis.get_dofs("right").flatten()
]))
outer_nodes = np.array([i for i in range(n_nodes) if nodal_dofs[0, i] in outer_dofs or nodal_dofs[1, i] in outer_dofs])

# Get Dirichlet DOFs (outer boundary: top, left, right)
D_outer = np.concatenate([
    basis.get_dofs("top").flatten(),
    basis.get_dofs("left").flatten(),
    basis.get_dofs("right").flatten()
])
D_outer = np.unique(D_outer)

# Apply interface displacement (Dirichlet side)
sol = basis.zeros()
if SIDE == "dirichlet":
    u_if = sample(imp, "values", (UI_X, UI_Y), iface_x)
    
    # Map interface nodes to their position in iface_nodes array
    node_to_idx = {node: i for i, node in enumerate(iface_nodes)}
    
    # Apply interface displacement (excluding corners)
    for k, node in enumerate(iface_nodes):
        dof_x = nodal_dofs[0, node]
        dof_y = nodal_dofs[1, node]
        # Check if this is a corner
        x_coord = mesh.p[0, node]
        is_corner = (abs(x_coord) < 1e-10) or (abs(x_coord - 1.0) < 1e-10)
        if not is_corner:
            sol[dof_x] = u_if[k, 0]
            sol[dof_y] = u_if[k, 1]
    
    # Combine Dirichlet DOFs
    D = np.concatenate([D_outer, iface_dofs.flatten()])
    D = np.unique(D)
else:
    D = D_outer

# Condense and solve
K_I, b_I, _, _ = condense(K, b_vol, D=D, x=sol if SIDE == "dirichlet" else None)
sol_I = solve(K_I, b_I)

# Full solution
sol_full = np.zeros(basis.N)
free_dofs = np.setdiff1d(np.arange(basis.N), D)
sol_full[free_dofs] = sol_I
sol_full[D] = sol[D]
sol = sol_full

# Facet basis for traction computation
fbasis = FacetBasis(mesh, ElementVector(ElementTriP1()))

# Compute interface traction export: q_out = -(sigma . n_own)
# For subdomain B, n_own = (0, -1) at interface (pointing down)
# Use consistent (reaction) flux method

@LinearForm
def unit_load(v, w):
    """w_i = int_Gamma phi_i ds."""
    return 1.0 * v[0] + 1.0 * v[1]

# Residual
r = K @ sol - b_vol

# Weight
wgt = asm(unit_load, fbasis)

# Get interface facet DOFs
bottom_facet_dofs = set(fbasis.get_dofs("bottom").flatten())
fb_nodal_dofs = fbasis.nodal_dofs  # shape (2, n_nodes)
fb_n_nodes = fb_nodal_dofs.shape[1]
iface_facet_nodes = [i for i in range(fb_n_nodes) if fb_nodal_dofs[0, i] in bottom_facet_dofs and fb_nodal_dofs[1, i] in bottom_facet_dofs]
iface_facet_nodes = np.array(iface_facet_nodes)
idx = np.column_stack([fb_nodal_dofs[0, iface_facet_nodes], fb_nodal_dofs[1, iface_facet_nodes]])

wi = wgt[idx]
Q = np.zeros_like(wi)
ok = np.abs(wi) > 1e-14
Q[ok] = -r[idx][ok] / wi[ok]

# Handle corners
suspect = np.isin(iface_nodes, outer_nodes) | ~ok.all(axis=1)
good = np.where(~suspect)[0]
if len(good) > 0:
    for i in np.where(suspect)[0]:
        Q[i] = Q[good[np.argmin(np.abs(good - i))]]

# Export data
exports = {
    "field_name": "displacement",
    "n_points": int(len(iface_nodes)),
    "coordinates": [[float(iface_x[k]), float(IFACE_Y)] for k in range(len(iface_nodes))],
    "values": [[float(sol[iface_dofs[k, 0]]), float(sol[iface_dofs[k, 1]])] 
               for k in range(len(iface_nodes))],
    "normal_fluxes": [[float(Q[k, 0]), float(Q[k, 1])] for k in range(len(iface_nodes))]
}

Path("exports.json").write_text(json.dumps(exports, indent=2))

# Write run log
ndof = basis.N
with open("run.log", "w") as log:
    log.write(f"NDOF = {ndof}\n")
    log.write(f"Level = {LEVEL}\n")
    log.write(f"Elements = {mesh.t.shape[1]}\n")
    log.write(f"Interface nodes = {len(iface_nodes)}\n")

print(f"Subdomain B (Dirichlet): NDOF={ndof}, converged")
