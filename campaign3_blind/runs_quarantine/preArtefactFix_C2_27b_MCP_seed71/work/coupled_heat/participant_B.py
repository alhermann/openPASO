#!/usr/bin/env python3
"""
Participant B (Kratos) - Subdomain B: (0.625, 1.5) x (0, 1), k=200
Neumann side: receives flux from A, applies as natural BC at x=0.625
Returns: interface field values at x=0.625
"""
import json
import os
import sys
import numpy as np

# Read level from file
work_dir = os.getcwd()
level_file = os.path.join(work_dir, "level.json")
if os.path.exists(level_file):
    with open(level_file) as f:
        level_info = json.load(f)
    LEVEL = level_info.get("level", 1)
else:
    LEVEL = int(os.environ.get("LEVEL", 1))

# Mesh resolution
h_values = {1: 1/8, 2: 1/16, 3: 1/32}
h = h_values[LEVEL]

# Subdomain B geometry
xB_min, xB_max = 0.625, 1.5
yB_min, yB_max = 0.0, 1.0
k_B = 200.0

nx_B = int(round((xB_max - xB_min) / h))
ny_B = int(round((yB_max - yB_min) / h))

print(f"=== Participant B (Kratos) ===")
print(f"Level: {LEVEL}, h: {h}")
print(f"Mesh: {nx_B} x {ny_B} elements")

# Source term in subdomain B (Python syntax)
def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Generate mesh nodes and elements
nodes = []
node_map = {}
nid = 1

for j in range(ny_B + 1):
    for i in range(nx_B + 1):
        x = xB_min + i * (xB_max - xB_min) / nx_B
        y = yB_min + j * (yB_max - yB_min) / ny_B
        node_map[(i, j)] = nid
        nodes.append((nid, x, y))
        nid += 1

n_nodes = nid - 1

# Create triangular elements (P1)
elements = []
for j in range(ny_B):
    for i in range(nx_B):
        n_bl = node_map[(i, j)]
        n_br = node_map[(i+1, j)]
        n_tr = node_map[(i+1, j+1)]
        n_tl = node_map[(i, j+1)]
        # Split quad into two triangles
        elements.append([n_bl, n_br, n_tl])
        elements.append([n_br, n_tr, n_tl])

n_elements = len(elements)

# Identify boundary nodes
interface_x = 0.625
tol = 1e-6

# Interface nodes (x = 0.625, excluding corners)
interface_nodes = []
for j in range(1, ny_B):  # Exclude corners
    nid = node_map[(0, j)]
    interface_nodes.append(nid)

# Left boundary (interface) - where Neumann BC is applied
# Right boundary (x = 1.5) - Dirichlet u=0
# Bottom (y=0) - Dirichlet u=0  
# Top (y=1) - Dirichlet u=0

dirichlet_nodes = set()
# Right edge
for j in range(ny_B + 1):
    dirichlet_nodes.add(node_map[(nx_B, j)])
# Bottom edge
for i in range(nx_B + 1):
    dirichlet_nodes.add(node_map[(i, 0)])
# Top edge
for i in range(nx_B + 1):
    dirichlet_nodes.add(node_map[(i, ny_B)])

interior_nodes = [n for n in range(1, n_nodes + 1) if n not in dirichlet_nodes and n not in interface_nodes]

print(f"Nodes: {n_nodes}, Elements: {n_elements}")
print(f"Dirichlet nodes: {len(dirichlet_nodes)}, Interior: {len(interior_nodes)}, Interface: {len(interface_nodes)}")

# Read imports from partner A
imports_path = os.path.join(work_dir, "imports.json")
if os.path.exists(imports_path):
    with open(imports_path) as f:
        imports = json.load(f)
    has_imports = True
else:
    imports = {}
    has_imports = False

# Get flux from partner A
if has_imports and "A" in imports:
    interface_data = imports["A"]
    import_fluxes = np.array(interface_data["normal_fluxes"])
    print(f"Received flux from A: {len(import_fluxes)} points")
else:
    import_fluxes = np.zeros(len(interface_nodes))
    print("No imports available, using zero flux")

# Assemble global stiffness matrix and load vector
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

K = lil_matrix((n_nodes, n_nodes))
F = np.zeros(n_nodes)

# Element assembly for P1 triangles
for elem in elements:
    n1, n2, n3 = elem
    coords_elem = [(nodes[n-1][1], nodes[n-1][2]) for n in elem]
    x = np.array([c[0] for c in coords_elem])
    y = np.array([c[1] for c in coords_elem])
    
    # Triangle area
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    if area < 1e-15:
        continue
    
    # Gradient of shape functions
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    # Stiffness matrix (conductivity k_B)
    Ke = (k_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
    
    # Load vector (source term integrated over element)
    # Using 3-point quadrature for triangle
    centroid_x = np.mean(x)
    centroid_y = np.mean(y)
    f_val = source_B(centroid_x, centroid_y)
    Fe = f_val * area / 3.0
    
    # Assemble
    for a in range(3):
        for bb in range(3):
            K[elem[a]-1, elem[bb]-1] += Ke[a, bb]
        F[elem[a]-1] += Fe[a]

# Apply Neumann BC on interface (left edge of subdomain B)
# The flux from A is the outward normal flux from A, which points in +x direction
# For subdomain B, this is an INWARD flux, so we need to negate it for the weak form
# Natural BC: integral of q * v ds on boundary
# For linear elements, distribute flux to interface nodes

# Map imported fluxes to interface nodes
# The imported fluxes are at specific y-coordinates; interpolate to our nodes
if len(import_fluxes) > 0 and len(interface_nodes) > 0:
    # Get y-coordinates of interface nodes
    interface_y_coords = [nodes[n-1][2] for n in interface_nodes]
    
    # Simple assignment: if counts match, use directly; otherwise interpolate
    if len(import_fluxes) == len(interface_nodes):
        neumann_fluxes = import_fluxes
    else:
        # Interpolate imported fluxes to our interface nodes
        import_y = np.linspace(0, 1, len(import_fluxes))[1:-1] if len(import_fluxes) > 2 else np.array(interface_y_coords)
        neumann_fluxes = np.interp(interface_y_coords, import_y, import_fluxes)
    
    # Apply Neumann BC: add to RHS
    # Flux is defined as outward from A (+x direction), so for B it's inward
    # In weak form: -integral(k*du/dn*v) = integral(q*v) where q is prescribed flux
    # For heat equation -div(k grad u) = f, Neumann BC is k*du/dn = g
    # Here g = -flux_from_A (since flux_from_A is outward from A)
    for idx, nid in enumerate(interface_nodes):
        F[nid-1] -= neumann_fluxes[idx]  # Negative because it's inward flux

K = K.tocsr()

# Apply Dirichlet BCs (u=0)
# Remove rows/cols for Dirichlet nodes
free_dofs = [n-1 for n in interior_nodes]
fixed_dofs = [n-1 for n in dirichlet_nodes]

if len(free_dofs) > 0:
    K_ff = K[np.ix_(free_dofs, free_dofs)]
    F_f = F[free_dofs]
    
    # Solve
    u_free = spsolve(K_ff, F_f)
    
    # Reconstruct full solution
    u = np.zeros(n_nodes)
    u[free_dofs] = u_free
    # Dirichlet nodes have u=0
else:
    u = np.zeros(n_nodes)

print(f"Solution computed: min={u.min():.6f}, max={u.max():.6f}")

# Extract interface values
interface_u = np.array([u[n-1] for n in interface_nodes])
interface_coords = np.array([[nodes[n-1][1], nodes[n-1][2]] for n in interface_nodes])

# Compute outward normal flux for subdomain B
# Outward normal at interface is -x direction
# qn = -k * grad(u) . n = -k * (-du/dx) = k * du/dx

interface_fluxes = []
for idx, nid in enumerate(interface_nodes):
    y_coord = nodes[nid-1][2]
    
    # Find node to the right (in element interior)
    same_row = [n for n in range(1, n_nodes+1) if abs(nodes[n-1][2] - y_coord) < tol]
    row_x = [nodes[n-1][1] for n in same_row]
    row_u = [u[n-1] for n in same_row]
    
    # Find node just right of interface
    right_of_interface = [(x, u_val) for x, u_val in zip(row_x, row_u) if x > interface_x + tol]
    if right_of_interface:
        x_right = min(right_of_interface, key=lambda p: p[0])
        dx = x_right[0] - interface_x
        du_dx = (x_right[1] - u[nid-1]) / dx
    else:
        du_dx = 0.0
    
    # Outward normal flux for B: qn = k_B * du/dx (outward = -x)
    qn = k_B * du_dx
    interface_fluxes.append(qn)

interface_fluxes = np.array(interface_fluxes)
print(f"Interface fluxes (B): min={interface_fluxes.min():.6f}, max={interface_fluxes.max():.6f}")

# Write exports.json
exports = {
    "field_name": "temperature",
    "coordinates": interface_coords.tolist(),
    "values": interface_u.tolist(),
    "normal_fluxes": interface_fluxes.tolist(),
    "n_points": len(interface_u)
}

exports_path = os.path.join(work_dir, "exports.json")
with open(exports_path, 'w') as f:
    json.dump(exports, f, indent=2)

print(f"Wrote exports.json with {len(interface_u)} points")

# Save solution as VTU for post-processing
import meshio
pts = np.array([[nodes[n-1][1], nodes[n-1][2], 0.0] for n in range(1, n_nodes+1)])
cells = np.array([[e[0]-1, e[1]-1, e[2]-1] for e in elements])
mesh = meshio.Mesh(pts, [("triangle", cells)], point_data={"temperature": u})
mesh.save(os.path.join(work_dir, f"solution_B_level{LEVEL}.vtu"))

print(f"NDOF = {n_nodes}")
