#!/usr/bin/env python3
"""
Participant B (manual assembly): Subdomain (0.625, 1.5) x (0, 1), k=200
Neumann side: receives flux from partner A, applies as BC on interface, returns temperature
"""

import os
import json
import numpy as np
from pathlib import Path
import sys

# Problem parameters
INTERFACE_X = 5/8  # 0.625
DOMAIN_B_WIDTH = 1.5 - INTERFACE_X
DOMAIN_HEIGHT = 1.0
K_B = 200.0

# Read level info
work_dir = Path(os.getcwd())
level_info_file = work_dir / "level.json"
with open(level_info_file) as f:
    level_info = json.load(f)

n_divisions = level_info["n_divisions"]
nx = level_info["nx_B"]
ny = level_info["ny"]

print(f"Subdomain B: nx={nx}, ny={ny}, h=1/{n_divisions}")

# Generate mesh nodes for subdomain B
nodes = []
node_id = 1
for j in range(ny + 1):
    for i in range(nx + 1):
        x = INTERFACE_X + i * DOMAIN_B_WIDTH / nx
        y = j * DOMAIN_HEIGHT / ny
        nodes.append((node_id, x, y))
        node_id += 1

n_nodes = len(nodes)
print(f"Number of nodes: {n_nodes}")

# Elements (triangles from quad splitting)
elements = []
elem_id = 1
for j in range(ny):
    for i in range(nx):
        n1 = (j * (nx + 1) + i) + 1
        n2 = (j * (nx + 1) + i + 1) + 1
        n3 = ((j + 1) * (nx + 1) + i + 1) + 1
        n4 = ((j + 1) * (nx + 1) + i) + 1
        elements.append([n1, n2, n4])
        elements.append([n2, n3, n4])

n_elements = len(elements)
print(f"Number of elements: {n_elements}")

# Source term function
def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Build stiffness matrix and load vector using scipy
try:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for elem_conn in elements:
        ids = [nid - 1 for nid in elem_conn]  # Convert to 0-based
        coords = np.array([nodes[nid-1][1:] for nid in elem_conn])  # Get (x, y) only
        x = coords[:, 0]
        y = coords[:, 1]
        
        # Triangle area
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        
        # Shape function derivatives
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        
        # Element stiffness matrix
        Ke = (K_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        # Element load vector (using centroid integration)
        xc = np.mean(x)
        yc = np.mean(y)
        fc = source_B(xc, yc)
        Fe = fc * area / 3.0 * np.ones(3)
        
        # Assemble
        for a in range(3):
            for bb in range(3):
                K[ids[a], ids[bb]] += Ke[a, bb]
            F[ids[a]] += Fe[a]
    
    K = K.tocsr()
    print("Stiffness matrix assembled")
except Exception as e:
    print(f"Error assembling matrix: {e}")
    sys.exit(1)

# Identify boundary nodes
tol = 1e-10
interface_nodes = set()  # Left edge of domain B (x = INTERFACE_X)
right_nodes = set()     # Right edge (x = 1.5)
bottom_nodes = set()    # Bottom edge (y = 0)
top_nodes = set()       # Top edge (y = 1)

for nid, (node_id, x, y) in enumerate(nodes):
    if abs(x - INTERFACE_X) < tol:
        interface_nodes.add(nid)
    if abs(x - (INTERFACE_X + DOMAIN_B_WIDTH)) < tol:
        right_nodes.add(nid)
    if abs(y) < tol:
        bottom_nodes.add(nid)
    if abs(y - DOMAIN_HEIGHT) < tol:
        top_nodes.add(nid)

# Dirichlet nodes (outer boundary, excluding interface corners)
dirichlet_nodes = right_nodes | bottom_nodes | top_nodes
interface_corners = {nid for nid in interface_nodes if nid in (bottom_nodes | top_nodes)}
dirichlet_nodes -= interface_corners

print(f"Interface nodes: {len(interface_nodes)}, Dirichlet nodes: {len(dirichlet_nodes)}")

# Read imports from partner A (flux values)
imports_file = work_dir / "imports.json"
interface_fluxes_from_partner = []
interface_coords_from_partner = []

if imports_file.exists():
    with open(imports_file) as f:
        imports = json.load(f)
    if "A" in imports:
        partner_data = imports["A"]
        interface_fluxes_from_partner = partner_data.get("normal_fluxes", [])
        interface_coords_from_partner = partner_data.get("coordinates", [])
        print(f"Received {len(interface_fluxes_from_partner)} flux values from partner A")
    else:
        print("No data received from partner A")
else:
    print("No imports.json found (first iteration)")

# Apply Neumann BC on interface (if we have flux data)
if interface_fluxes_from_partner and len(interface_fluxes_from_partner) > 0:
    # Map partner fluxes to our interface nodes
    for idx, (px, py) in enumerate(interface_coords_from_partner):
        qn = interface_fluxes_from_partner[idx]
        # Find closest interface node
        min_dist = float('inf')
        closest_node = None
        for nid in interface_nodes:
            _, x, y = nodes[nid]
            dist = (x - px)**2 + (y - py)**2
            if dist < min_dist:
                min_dist = dist
                closest_node = nid
        if closest_node is not None:
            # Neumann BC: add flux to RHS
            F[closest_node] += qn * (DOMAIN_HEIGHT / ny)

# Solve system with Dirichlet BCs
u = np.zeros(n_nodes)
interior_nodes = list(set(range(n_nodes)) - dirichlet_nodes)  # Include interface nodes

print(f"Interior nodes: {len(interior_nodes)}")

if interior_nodes:
    K_int = K[np.ix_(interior_nodes, interior_nodes)]
    F_int = F[interior_nodes]
    
    u[interior_nodes] = spsolve(K_int, F_int)

print(f"Solution computed: min={u.min():.6f}, max={u.max():.6f}")

# Export interface data (temperature values)
interface_export_coords = []
interface_temps = []

for nid in sorted(interface_nodes):
    _, x, y = nodes[nid]
    interface_export_coords.append([x, y])
    interface_temps.append(u[nid])

# Compute flux at interface (outward normal is -x direction for subdomain B)
h_x = DOMAIN_B_WIDTH / nx
interface_fluxes = []

for nid in sorted(interface_nodes):
    _, x, y = nodes[nid]
    # Find node one element to the right
    x_right = x + h_x
    right_val = None
    for other_nid, (other_id, ox, oy) in enumerate(nodes):
        if abs(ox - x_right) < h_x/2 and abs(oy - y) < h_x/2:
            right_val = u[other_nid]
            break
    
    if right_val is not None:
        du_dx = (right_val - u[nid]) / h_x
        qn = K_B * du_dx  # outward normal flux (n = [-1, 0])
        interface_fluxes.append(qn)
    else:
        interface_fluxes.append(0.0)

# Write exports.json
exports = {
    "field_name": "temperature",
    "coordinates": interface_export_coords,
    "values": interface_temps,
    "normal_fluxes": interface_fluxes,
    "n_points": len(interface_export_coords)
}

with open(work_dir / "exports.json", 'w') as f:
    json.dump(exports, f, indent=2)

print(f"Exported {len(interface_export_coords)} interface points")
print(f"Interface temperature: min={min(interface_temps):.6f}, max={max(interface_temps):.6f}")
print(f"Interface flux: min={min(interface_fluxes):.6f}, max={max(interface_fluxes):.6f}")

# Write NDOF
with open(work_dir / "ndof.txt", 'w') as f:
    f.write(f"NDOF = {n_nodes}\n")

print(f"NDOF = {n_nodes}")
