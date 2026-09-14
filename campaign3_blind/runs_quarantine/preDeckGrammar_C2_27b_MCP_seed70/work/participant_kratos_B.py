#!/usr/bin/env python3
"""
Kratos participant for Subdomain B (Neumann side) in coupled thermal problem.
Domain: (0.625, 1.5) x (0, 1), k = 200
Receives flux at interface from partner A, exports temperature.
"""
import os, sys, json, numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

work_dir = os.getcwd()
level = int(os.environ.get('LEVEL', 1))
resolution = int(os.environ.get('RESOLUTION', 8))

# Subdomain B parameters
x_B_min = 0.625
x_B_max = 1.5
y_max = 1.0
k_B = 200.0

nx_B = int((x_B_max - x_B_min) * resolution)
ny = int(y_max * resolution)

# Source term f(x,y) in subdomain B
def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Generate mesh nodes (local coordinates shifted to start at x_B_min)
nodes = []
node_map = {}
nid = 1
for j in range(ny + 1):
    for i in range(nx_B + 1):
        x = x_B_min + i / resolution
        y = j / resolution
        nodes.append((nid, x, y))
        node_map[(i, j)] = nid
        nid += 1

n_nodes = len(nodes)

# Generate triangular elements (split each quad into two triangles)
elements = []
for j in range(ny):
    for i in range(nx_B):
        n1 = node_map[(i, j)]
        n2 = node_map[(i+1, j)]
        n3 = node_map[(i+1, j+1)]
        n4 = node_map[(i, j+1)]
        elements.append([n1, n2, n4])  # triangle 1
        elements.append([n2, n3, n4])  # triangle 2

n_elements = len(elements)

# Identify boundary nodes
left_nodes = set(node_map[(0, j)] for j in range(ny + 1))      # interface (x = 0.625)
right_nodes = set(node_map[(nx_B, j)] for j in range(ny + 1))  # right (x = 1.5)
bottom_nodes = set(node_map[(i, 0)] for i in range(nx_B + 1))  # bottom (y = 0)
top_nodes = set(node_map[(i, ny)] for i in range(nx_B + 1))    # top (y = 1)

# Dirichlet nodes: right, bottom, top (but not corners that are also on interface)
dirichlet_nodes = right_nodes | bottom_nodes | top_nodes - left_nodes

# Read imports from partner A
imports_file = os.path.join(work_dir, 'imports.json')
if os.path.exists(imports_file):
    with open(imports_file, 'r') as f:
        imports = json.load(f)
    if 'A' in imports:
        imp_data = imports['A']
        iface_coords = np.array(imp_data['coordinates'])
        iface_fluxes = np.array(imp_data.get('normal_fluxes', [0.0] * len(imp_data['values'])))
        # Sort by y
        sidx = np.argsort(iface_coords[:, 1])
        y_iface_sorted = iface_coords[sidx, 1]
        q_iface_sorted = iface_fluxes[sidx]
    else:
        y_iface_sorted = np.linspace(0, 1, ny + 1)
        q_iface_sorted = np.zeros(len(y_iface_sorted))
else:
    y_iface_sorted = np.linspace(0, 1, ny + 1)
    q_iface_sorted = np.zeros(len(y_iface_sorted))

# Create mapping from y-coordinate to flux value for interface nodes
# Interface nodes have y = j/ny for j = 0..ny
flux_at_interface_y = {}
for j in range(ny + 1):
    y_j = j / ny
    # Find closest flux value
    best_idx = np.argmin(np.abs(y_iface_sorted - y_j))
    flux_at_interface_y[y_j] = q_iface_sorted[best_idx]

# Assemble stiffness matrix and load vector
K = lil_matrix((n_nodes, n_nodes))
F = np.zeros(n_nodes)

dx = 1.0 / resolution
dy = 1.0 / ny

for elem in elements:
    ids = [n - 1 for n in elem]  # convert to 0-based
    coords = np.array([nodes[n-1][1:] for n in elem])  # (x, y) for each node
    
    # Triangle area
    x = coords[:, 0]
    y = coords[:, 1]
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    
    # Gradient of shape functions
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    # Element stiffness matrix (with conductivity k_B)
    Ke = (k_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
    
    for a in range(3):
        for bb in range(3):
            K[ids[a], ids[bb]] += Ke[a, bb]
    
    # Load vector: integrate f * N_i over element using centroid rule
    xc = np.mean(x)
    yc = np.mean(y)
    f_val = source_B(xc, yc)
    for a in range(3):
        F[ids[a]] += f_val * area / 3.0

# Apply Neumann BC on interface (left boundary)
# The imported flux is the outward normal flux from domain A
# For domain B, the outward normal at left boundary points LEFT (-x direction)
# So we need to apply the negative of the imported flux
for j in range(ny + 1):
    node_id = node_map[(0, j)]
    idx = node_id - 1
    y_j = j / ny
    q_imported = flux_at_interface_y.get(y_j, 0.0)
    # Neumann contribution: integral of q * N_i along edge
    # For linear elements, this is q * dy / 2 for each endpoint
    # But since we're applying at nodes, we add q * dy directly
    # The sign: imported flux is A's outward flux (pointing +x)
    # B's outward normal at left is -x, so we apply -q_imported
    if j == 0 or j == ny:
        # Corner nodes get half weight
        F[idx] += (-q_imported) * dy / 2
    else:
        F[idx] += (-q_imported) * dy

# Apply Dirichlet BCs
K_csr = K.tocsr()
u = np.zeros(n_nodes)

# Set Dirichlet values
for node_id in dirichlet_nodes:
    idx = node_id - 1
    u[idx] = 0.0

# Solve for interior nodes
interior_indices = [i for i in range(n_nodes) if (i+1) not in dirichlet_nodes]
if interior_indices:
    K_int = K_csr[np.ix_(interior_indices, interior_indices)]
    F_int = F[interior_indices]
    u[interior_indices] = spsolve(K_int, F_int)

# Export results
# Get interface nodes sorted by y
interface_node_ids = sorted(left_nodes)
interface_data = {
    'field_name': 'temperature',
    'n_points': len(interface_node_ids),
    'coordinates': [],
    'values': [],
    'normal_fluxes': []
}

for node_id in interface_node_ids:
    idx = node_id - 1
    x, y = nodes[node_id][1], nodes[node_id][2]
    interface_data['coordinates'].append([x, y, 0.0])
    interface_data['values'].append(u[idx])

# Compute outward normal flux for domain B at interface
# Outward normal for B at left boundary is (-1, 0, 0)
# Flux q = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
interface_fluxes = []
for node_id in interface_node_ids:
    idx = node_id - 1
    x, y = nodes[node_id][1], nodes[node_id][2]
    # Find interior node to the right
    best_dist = float('inf')
    best_temp = None
    for jdx, (jid, jx, jy) in enumerate(nodes):
        if jx > x + dx/3:
            dist = (jx - (x + dx))**2 + **(jy - y)2
            if dist < best_dist:
                best_dist = dist
                best_temp = u[jid - 1]
    
    if best_temp is not None:
        grad_u_x = (best_temp - u[idx]) / dx
        flux = k_B * grad_u_x  # outward normal is -x, so q = -k*(-du/dx) = k*du/dx
    else:
        flux = 0.0
    interface_fluxes.append(flux)

interface_data['normal_fluxes'] = interface_fluxes

with open(os.path.join(work_dir, 'exports.json'), 'w') as f:
    json.dump({'B': interface_data}, f)

print(f"NDOF = {n_nodes}")
print(f"Interface nodes: {len(interface_node_ids)}")
print(f"Temperature range: [{u.min():.6e}, {u.max():.6e}]")
