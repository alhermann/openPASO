#!/usr/bin/env python3
"""
scikit-fem participant for Subdomain A (Dirichlet side) in coupled thermal problem.
Domain: (0, 0.625) x (0, 1), k = 1
Receives temperature at interface from partner B, exports flux.
"""
import os, sys, json, numpy as np
from skfem import *
from skfem.models.poisson import laplace, mass

work_dir = os.getcwd()
level = int(os.environ.get('LEVEL', 1))
resolution = int(os.environ.get('RESOLUTION', 8))

# Subdomain A parameters
x_A_max = 0.625
y_max = 1.0
k_A = 1.0

nx_A = int(x_A_max * resolution)
ny = int(y_max * resolution)

# Source term f(x,y) in subdomain A
def source_A(X):
    x, y = X
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

# Create mesh using MeshTri
p = np.zeros((2, (nx_A+1)*(ny+1)))
e = np.zeros((3, nx_A*ny*2), dtype=int)

# Generate structured triangular mesh
nid = 0
node_map = {}
for j in range(ny + 1):
    for i in range(nx_A + 1):
        p[0, nid] = i / resolution
        p[1, nid] = j / resolution
        node_map[(i, j)] = nid
        nid += 1

# Create triangles by splitting each quad into two
eid = 0
for j in range(ny):
    for i in range(nx_A):
        n1 = node_map[(i, j)]
        n2 = node_map[(i+1, j)]
        n3 = node_map[(i+1, j+1)]
        n4 = node_map[(i, j+1)]
        # Triangle 1: n1, n2, n4
        e[:, eid] = [n1, n2, n4]
        eid += 1
        # Triangle 2: n2, n3, n4
        e[:, eid] = [n2, n3, n4]
        eid += 1

mesh = MeshTri(p, e)
mesh.scale([x_A_max, y_max])

# Create basis
basis = Basis(mesh, ElementTriP1())

# Assemble stiffness matrix (with conductivity k_A)
K = asm(laplace, basis) * k_A

# Assemble load vector using quadrature
F = np.zeros(basis.nodal_dofs().max() + 1)
for elem in mesh.elements():
    coords = mesh.p[:, elem]
    area = 0.5 * abs(np.linalg.det(np.column_stack([np.ones(3), coords[:2]])))
    centroid = coords.mean(axis=1)
    f_val = source_A(centroid)
    for idx in elem:
        F[idx] += f_val * area / 3

# Read imports from partner B
imports_file = os.path.join(work_dir, 'imports.json')
interface_temps = None
if os.path.exists(imports_file):
    with open(imports_file, 'r') as f:
        imports = json.load(f)
    if 'B' in imports:
        imp_data = imports['B']
        iface_coords = np.array(imp_data['coordinates'])
        iface_temps = np.array(imp_data['values'])
        sidx = np.argsort(iface_coords[:, 1])
        y_iface_sorted = iface_coords[sidx, 1]
        t_iface_sorted = iface_temps[sidx]

# Identify boundary nodes
tol = 1e-6
nodes = mesh.p
left_nodes = np.where(np.abs(nodes[0]) < tol)[0]
bottom_nodes = np.where(np.abs(nodes[1]) < tol)[0]
top_nodes = np.where(np.abs(nodes[1] - y_max) < tol)[0]
right_nodes = np.where(np.abs(nodes[0] - x_A_max) < tol)[0]

dirichlet_nodes = np.union_1d(left_nodes, np.union_1d(bottom_nodes, top_nodes))
dirichlet_nodes = np.setdiff1d(dirichlet_nodes, right_nodes)

right_nodes_sorted = right_nodes[np.argsort(nodes[1, right_nodes])]

D = np.concatenate([dirichlet_nodes, right_nodes_sorted])
u = solve_condensed(K, F, D=D)
u[dirichlet_nodes] = 0.0

if interface_temps is not None:
    for i, idx in enumerate(right_nodes_sorted):
        y_i = nodes[1, idx]
        j = np.argmin(np.abs(y_iface_sorted - y_i))
        u[idx] = t_iface_sorted[j]
else:
    u[right_nodes_sorted] = 0.0

# Compute flux at interface
interface_fluxes = []
for idx in right_nodes_sorted:
    x_i, y_i = nodes[0, idx], nodes[1, idx]
    dx = 1.0 / resolution
    
    best_dist = float('inf')
    best_u = None
    for jdx in range(len(nodes[0])):
        if nodes[0, jdx] < x_i - dx/3:
            dist = (nodes[0, jdx] - (x_i - dx))**2 + (nodes[1, jdx] - y_i)**2
            if dist < best_dist:
                best_dist = dist
                best_u = u[jdx]
    
    if best_u is not None:
        grad_u_x = (u[idx] - best_u) / dx
        flux = -k_A * grad_u_x
    else:
        flux = 0.0
    interface_fluxes.append(flux)

export_data = {
    'field_name': 'temperature',
    'n_points': len(right_nodes_sorted),
    'coordinates': [[x_A_max, nodes[1, idx], 0.0] for idx in right_nodes_sorted],
    'values': [u[idx] for idx in right_nodes_sorted],
    'normal_fluxes': interface_fluxes
}

with open(os.path.join(work_dir, 'exports.json'), 'w') as f:
    json.dump({'A': export_data}, f)

print(f"NDOF = {len(u)}")
print(f"Interface nodes: {len(right_nodes_sorted)}")
