#!/usr/bin/env python3
"""
Complete coupled thermal simulation - produces all required output files.
Uses scikit-fem for both subdomains (can be adapted to use 4C for one side).
"""
import os, sys, json, numpy as np
from skfem import *
from skfem.models.poisson import laplace

work_dir = os.getcwd()
level = int(os.environ.get('LEVEL', 1))
resolution = int(os.environ.get('RESOLUTION', 8))

# Problem parameters
x_A_max = 0.625
x_B_min = 0.625
x_B_max = 1.5
y_max = 1.0
k_A = 1.0
k_B = 200.0

nx_A = int(x_A_max * resolution)
nx_B = int((x_B_max - x_B_min) * resolution)
ny = int(y_max * resolution)

# Source terms
def source_A(X):
    x, y = X
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(X):
    x, y = X
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def create_mesh(nx, ny, x_start, x_end, y_end):
    p = np.zeros((2, (nx+1)*(ny+1)))
    e = np.zeros((3, nx*ny*2), dtype=int)
    
    nid = 0
    node_map = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            p[0, nid] = x_start + i / resolution
            p[1, nid] = j / resolution
            node_map[(i, j)] = nid
            nid += 1
    
    eid = 0
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            e[:, eid] = [n1, n2, n4]
            eid += 1
            e[:, eid] = [n2, n3, n4]
            eid += 1
    
    return MeshTri(p, e), node_map

def assemble_system(mesh, basis, k, source_func):
    K = asm(laplace, basis) * k
    F = np.zeros(basis.nodal_dofs.max() + 1)
    for elem in mesh.elements():
        coords = mesh.p[:, elem]
        area = 0.5 * abs(np.linalg.det(np.column_stack([np.ones(3), coords[:2]])))
        centroid = coords.mean(axis=1)
        f_val = source_func(centroid)
        for idx in elem:
            F[idx] += f_val * area / 3
    return K, F

def get_interface_nodes(mesh, x_interface, tol=1e-6):
    nodes = mesh.p
    iface_indices = np.where(np.abs(nodes[0] - x_interface) < tol)[0]
    return iface_indices[np.argsort(nodes[1, iface_indices])]

def compute_flux(mesh, u, k, x_interface, outward_normal_x, tol=1e-6):
    nodes = mesh.p
    iface_indices = np.where(np.abs(nodes[0] - x_interface) < tol)[0]
    dx = 1.0 / resolution
    
    fluxes = []
    for idx in iface_indices:
        x_i, y_i = nodes[0, idx], nodes[1, idx]
        best_dist = float('inf')
        best_u = None
        for jdx in range(len(nodes[0])):
            if outward_normal_x > 0 and nodes[0, jdx] < x_i - dx/3:
                dist = (nodes[0, jdx] - (x_i - dx))**2 + (nodes[1, jdx] - y_i)**2
                if dist < best_dist:
                    best_dist = dist
                    best_u = u[jdx]
            elif outward_normal_x < 0 and nodes[0, jdx] > x_i + dx/3:
                dist = (nodes[0, jdx] - (x_i + dx))**2 + (nodes[1, jdx] - y_i)**2
                if dist < best_dist:
                    best_dist = dist
                    best_u = u[jdx]
        if best_u is not None:
            grad_u_x = (best_u - u[idx]) / (outward_normal_x * dx)
            flux = -k * grad_u_x * outward_normal_x
        else:
            flux = 0.0
        fluxes.append(flux)
    return fluxes

# Coupling iteration
max_iter = 50
tol = 1e-6
theta = 0.5

mesh_A, _ = create_mesh(nx_A, ny, 0, x_A_max, y_max)
basis_A = Basis(mesh_A, ElementTriP1())
K_A, F_A = assemble_system(mesh_A, basis_A, k_A, source_A)

mesh_B, _ = create_mesh(nx_B, ny, x_B_min, x_B_max, y_max)
basis_B = Basis(mesh_B, ElementTriP1())
K_B, F_B = assemble_system(mesh_B, basis_B, k_B, source_B)

tol_bc = 1e-6
nodes_A = mesh_A.p
nodes_B = mesh_B.p

A_left = np.where(np.abs(nodes_A[0]) < tol_bc)[0]
A_bottom = np.where(np.abs(nodes_A[1]) < tol_bc)[0]
A_top = np.where(np.abs(nodes_A[1] - y_max) < tol_bc)[0]
A_right = get_interface_nodes(mesh_A, x_A_max)
A_dirichlet = np.union_1d(A_left, np.union_1d(A_bottom, A_top))
A_dirichlet = np.setdiff1d(A_dirichlet, A_right)

B_left = get_interface_nodes(mesh_B, x_B_min)
B_right = np.where(np.abs(nodes_B[0] - x_B_max) < tol_bc)[0]
B_bottom = np.where(np.abs(nodes_B[1]) < tol_bc)[0]
B_top = np.where(np.abs(nodes_B[1] - y_max) < tol_bc)[0]
B_dirichlet = np.union_1d(B_right, np.union_1d(B_bottom, B_top))
B_dirichlet = np.setdiff1d(B_dirichlet, B_left)

T_interface_B = np.zeros(len(B_left))
residuals = []

for iteration in range(max_iter):
    D_A = np.concatenate([A_dirichlet, A_right])
    u_A = solve_condensed(K_A, F_A, D=D_A)
    u_A[A_dirichlet] = 0.0
    u_A[A_right] = T_interface_B
    
    q_A = compute_flux(mesh_A, u_A, k_A, x_A_max, outward_normal_x=1)
    
    F_B_modified = F_B.copy()
    dy = 1.0 / ny
    for i, idx in enumerate(B_left):
        if i == 0 or i == len(B_left) - 1:
            F_B_modified[idx] += q_A[i] * dy / 2
        else:
            F_B_modified[idx] += q_A[i] * dy
    
    D_B = B_dirichlet
    u_B = solve_condensed(K_B, F_B_modified, D=D_B)
    u_B[B_dirichlet] = 0.0
    
    T_interface_B_new = u_B[B_left]
    
    residual = np.max(np.abs(T_interface_B_new - T_interface_B)) / (np.max(np.abs(T_interface_B)) + 1e-15)
    residuals.append(residual)
    
    T_interface_B = (1 - theta) * T_interface_B + theta * T_interface_B_new
    
    if residual < tol:
        print(f"Converged at iteration {iteration + 1}, residual = {residual:.2e}")
        break
else:
    print(f"Did not converge after {max_iter} iterations, final residual = {residual:.2e}")

# Final solve
u_A[A_dirichlet] = 0.0
u_A[A_right] = T_interface_B
u_A = solve_condensed(K_A, F_A, D=np.concatenate([A_dirichlet, A_right]))

F_B_modified = F_B.copy()
for i, idx in enumerate(B_left):
    if i == 0 or i == len(B_left) - 1:
        F_B_modified[idx] += q_A[i] * dy / 2
    else:
        F_B_modified[idx] += q_A[i] * dy
u_B = solve_condensed(K_B, F_B_modified, D=B_dirichlet)
u_B[B_dirichlet] = 0.0

q_A_final = compute_flux(mesh_A, u_A, k_A, x_A_max, outward_normal_x=1)
q_B_final = compute_flux(mesh_B, u_B, k_B, x_B_min, outward_normal_x=-1)

print(f"NDOF_A = {len(u_A)}")
print(f"NDOF_B = {len(u_B)}")
print(f"Final residual = {residuals[-1]:.2e}")
print(f"Iterations = {len(residuals)}")

# Write residual history
with open(f'residual_level{level}.csv', 'w') as f:
    f.write('iteration,interface_residual\n')
    for i, r in enumerate(residuals):
        f.write(f'{i+1},{r:.15e}\n')

# Probe points for subdomain A
probe_points_A = []
for i_y in range(44):
    for i_x in range(44):
        x = 0 + (i_x + 0.5) * 0.625 / 44
        y = 0 + (i_y + 0.5) * 1 / 44
        probe_points_A.append((x, y))

# Probe points for subdomain B
probe_points_B = []
for i_y in range(44):
    for i_x in range(44):
        x = 0.625 + (i_x + 0.5) * 0.875 / 44
        y = 0 + (i_y + 0.5) * 1 / 44
        probe_points_B.append((x, y))

# Interpolate solution at probe points using barycentric coordinates
def interpolate_at_point(mesh, u, x, y):
    """Find element containing point and interpolate."""
    nodes = mesh.p
    best_elem = None
    best_area = float('inf')
    
    for elem in mesh.elements():
        coords = nodes[:, elem]
        # Check if point is inside triangle using barycentric coordinates
        denom = ((coords[1, 1] - coords[1, 2]) * (coords[0, 0] - coords[0, 2]) + 
                 (coords[0, 2] - coords[0, 1]) * (coords[1, 0] - coords[1, 2]))
        if abs(denom) < 1e-15:
            continue
        a = (((coords[1, 1] - coords[1, 2]) * (x - coords[0, 2]) + 
              (coords[0, 2] - coords[0, 1]) * (y - coords[1, 2])) / denom)
        b = (((coords[1, 2] - coords[1, 0]) * (x - coords[0, 2]) + 
              (coords[0, 0] - coords[0, 2]) * (y - coords[1, 2])) / denom)
        c = 1 - a - b
        
        if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
            return a * u[elem[0]] + b * u[elem[1]] + c * u[elem[2]]
    
    # Point not found in any element - return nearest node value
    dists = [(nodes[0, i] - x)**2 + (nodes[1, i] - y)**2 for i in range(len(nodes[0]))]
    return u[np.argmin(dists)]

# Write solution CSVs
with open(f'solution_level{level}_A.csv', 'w') as f:
    f.write('x,y,u\n')
    for x, y in probe_points_A:
        val = interpolate_at_point(mesh_A, u_A, x, y)
        f.write(f'{x:.15e},{y:.15e},{val:.15e}\n')

with open(f'solution_level{level}_B.csv', 'w') as f:
    f.write('x,y,u\n')
    for x, y in probe_points_B:
        val = interpolate_at_point(mesh_B, u_B, x, y)
        f.write(f'{x:.15e},{y:.15e},{val:.15e}\n')

# Interface probe points: (5/8, (j+0.5)/44) for j = 11, ..., 32
interface_probes = [(5/8, (j + 0.5) / 44) for j in range(11, 33)]

# Get interface values and fluxes
def get_interface_data(mesh, u, q, x_interface, tol=1e-6):
    nodes = mesh.p
    iface_indices = np.where(np.abs(nodes[0] - x_interface) < tol)[0]
    iface_sorted = iface_indices[np.argsort(nodes[1, iface_indices])]
    
    data = []
    for x_p, y_p in interface_probes:
        # Find closest interface node
        best_idx = min(iface_sorted, key=lambda i: abs(nodes[1, i] - y_p))
        data.append((x_p, y_p, u[best_idx], q[iface_sorted.tolist().index(best_idx)]))
    return data

data_A = get_interface_data(mesh_A, u_A, q_A_final, x_A_max)
data_B = get_interface_data(mesh_B, u_B, q_B_final, x_B_min)

with open(f'interface_level{level}_A.csv', 'w') as f:
    f.write('x,y,u,qn\n')
    for x, y, u_val, q_val in data_A:
        f.write(f'{x:.15e},{y:.15e},{u_val:.15e},{q_val:.15e}\n')

with open(f'interface_level{level}_B.csv', 'w') as f:
    f.write('x,y,u,qn\n')
    for x, y, u_val, q_val in data_B:
        f.write(f'{x:.15e},{y:.15e},{u_val:.15e},{q_val:.15e}\n')

print("Output files written successfully")
