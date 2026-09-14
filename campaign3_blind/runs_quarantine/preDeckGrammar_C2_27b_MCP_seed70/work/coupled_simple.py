#!/usr/bin/env python3
"""
Coupled thermal simulation - simplified version that works.
"""
import os, sys, json, numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

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
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def create_mesh_and_assemble(nx, ny, x_start, k, source_func):
    """Create mesh and assemble system for a rectangular domain."""
    n_nodes = (nx + 1) * (ny + 1)
    
    # Node coordinates
    nodes = {}
    node_map = {}
    nid = 0
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x_start + i / resolution
            y = j / resolution
            nodes[nid] = (x, y)
            node_map[(i, j)] = nid
            nid += 1
    
    # Create triangular elements
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            elements.append([n1, n2, n4])
            elements.append([n2, n3, n4])
    
    # Assemble stiffness matrix
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for elem in elements:
        coords = [nodes[n] for n in elem]
        x = np.array([c[0] for c in coords])
        y = np.array([c[1] for c in coords])
        
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        
        Ke = (k / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        for a in range(3):
            for bb in range(3):
                K[elem[a], elem[bb]] += Ke[a, bb]
        
        # Load vector
        xc = np.mean(x)
        yc = np.mean(y)
        f_val = source_func(xc, yc)
        for a in range(3):
            F[elem[a]] += f_val * area / 3
    
    return K.tocsr(), F, nodes, node_map, elements

# Create systems
K_A, F_A, nodes_A, node_map_A, elems_A = create_mesh_and_assemble(nx_A, ny, 0, k_A, source_A)
K_B, F_B, nodes_B, node_map_B, elems_B = create_mesh_and_assemble(nx_B, ny, x_B_min, k_B, source_B)

n_nodes_A = len(nodes_A)
n_nodes_B = len(nodes_B)

# Identify boundary nodes
tol_bc = 1e-6

# Domain A boundaries
A_left = [node_map_A[(0, j)] for j in range(ny + 1)]
A_bottom = [node_map_A[(i, 0)] for i in range(nx_A + 1)]
A_top = [node_map_A[(i, ny)] for i in range(nx_A + 1)]
A_right = [node_map_A[(nx_A, j)] for j in range(ny + 1)]

A_dirichlet = set(A_left + A_bottom + A_top) - set(A_right)

# Domain B boundaries  
B_left = [node_map_B[(0, j)] for j in range(ny + 1)]
B_right = [node_map_B[(nx_B, j)] for j in range(ny + 1)]
B_bottom = [node_map_B[(i, 0)] for i in range(nx_B + 1)]
B_top = [node_map_B[(i, ny)] for i in range(nx_B + 1)]

B_dirichlet = set(B_right + B_bottom + B_top) - set(B_left)

# Coupling iteration
max_iter = 50
tol = 1e-6
theta = 0.5

T_interface_B = np.zeros(len(B_left))
residuals = []

dy = 1.0 / ny

for iteration in range(max_iter):
    # Solve domain A with Dirichlet BC from B
    D_A = list(A_dirichlet) + A_right
    u_A = np.zeros(n_nodes_A)
    u_A[[n-1 for n in D_A if n > 0]] = 0  # Will be overwritten
    
    # Set interface values
    for i, idx in enumerate(A_right):
        u_A[idx] = T_interface_B[i]
    
    # Set outer Dirichlet to 0
    for idx in A_dirichlet:
        u_A[idx] = 0.0
    
    # Solve for interior
    interior_A = [i for i in range(n_nodes_A) if (i+1) not in D_A]
    if interior_A:
        K_int = K_A[np.ix_(interior_A, interior_A)]
        F_int = F_A[interior_A] - K_A[np.ix_(interior_A, D_A)].dot(u_A[D_A])
        u_A[interior_A] = spsolve(K_int, F_int)
    
    # Compute flux at interface (outward normal is +x)
    q_A = []
    dx = 1.0 / resolution
    for idx in A_right:
        x_i, y_i = nodes_A[idx]
        best_dist = float('inf')
        best_u = None
        for jdx in range(n_nodes_A):
            jx, jy = nodes_A[jdx]
            if jx < x_i - dx/3:
                dist = (jx - (x_i - dx))**2 + (jy - y_i)**2
                if dist < best_dist:
                    best_dist = dist
                    best_u = u_A[jdx]
        if best_u is not None:
            grad_u_x = (u_A[idx] - best_u) / dx
            flux = -k_A * grad_u_x
        else:
            flux = 0.0
        q_A.append(flux)
    
    # Solve domain B with Neumann BC from A
    F_B_mod = F_B.copy()
    for i, idx in enumerate(B_left):
        if i == 0 or i == len(B_left) - 1:
            F_B_mod[idx] += q_A[i] * dy / 2
        else:
            F_B_mod[idx] += q_A[i] * dy
    
    D_B = list(B_dirichlet)
    u_B = np.zeros(n_nodes_B)
    
    for idx in B_dirichlet:
        u_B[idx] = 0.0
    
    interior_B = [i for i in range(n_nodes_B) if (i+1) not in D_B]
    if interior_B:
        K_int = K_B[np.ix_(interior_B, interior_B)]
        F_int = F_B_mod[interior_B] - K_B[np.ix_(interior_B, D_B)].dot(u_B[D_B])
        u_B[interior_B] = spsolve(K_int, F_int)
    
    T_interface_B_new = u_B[B_left]
    
    residual = np.max(np.abs(T_interface_B_new - T_interface_B)) / (np.max(np.abs(T_interface_B)) + 1e-15)
    residuals.append(residual)
    
    T_interface_B = (1 - theta) * T_interface_B + theta * T_interface_B_new
    
    if residual < tol:
        print(f"Converged at iteration {iteration + 1}, residual = {residual:.2e}")
        break
else:
    print(f"Did not converge after {max_iter} iterations")

# Final solve without relaxation
D_A = list(A_dirichlet) + A_right
u_A = np.zeros(n_nodes_A)
for i, idx in enumerate(A_right):
    u_A[idx] = T_interface_B[i]
for idx in A_dirichlet:
    u_A[idx] = 0.0
interior_A = [i for i in range(n_nodes_A) if (i+1) not in D_A]
if interior_A:
    K_int = K_A[np.ix_(interior_A, interior_A)]
    F_int = F_A[interior_A] - K_A[np.ix_(interior_A, D_A)].dot(u_A[D_A])
    u_A[interior_A] = spsolve(K_int, F_int)

F_B_mod = F_B.copy()
for i, idx in enumerate(B_left):
    if i == 0 or i == len(B_left) - 1:
        F_B_mod[idx] += q_A[i] * dy / 2
    else:
        F_B_mod[idx] += q_A[i] * dy

D_B = list(B_dirichlet)
u_B = np.zeros(n_nodes_B)
for idx in B_dirichlet:
    u_B[idx] = 0.0
interior_B = [i for i in range(n_nodes_B) if (i+1) not in D_B]
if interior_B:
    K_int = K_B[np.ix_(interior_B, interior_B)]
    F_int = F_B_mod[interior_B] - K_B[np.ix_(interior_B, D_B)].dot(u_B[D_B])
    u_B[interior_B] = spsolve(K_int, F_int)

print(f"NDOF_A = {n_nodes_A}")
print(f"NDOF_B = {n_nodes_B}")
print(f"Final residual = {residuals[-1]:.2e}")
print(f"Iterations = {len(residuals)}")

# Write residual history
with open(f'residual_level{level}.csv', 'w') as f:
    f.write('iteration,interface_residual\n')
    for i, r in enumerate(residuals):
        f.write(f'{i+1},{r:.15e}\n')

print("Done!")
