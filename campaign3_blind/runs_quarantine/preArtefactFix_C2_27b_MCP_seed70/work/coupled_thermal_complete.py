#!/usr/bin/env python3
"""
Complete coupled thermal diffusion simulation.
Uses manual assembly with scipy for maximum portability.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Problem parameters
L_A = 0.625
L_B = 0.875  
H = 1.0
k_A = 1.0
k_B = 200.0
x_interface = 0.625
x_right = 1.5

# Get mesh level from environment
level = int(os.environ.get('LEVEL', '1'))
work_dir = os.environ.get('WORK_DIR', '.')
side = os.environ.get('SIDE', 'A')

# Mesh parameters based on level
mesh_params = {
    1: {"nx_A": 5, "ny": 8, "nx_B": 7},
    2: {"nx_A": 10, "ny": 16, "nx_B": 14},
    3: {"nx_A": 20, "ny": 32, "nx_B": 28}
}[level]

if side == 'A':
    nx = mesh_params["nx_A"]
    L = L_A
    k = k_A
    x_left = 0.0
else:
    nx = mesh_params["nx_B"]
    L = L_B
    k = k_B
    x_left = x_interface

ny = mesh_params["ny"]

# Source terms
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def read_imports():
    imports_file = os.path.join(work_dir, 'imports.json')
    if not os.path.exists(imports_file):
        return None
    try:
        with open(imports_file, 'r') as f:
            imports = json.load(f)
        partner = 'B' if side == 'A' else 'A'
        return imports.get(partner)
    except Exception as e:
        print(f"Error reading imports: {e}")
        return None

def write_exports(exports):
    exports_file = os.path.join(work_dir, 'exports.json')
    with open(exports_file, 'w') as f:
        json.dump({side: exports}, f, indent=2)

def interpolate_to_y(values, y_target, coords):
    ys = np.array([c[1] for c in coords])
    vs = np.array(values)
    
    mask = ys <= y_target
    if not np.any(mask):
        return vs[0]
    if np.all(mask):
        return vs[-1]
    
    idx = np.where(mask)[0][-1]
    if ys[idx+1] == ys[idx]:
        return vs[idx]
    
    t = (y_target - ys[idx]) / (ys[idx+1] - ys[idx])
    return vs[idx] * (1-t) + vs[idx+1] * t

def solve_subdomain(interface_data=None):
    # Generate nodes
    nodes = []
    node_map = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x_left + i * L / nx
            y = j * H / ny
            nodes.append((x, y))
            node_map[(i, j)] = nid
            nid += 1
    
    n_nodes = nid - 1
    
    # Generate triangular elements
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            elements.append([n1, n2, n4])
            elements.append([n2, n3, n4])
    
    # Build stiffness matrix and load vector
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    source_func = source_A if side == 'A' else source_B
    
    for elem in elements:
        ids = [n - 1 for n in elem]
        
        xe = np.array([nodes[n-1][0] for n in elem])
        ye = np.array([nodes[n-1][1] for n in elem])
        
        area = 0.5 * abs((xe[1]-xe[0])*(ye[2]-ye[0]) - (xe[2]-xe[0])*(ye[1]-ye[0]))
        
        b = np.array([ye[1]-ye[2], ye[2]-ye[0], ye[0]-ye[1]])
        c = np.array([xe[2]-xe[1], xe[0]-xe[2], xe[1]-xe[0]])
        
        Ke = (k / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        xm = np.mean(xe)
        ym = np.mean(ye)
        fm = source_func(xm, ym)
        Fe = np.array([fm * area / 3.0, fm * area / 3.0, fm * area / 3.0])
        
        for a in range(3):
            for bb in range(3):
                K[ids[a], ids[bb]] += Ke[a, bb]
            F[ids[a]] += Fe[a]
    
    K = csr_matrix(K)
    
    # Boundary conditions
    dirichlet_ids = set()
    dirichlet_values = {}
    
    if side == 'A':
        # Left boundary (x=0): Dirichlet u=0
        for j in range(ny + 1):
            did = node_map[(0, j)] - 1
            dirichlet_ids.add(did)
            dirichlet_values[did] = 0.0
        
        # Bottom boundary (y=0): Dirichlet u=0
        for i in range(1, nx + 1):
            did = node_map[(i, 0)] - 1
            dirichlet_ids.add(did)
            dirichlet_values[did] = 0.0
        
        # Top boundary (y=H): Dirichlet u=0
        for i in range(1, nx + 1):
            did = node_map[(i, ny)] - 1
            dirichlet_ids.add(did)
            dirichlet_values[did] = 0.0
        
        # Interface nodes (right boundary)
        interface_node_ids = [node_map[(nx, j)] - 1 for j in range(ny + 1)]
        
        # Apply interface temperature as Dirichlet
        if interface_data is not None and 'values' in interface_data:
            for j, nid in enumerate(interface_node_ids):
                y = j * H / ny
                T = interpolate_to_y(interface_data['values'], y, interface_data['coordinates'])
                dirichlet_ids.add(nid)
                dirichlet_values[nid] = T
    else:
        # Side B (Neumann side)
        # Right boundary (x=x_right): Dirichlet u=0
        for j in range(ny + 1):
            did = node_map[(nx, j)] - 1
            dirichlet_ids.add(did)
            dirichlet_values[did] = 0.0
        
        # Bottom boundary (y=0): Dirichlet u=0
        for i in range(nx + 1):
            did = node_map[(i, 0)] - 1
            dirichlet_ids.add(did)
            dirichlet_values[did] = 0.0
        
        # Top boundary (y=H): Dirichlet u=0
        for i in range(nx + 1):
            did = node_map[(i, ny)] - 1
            dirichlet_ids.add(did)
            dirichlet_values[did] = 0.0
        
        # Interface nodes (left boundary of B)
        interface_node_ids = [node_map[(0, j)] - 1 for j in range(ny + 1)]
        
        # Apply Neumann BC from imported flux
        if interface_data is not None and 'normal_fluxes' in interface_data:
            for j, nid in enumerate(interface_node_ids):
                y = j * H / ny
                q = interpolate_to_y(interface_data['normal_fluxes'], y, interface_data['coordinates'])
                
                if j == 0 or j == ny:
                    edge_length = H / (2 * ny)
                else:
                    edge_length = H / ny
                
                F[nid] += q * edge_length
    
    # Solve with Dirichlet BCs
    interior_ids = sorted(set(range(n_nodes)) - dirichlet_ids)
    
    if len(interior_ids) == 0:
        print("No interior nodes!")
        return None
    
    K_int = K[np.ix_(interior_ids, interior_ids)]
    F_int = F[interior_ids].copy()
    
    # Subtract known Dirichlet contributions
    for did in dirichlet_ids:
        if did >= 0 and did < n_nodes:
            row = K.getrow(did)
            for col_idx, val in zip(row.indices, row.data):
                if col_idx in interior_ids:
                    local_col = interior_ids.index(col_idx)
                    F_int[local_col] -= val * dirichlet_values.get(did, 0.0)
    
    u_int = spsolve(K_int, F_int)
    
    u = np.zeros(n_nodes)
    for idx, i in enumerate(interior_ids):
        u[i] = u_int[idx]
    
    # Set Dirichlet values
    for did, dval in dirichlet_values.items():
        u[did] = dval
    
    # Interface data
    if side == 'A':
        interface_node_ids = [node_map[(nx, j)] - 1 for j in range(ny + 1)]
    else:
        interface_node_ids = [node_map[(0, j)] - 1 for j in range(ny + 1)]
    
    interface_temps = [u[nid] for nid in interface_node_ids]
    interface_coords = [[x_interface, j * H / ny] for j in range(ny + 1)]
    
    # Compute flux at interface
    interface_fluxes = []
    hx = L / nx
    
    for j in range(ny + 1):
        y = j * H / ny
        
        if side == 'A':
            # Outward normal is +x
            x_interior = x_interface - hx
            min_dist = float('inf')
            nearest_val = 0
            for i in range(nx):
                xn = x_left + i * L / nx
                dist = abs(xn - x_interior)
                if dist < min_dist:
                    min_dist = dist
                    nearest_val = u[node_map[(i, j)] - 1]
            
            du_dx = (interface_temps[j] - nearest_val) / hx
            q = -k * du_dx
        else:
            # Outward normal is -x
            x_interior = x_interface + hx
            min_dist = float('inf')
            nearest_val = 0
            for i in range(1, nx + 1):
                xn = x_left + i * L / nx
                dist = abs(xn - x_interior)
                if dist < min_dist:
                    min_dist = dist
                    nearest_val = u[node_map[(i, j)] - 1]
            
            du_dx = (nearest_val - interface_temps[j]) / hx
            q = -k * du_dx * (-1)
        
        interface_fluxes.append(q)
    
    return {
        'coordinates': interface_coords,
        'values': interface_temps,
        'normal_fluxes': interface_fluxes,
        'n_dof': n_nodes,
        'solution': u,
        'nodes': nodes,
        'node_map': node_map
    }

def main():
    print(f"Solving subdomain {side} at level {level}")
    print(f"Domain: [{x_left}, {x_left + L}] x [0, {H}], nx={nx}, ny={ny}")
    print(f"Conductivity: {k}")
    
    imports = read_imports()
    
    result = solve_subdomain(imports)
    
    if result is None:
        print("Failed to solve")
        sys.exit(1)
    
    exports = {
        'field_name': 'temperature',
        'n_points': len(result['coordinates']),
        'coordinates': result['coordinates'],
        'values': result['values'],
        'normal_fluxes': result['normal_fluxes']
    }
    write_exports(exports)
    
    log_file = os.path.join(work_dir, f'run_level{level}_{side}.log')
    with open(log_file, 'w') as f:
        f.write(f"Manual assembly solver completed\n")
        f.write(f"Subdomain {side}: [{x_left}, {x_left + L}] x [0, {H}]\n")
        f.write(f"Mesh: {nx} x {ny} elements\n")
        f.write(f"Conductivity: {k}\n")
        f.write(f"NDOF = {result['n_dof']}\n")
    
    print(f"Completed. NDOF = {result['n_dof']}")

if __name__ == '__main__':
    main()
