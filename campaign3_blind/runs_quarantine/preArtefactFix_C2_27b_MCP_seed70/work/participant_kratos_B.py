#!/usr/bin/env python3
"""
Kratos Participant for Subdomain B (Neumann side) - Coupled thermal diffusion
- Receives flux from interface (from 4C/A)
- Applies as Neumann BC at x = 0.625
- Returns temperature at interface
- Outward normal for B at interface is -x direction
"""

import json
import os
import sys
import numpy as np
from pathlib import Path

# Problem parameters
L_A = 0.625  # Width of subdomain A (interface location)
L_B = 0.875  # Width of subdomain B
H = 1.0      # Height
k_B = 200.0  # Conductivity in B
x_interface = 0.625
x_right = 1.5  # Right boundary of B

# Get mesh level and work directory from environment
level = int(os.environ.get('LEVEL', '1'))
work_dir = os.environ.get('WORK_DIR', '.')

# Mesh parameters based on level
mesh_params = {
    1: {"nx_B": 7, "ny": 8},
    2: {"nx_B": 14, "ny": 16},
    3: {"nx_B": 28, "ny": 32}
}[level]

nx = mesh_params["nx_B"]
ny = mesh_params["ny"]

# Source term for subdomain B (polynomial)
def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def read_imports():
    """Read imports.json if it exists."""
    imports_file = os.path.join(work_dir, 'imports.json')
    if not os.path.exists(imports_file):
        return None
    try:
        with open(imports_file, 'r') as f:
            imports = json.load(f)
        return imports.get('A')
    except:
        return None

def interpolate_to_y(values, y_target, coords):
    """Interpolate interface flux values to a specific y coordinate."""
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

def solve_subdomain(interface_flux_values=None, interface_coords=None):
    """Solve the heat equation on subdomain B using manual assembly."""
    
    # Generate nodes
    nodes = []
    node_map = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x_interface + i * L_B / nx
            y = j * H / ny
            nodes.append((x, y))
            node_map[(i, j)] = nid
            nid += 1
    
    n_nodes = nid - 1
    
    # Generate triangular elements (split each quad into two triangles)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            # Two triangles per quad
            elements.append([n1, n2, n4])
            elements.append([n2, n3, n4])
    
    # Build stiffness matrix and load vector
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for elem in elements:
        ids = [n - 1 for n in elem]  # Convert to 0-based
        
        # Element coordinates
        xe = np.array([nodes[n-1][0] for n in elem])
        ye = np.array([nodes[n-1][1] for n in elem])
        
        # Element area
        area = 0.5 * abs((xe[1]-xe[0])*(ye[2]-ye[0]) - (xe[2]-xe[0])*(ye[1]-ye[0]))
        
        # Shape function derivatives
        b = np.array([ye[1]-ye[2], ye[2]-ye[0], ye[0]-ye[1]])
        c = np.array([xe[2]-xe[1], xe[0]-xe[2], xe[1]-xe[0]])
        
        # Stiffness matrix for -div(k grad u)
        Ke = (k_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        # Load vector (source term integrated over element)
        # Use midpoint quadrature
        xm = np.mean(xe)
        ym = np.mean(ye)
        fm = source_B(xm, ym)
        Fe = fm * area / 3.0  # Equal distribution to nodes
        
        # Assemble
        for a in range(3):
            for bb in range(3):
                K[ids[a], ids[bb]] += Ke[a, bb]
            F[ids[a]] += Fe[a]
    
    K = csr_matrix(K)
    
    # Boundary conditions
    # Left boundary (x = x_interface): Neumann from imported flux
    # Right boundary (x = x_right): Dirichlet u = 0
    # Top/bottom: natural (zero flux)
    
    # Interface nodes (left boundary of B)
    interface_node_ids = [node_map[(0, j)] - 1 for j in range(ny + 1)]
    
    # Right boundary nodes (Dirichlet u = 0)
    right_node_ids = [node_map[(nx, j)] - 1 for j in range(ny + 1)]
    
    # Bottom boundary nodes (Dirichlet u = 0)
    bottom_node_ids = [node_map[(i, 0)] - 1 for i in range(nx + 1)]
    
    # Top boundary nodes (Dirichlet u = 0)
    top_node_ids = [node_map[(i, ny)] - 1 for i in range(nx + 1)]
    
    # All Dirichlet nodes (u = 0)
    dirichlet_ids = set(right_node_ids + bottom_node_ids + top_node_ids)
    
    # Apply Neumann BC on interface
    # The imported flux is the outward normal flux from domain A
    # For domain B, the outward normal at the interface is -x
    # So we need to apply the flux with the correct sign
    if interface_flux_values is not None and len(interface_flux_values) > 0:
        for j, nid in enumerate(interface_node_ids):
            y = j * H / ny
            # Interpolate flux value
            q = interpolate_to_y(interface_flux_values, y, interface_coords)
            
            # The flux contribution to the load vector
            # For Neumann: integral of q * v ds
            # For linear elements, this distributes to adjacent edges
            if j == 0 or j == ny:
                edge_length = H / (2 * ny)
            else:
                edge_length = H / ny
            
            F[nid] += q * edge_length
    
    # Solve with Dirichlet BCs
    interior_ids = sorted(set(range(n_nodes)) - dirichlet_ids)
    
    if len(interior_ids) == 0:
        print("No interior nodes!")
        return None, None
    
    # Extract submatrix
    K_int = K[np.ix_(interior_ids, interior_ids)]
    F_int = F[interior_ids]
    
    # Solve
    u_int = spsolve(K_int, F_int)
    
    # Full solution
    u = np.zeros(n_nodes)
    for idx, i in enumerate(interior_ids):
        u[i] = u_int[idx]
    
    # Interface temperatures
    interface_temps = [u[nid] for nid in interface_node_ids]
    interface_coords_out = [[x_interface, j * H / ny] for j in range(ny + 1)]
    
    # Compute flux at interface (outward normal is -x for domain B)
    # q = -k * du/dx * n_x = -k * du/dx * (-1) = k * du/dx
    interface_fluxes = []
    for j in range(ny + 1):
        y = j * H / ny
        # Find element adjacent to interface
        # Use finite difference approximation
        x_interior = x_interface + L_B / nx
        # Find node closest to (x_interior, y)
        min_dist = float('inf')
        nearest_val = 0
        for i in range(nx + 1):
            for jj in range(ny + 1):
                xn = x_interface + i * L_B / nx
                yn = jj * H / ny
                dist = (xn - x_interior)**2 + **(yn - y)2
                if dist < min_dist:
                    min_dist = dist
                    nearest_val = u[node_map[(i, jj)] - 1]
        
        du_dx = (nearest_val - u[node_map[(0, j)] - 1]) / (L_B / nx)
        # Outward normal is -x, so q = -k * du/dx * (-1) = k * du/dx
        q = k_B * du_dx
        interface_fluxes.append(q)
    
    n_dof = n_nodes
    
    return interface_coords_out, interface_temps, interface_fluxes, n_dof

def main():
    global_level = level
    global_work_dir = work_dir
    
    # Read imports if available
    imports = read_imports()
    interface_flux_values = None
    interface_coords = None
    
    if imports is not None and 'normal_fluxes' in imports:
        interface_coords = imports['coordinates']
        interface_flux_values = imports['normal_fluxes']
        print(f"Imported {len(interface_flux_values)} interface flux values from A")
    
    # Solve
    result = solve_subdomain(interface_flux_values, interface_coords)
    
    if result is None:
        print("Failed to solve subdomain B")
        sys.exit(1)
    
    coords, temps, fluxes, n_dof = result
    
    # Write log file
    log_file = os.path.join(work_dir, f'run_level{level}_B.log')
    with open(log_file, 'w') as f:
        f.write(f"Kratos solver completed\n")
        f.write(f"Subdomain B: [{x_interface}, {x_right}] x [0, {H}]\n")
        f.write(f"Mesh: {nx} x {ny} elements\n")
        f.write(f"Conductivity: {k_B}\n")
        f.write(f"NDOF = {n_dof}\n")
    
    # Export data
    exports = {
        'field_name': 'temperature',
        'n_points': len(coords),
        'coordinates': coords,
        'values': temps,
        'normal_fluxes': fluxes
    }
    
    # Write exports
    exports_file = os.path.join(work_dir, 'exports.json')
    with open(exports_file, 'w') as f:
        json.dump({'B': exports}, f, indent=2)
    
    print(f"Kratos completed successfully. NDOF = {n_dof}, exported {len(coords)} interface points")

if __name__ == '__main__':
    main()
