#!/usr/bin/env python3
"""
Participant B (Kratos manual assembly) - Subdomain B: (0.625, 1.5) x (0, 1), k=200
NEUMANN side: receives flux from A, applies as Neumann BC on interface, returns field
Interface at x = 0.625, outward normal points -x (left)
"""
import json
import os
import sys
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import meshio

# Problem parameters
L_B = 0.875   # width of subdomain B (1.5 - 0.625)
H = 1.0       # height
x_start = 0.625  # left edge of B
k_B = 200.0   # thermal conductivity in B
interface_x = 0.625

def source_B(x, y):
    """Source term in subdomain B"""
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def get_mesh_level():
    """Determine mesh level from work_dir name"""
    work_dir = os.getcwd()
    if 'level1' in work_dir:
        return 1
    elif 'level2' in work_dir:
        return 2
    elif 'level3' in work_dir:
        return 3
    else:
        raise ValueError(f"Cannot determine mesh level from {work_dir}")

def assemble_and_solve(nx, ny, interface_fluxes=None):
    """Assemble FEM system and solve for subdomain B"""
    
    # Generate uniform mesh
    coords = {}
    node_map = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = x_start + i * L_B / nx
            y = j * H / ny
            coords[nid] = (x, y)
            node_map[(i, j)] = nid
            nid += 1
    
    n_nodes = nid - 1
    
    # Generate triangular elements (P1)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n_bl = node_map[(i, j)]
            n_br = node_map[(i+1, j)]
            n_tr = node_map[(i+1, j+1)]
            n_tl = node_map[(i, j+1)]
            # Two triangles per quad
            elements.append((n_bl, n_br, n_tl))
            elements.append((n_br, n_tr, n_tl))
    
    n_elements = len(elements)
    
    # Assemble stiffness matrix with conductivity k_B
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for tri in elements:
        ids = [t-1 for t in tri]  # 0-indexed
        x = np.array([coords[t][0] for t in tri])
        y = np.array([coords[t][1] for t in tri])
        
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        
        # Shape function derivatives
        b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
        c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
        
        # Element stiffness matrix with conductivity
        Ke = (k_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        for a in range(3):
            for bb in range(3):
                K[ids[a], ids[bb]] += Ke[a, bb]
        
        # Element force vector (source term integrated over element)
        # Using midpoint quadrature for linear elements
        x_mid = np.mean(x)
        y_mid = np.mean(y)
        f_val = source_B(x_mid, y_mid)
        
        # Integral of shape function over triangle = area/3
        for a in range(3):
            F[ids[a]] += f_val * area / 3.0
    
    K = K.tocsr()
    
    # Identify boundary nodes
    # Left boundary (x=0.625, interface) - exclude corners
    left_interface = set()
    for j in range(1, ny):
        left_interface.add(node_map[(0, j)] - 1)
    
    # Right boundary (x=1.5) - Dirichlet u=0
    right = set()
    for j in range(ny + 1):
        right.add(node_map[(nx, j)] - 1)
    
    # Bottom boundary (y=0) - Dirichlet u=0
    bottom = set()
    for i in range(nx + 1):
        bottom.add(node_map[(i, 0)] - 1)
    
    # Top boundary (y=1) - Dirichlet u=0
    top = set()
    for i in range(nx + 1):
        top.add(node_map[(i, ny)] - 1)
    
    dirichlet_nodes = right | bottom | top
    interior_nodes = sorted(set(range(n_nodes)) - dirichlet_nodes)
    
    # Apply Neumann BC on interface if provided
    if interface_fluxes is not None:
        # Interface flux from partner (A's outward flux)
        # Our outward normal is -x, so we need to apply the flux correctly
        # The partner exports qn = -k_A * du/dx|_A (their outward normal is +x)
        # We receive this and apply it as traction on our interface
        # For heat: -k_B * grad(u) . n = q_n_received
        # With n = (-1, 0), this gives k_B * du/dx = q_n_received
        
        # Map fluxes to interface nodes
        # Our interface nodes are at y = j/ny for j=1..ny-1
        dx = L_B / nx
        
        for idx, j in enumerate(range(1, ny)):
            node_idx = node_map[(0, j)] - 1
            if node_idx in left_interface:
                # Get corresponding flux value
                if idx < len(interface_fluxes):
                    qn_received = interface_fluxes[idx]
                    # Neumann contribution to RHS
                    # For linear elements, distribute to adjacent elements
                    # Simplified: add to the node directly
                    F[node_idx] += qn_received * (H / ny)  # approximate edge length
    
    # Solve system with Dirichlet BCs
    u = np.zeros(n_nodes)
    
    # Extract interior system
    interior_set = set(interior_nodes)
    K_int = K[np.ix_(interior_nodes, interior_nodes)]
    F_int = F[interior_nodes]
    
    # Account for Dirichlet nodes (all zero in this case)
    # Since u=0 on all Dirichlet boundaries, no modification needed
    
    u[interior_nodes] = spsolve(K_int, F_int)
    
    return u, coords, node_map, n_nodes, n_elements, left_interface

def main():
    work_dir = os.getcwd()
    print(f"Working directory: {work_dir}")
    sys.stdout.flush()
    
    level = get_mesh_level()
    divisions = [8, 16, 32][level - 1]
    nx = int(L_B * divisions)  # 0.875 * divisions
    ny = int(H * divisions)
    
    print(f"Level {level}: nx={nx}, ny={ny}")
    sys.stdout.flush()
    
    # Read imports
    imports_path = os.path.join(work_dir, 'imports.json')
    interface_fluxes = None
    if os.path.exists(imports_path):
        with open(imports_path, 'r') as f:
            imports_data = json.load(f)
        if 'A' in imports_data:
            partner_data = imports_data['A']
            interface_fluxes = partner_data.get('normal_fluxes', None)
        print(f"Read imports from {imports_path}")
    else:
        print("No imports found - first iteration")
    sys.stdout.flush()
    
    # Solve
    u, coords, node_map, n_nodes, n_elements, left_interface = assemble_and_solve(nx, ny, interface_fluxes)
    
    print(f"Solved: max u = {u.max():.6e}, min u = {u.min():.6e}")
    sys.stdout.flush()
    
    # Write solution to VTU
    pts = np.array([[coords[i+1][0], coords[i+1][1], 0.0] for i in range(n_nodes)])
    cells = []
    for tri in [(node_map[(i,j)], node_map[(i+1,j)], node_map[(i,j+1)]) 
                for j in range(ny) for i in range(nx)] + \
               [(node_map[(i+1,j)], node_map[(i+1,j+1)], node_map[(i,j+1)]) 
                for j in range(ny) for i in range(nx)]:
        cells.append([t-1 for t in tri])
    
    mesh = meshio.Mesh(pts, [("triangle", np.array(cells))], point_data={"temperature": u})
    vtu_file = os.path.join(work_dir, "solution_B.vtu")
    mesh.write(vtu_file)
    print(f"Wrote solution to {vtu_file}")
    sys.stdout.flush()
    
    # Export interface data
    # Interface nodes are at x = 0.625, y = j/ny for j = 1..ny-1
    interface_coords = []
    interface_values = []
    interface_fluxes_out = []
    
    dx = L_B / nx
    
    for j in range(1, ny):
        y_j = j * H / ny
        x_interface = x_start
        
        node_idx = node_map[(0, j)] - 1
        neighbor_idx = node_map[(1, j)] - 1
        
        interface_coords.append([x_interface, y_j])
        interface_values.append(u[node_idx])
        
        # Compute outward normal flux: qn = -k * grad(u) . n
        # Outward normal for B is -x, so qn = -k_B * (-du/dx) = k_B * du/dx
        du_dx = (u[neighbor_idx] - u[node_idx]) / dx
        qn = k_B * du_dx  # outward normal is -x
        interface_fluxes_out.append(qn)
    
    # Write exports
    exports = {
        'field_name': 'temperature',
        'coordinates': interface_coords,
        'values': interface_values,
        'normal_fluxes': interface_fluxes_out,
        'n_points': len(interface_coords)
    }
    
    exports_path = os.path.join(work_dir, 'exports.json')
    with open(exports_path, 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Wrote exports to {exports_path}")
    print(f"NDOF = {n_nodes}")
    sys.stdout.flush()

if __name__ == '__main__':
    main()
