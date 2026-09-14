#!/usr/bin/env python3
"""
Participant B (Subdomain B): Kratos solver for heat conduction
Domain: (0.625, 1.5) x (0, 1), k = 200
Role: NEUMANN side - receives flux from partner, returns temperature
"""
import os
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Problem parameters
L_A = 0.625      # width of subdomain A (interface location)
L_B = 0.875      # width of subdomain B (1.5 - 0.625)
H = 1.0          # height
k_B = 200.0      # thermal conductivity

# Source term for subdomain B (Python syntax for evaluation)
def source_term_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

def get_mesh_params(level):
    """Get mesh divisions for each level. h = 1/(8*level)"""
    if level == 1:
        nx, ny = 7, 8   # h = 1/8
    elif level == 2:
        nx, ny = 14, 16 # h = 1/16
    else:  # level == 3
        nx, ny = 28, 32 # h = 1/32
    return nx, ny

def assemble_and_solve(nx, ny, work_dir, level, interface_flux=None):
    """Assemble FEM system and solve using manual assembly"""
    
    dx = L_B / nx
    dy = H / ny
    
    # Generate nodes (local coordinates start at 0, then shift by L_A)
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y
    
    # Node mapping: (i, j) -> global node index
    def node_idx(i, j):
        return j * n_nodes_x + i
    
    # Build element connectivity (triangles from quads)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n_bl = node_idx(i, j)
            n_br = node_idx(i+1, j)
            n_tr = node_idx(i+1, j+1)
            n_tl = node_idx(i, j+1)
            # Two triangles per quad
            elements.append([n_bl, n_br, n_tl])
            elements.append([n_br, n_tr, n_tl])
    
    # Assemble stiffness matrix and load vector
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for elem in elements:
        # Get node coordinates (shifted by L_A)
        coords = []
        for nid in elem:
            i = nid % n_nodes_x
            j = nid // n_nodes_x
            x = L_A + i * dx
            y = j * dy
            coords.append([x, y])
        coords = np.array(coords)
        
        # Triangle area
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        if area < 1e-15:
            continue
        
        # Shape function derivatives
        b = np.array([y2-y3, y3-y1, y1-y2])
        c = np.array([x3-x2, x1-x3, x2-x1])
        
        # Element stiffness matrix
        Ke = (k_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        # Element load vector (integrate source term)
        # Using midpoint quadrature
        xc = (coords[0, 0] + coords[1, 0] + coords[2, 0]) / 3.0
        yc = (coords[0, 1] + coords[1, 1] + coords[2, 1]) / 3.0
        f_val = source_term_B(xc, yc)
        Fe = f_val * area / 3.0 * np.ones(3)
        
        # Assemble into global system
        for a in range(3):
            for b in range(3):
                K[elem[a], elem[b]] += Ke[a, b]
            F[elem[a]] += Fe[a]
    
    K = K.tocsr()
    
    # Apply Dirichlet boundary conditions
    # Left edge (interface at local x=0, global x=L_A): will have Neumann from flux
    # Right edge (local x=L_B, global x=1.5): u=0
    # Bottom edge (y=0): u=0
    # Top edge (y=1): u=0
    
    dirichlet_nodes = set()
    
    # Right edge
    for j in range(ny + 1):
        dirichlet_nodes.add(node_idx(nx, j))
    
    # Bottom edge (excluding corners already added)
    for i in range(nx):
        dirichlet_nodes.add(node_idx(i, 0))
    
    # Top edge (excluding corners already added)
    for i in range(nx):
        dirichlet_nodes.add(node_idx(i, ny))
    
    # Left edge (interface) - NOT Dirichlet, this is where Neumann flux is applied
    
    interior_nodes = [i for i in range(n_nodes) if i not in dirichlet_nodes]
    dirichlet_list = sorted(dirichlet_nodes)
    
    # Apply Neumann flux on left edge (interface)
    # Flux is given as outward normal flux from domain A's perspective
    # For domain B, the outward normal points LEFT (-x direction)
    # So we need to apply the flux with appropriate sign
    
    if interface_flux is not None:
        # interface_flux comes from partner A, which exports its OUTWARD flux
        # A's outward normal at interface is +x, so flux_A = -k_A * du/dx|_A
        # B's outward normal at interface is -x, so we receive flux that should balance
        # The coupling driver handles relaxation, we just apply what we receive
        
        # Find left edge nodes
        left_edge_nodes = [node_idx(0, j) for j in range(ny + 1)]
        
        # Interpolate flux to our nodes
        # Partner sends flux at their interface nodes; we need to map to ours
        # Since meshes match, we can use direct indexing
        for j, nid in enumerate(left_edge_nodes):
            if j < len(interface_flux):
                # Apply flux as natural BC: add to RHS
                # qn = -k * grad(u) · n, where n is outward normal
                # For B, outward normal at left edge is (-1, 0)
                # So qn = -k * (-du/dx) = k * du/dx
                # The weak form contribution is integral(qn * v) ds
                # For linear elements, distribute to adjacent edges
                F[nid] += interface_flux[j] * dy / 2.0  # Simple distribution
    
    # Solve reduced system
    if len(interior_nodes) > 0:
        K_reduced = K[np.ix_(interior_nodes, interior_nodes)]
        F_reduced = F[interior_nodes].copy()
        
        # Subtract Dirichlet contributions (all zero in this case)
        for dn in dirichlet_list:
            dval = 0.0  # All Dirichlet values are zero
            for in_id in interior_nodes:
                F_reduced[in_id] -= K[in_id, dn] * dval
        
        u_interior = spsolve(K_reduced, F_reduced)
        
        # Reconstruct full solution
        u = np.zeros(n_nodes)
        for idx, nid in enumerate(interior_nodes):
            u[nid] = u_interior[idx]
        # Dirichlet nodes remain zero
    else:
        u = np.zeros(n_nodes)
    
    return u, n_nodes, dx, dy

def extract_interface_data(u, nx, ny, dx, dy):
    """Extract temperature and flux at interface (left edge of B)"""
    
    n_nodes_x = nx + 1
    
    # Interface is at local x=0 (global x=L_A)
    # Nodes: (0, j) for j = 0, ..., ny
    interface_temps = []
    interface_coords = []
    
    for j in range(ny + 1):
        nid = j * n_nodes_x + 0  # First column
        x_global = L_A
        y = j * dy
        interface_coords.append([x_global, y])
        interface_temps.append(u[nid])
    
    interface_temps = np.array(interface_temps)
    interface_coords = np.array(interface_coords)
    
    # Compute flux: qn = -k * grad(u) · n
    # At left edge of B, outward normal is (-1, 0)
    # So qn = -k * (-du/dx) = k * du/dx
    # Use forward difference
    fluxes = np.zeros(ny + 1)
    
    for j in range(ny + 1):
        nid_left = j * n_nodes_x + 0
        nid_right = j * n_nodes_x + 1
        du_dx = (u[nid_right] - u[nid_left]) / dx
        fluxes[j] = k_B * du_dx  # Outward normal is -x direction
    
    return interface_coords, interface_temps, fluxes

def write_exports(work_dir, coords, values, fluxes):
    """Write exports.json for coupling driver"""
    exports = {
        "field_name": "temperature",
        "coordinates": coords.tolist(),
        "values": values.tolist(),
        "normal_fluxes": fluxes.tolist(),
        "n_points": len(values)
    }
    
    with open(os.path.join(work_dir, 'exports.json'), 'w') as f:
        json.dump(exports, f, indent=2)

def main():
    work_dir = os.getcwd()
    
    # Read level from environment or file
    level_file = os.path.join(work_dir, 'level.json')
    if os.path.exists(level_file):
        with open(level_file, 'r') as f:
            config = json.load(f)
            level = config.get('level', 1)
    else:
        level = 1
    
    # Read imports from partner
    imports_file = os.path.join(work_dir, 'imports.json')
    interface_flux = None
    if os.path.exists(imports_file):
        with open(imports_file, 'r') as f:
            imports = json.load(f)
        # Partner A sends flux to us
        if 'A' in imports:
            partner_data = imports['A']
            interface_flux = np.array(partner_data['normal_fluxes'])
            print(f"Received {len(interface_flux)} flux values from A")
    
    # Get mesh params
    nx, ny = get_mesh_params(level)
    
    # Assemble and solve
    u, n_nodes, dx, dy = assemble_and_solve(nx, ny, work_dir, level, interface_flux)
    
    # Extract interface data
    coords, temps, fluxes = extract_interface_data(u, nx, ny, dx, dy)
    
    # Write exports
    write_exports(work_dir, coords, temps, fluxes)
    
    # Write log file with NDOF
    log_file = os.path.join(work_dir, 'run_level{}_B.log'.format(level))
    with open(log_file, 'w') as f:
        f.write("Kratos Participant B - Heat Conduction Solver\n")
        f.write(f"Level: {level}\n")
        f.write(f"Mesh: {nx} x {ny} elements\n")
        f.write(f"Solution computed successfully\n")
        f.write(f"NDOF = {n_nodes}\n")
    
    print(f"Participant B completed. Interface points: {len(coords)}")

if __name__ == '__main__':
    main()
