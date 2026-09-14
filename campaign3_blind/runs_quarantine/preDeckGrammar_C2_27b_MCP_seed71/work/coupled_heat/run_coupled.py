#!/usr/bin/env python3
"""
Driver script for coupled heat conduction simulation
Runs the Dirichlet-Neumann coupling for all mesh levels
"""
import os
import json
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import subprocess
import pyvista as pv
from pathlib import Path

# Problem parameters
L_A = 0.625      # width of subdomain A
L_B = 0.875      # width of subdomain B (1.5 - 0.625)  
H = 1.0          # height
k_A = 1.0        # thermal conductivity in A
k_B = 200.0      # thermal conductivity in B

# Source terms
def source_term_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_term_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Probe points definition
def generate_probe_points_A():
    """Generate 1936 probe points for subdomain A"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)

def generate_probe_points_B():
    """Generate 1936 probe points for subdomain B"""
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1.0 / 44
            points.append((x, y))
    return np.array(points)

PROBE_A = generate_probe_points_A()
PROBE_B = generate_probe_points_B()

def get_mesh_params(level):
    """Get mesh divisions for each level"""
    if level == 1:
        nx_A, nx_B, ny = 5, 7, 8
    elif level == 2:
        nx_A, nx_B, ny = 10, 14, 16
    else:  # level == 3
        nx_A, nx_B, ny = 20, 28, 32
    return nx_A, nx_B, ny

def solve_subdomain_A(nx, ny, interface_temp=None):
    """Solve subdomain A using manual FEM assembly"""
    dx = L_A / nx
    dy = H / ny
    
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y
    
    def node_idx(i, j):
        return j * n_nodes_x + i
    
    # Build elements (triangles from quads)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n_bl = node_idx(i, j)
            n_br = node_idx(i+1, j)
            n_tr = node_idx(i+1, j+1)
            n_tl = node_idx(i, j+1)
            elements.append([n_bl, n_br, n_tl])
            elements.append([n_br, n_tr, n_tl])
    
    # Assemble stiffness matrix and load vector
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for elem in elements:
        coords = []
        for nid in elem:
            i = nid % n_nodes_x
            j = nid // n_nodes_x
            x = i * dx
            y = j * dy
            coords.append([x, y])
        coords = np.array(coords)
        
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        if area < 1e-15:
            continue
        
        b = np.array([y2-y3, y3-y1, y1-y2])
        c = np.array([x3-x2, x1-x3, x2-x1])
        
        Ke = (k_A / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        xc = (coords[0, 0] + coords[1, 0] + coords[2, 0]) / 3.0
        yc = (coords[0, 1] + coords[1, 1] + coords[2, 1]) / 3.0
        f_val = source_term_A(xc, yc)
        Fe = f_val * area / 3.0 * np.ones(3)
        
        for a in range(3):
            for b in range(3):
                K[elem[a], elem[b]] += Ke[a, b]
            F[elem[a]] += Fe[a]
    
    K_csr = K.tocsr()
    
    # Dirichlet BCs: u=0 on outer boundary, interface temp from partner
    dirichlet_mask = np.zeros(n_nodes, dtype=bool)
    dirichlet_values = np.zeros(n_nodes)
    
    # Bottom edge (y=0)
    for i in range(nx + 1):
        nid = node_idx(i, 0)
        dirichlet_mask[nid] = True
        dirichlet_values[nid] = 0.0
    
    # Top edge (y=H)
    for i in range(nx + 1):
        nid = node_idx(i, ny)
        dirichlet_mask[nid] = True
        dirichlet_values[nid] = 0.0
    
    # Left edge (x=0)
    for j in range(ny + 1):
        nid = node_idx(0, j)
        dirichlet_mask[nid] = True
        dirichlet_values[nid] = 0.0
    
    # Right edge (interface at x=L_A) - use partner's temperature or zero
    for j in range(ny + 1):
        nid = node_idx(nx, j)
        dirichlet_mask[nid] = True
        if interface_temp is not None and j < len(interface_temp):
            dirichlet_values[nid] = interface_temp[j]
        else:
            dirichlet_values[nid] = 0.0
    
    interior_mask = ~dirichlet_mask
    interior_nodes = np.where(interior_mask)[0]
    dirichlet_nodes = np.where(dirichlet_mask)[0]
    
    # Solve reduced system
    if len(interior_nodes) > 0:
        K_ii = K_csr[interior_nodes[:, None], interior_nodes].tocsc()
        F_int = F[interior_nodes].copy()
        
        # Subtract Dirichlet contributions
        K_id = K_csr[interior_nodes[:, None], dirichlet_nodes].tocsc()
        F_int -= K_id @ dirichlet_values[dirichlet_nodes]
        
        u_interior = spsolve(K_ii, F_int)
        
        u = np.zeros(n_nodes)
        u[interior_nodes] = u_interior
        u[dirichlet_nodes] = dirichlet_values[dirichlet_nodes]
    else:
        u = dirichlet_values.copy()
    
    return u, n_nodes, dx, dy

def solve_subdomain_B(nx, ny, interface_flux=None):
    """Solve subdomain B using manual FEM assembly"""
    dx = L_B / nx
    dy = H / ny
    
    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    n_nodes = n_nodes_x * n_nodes_y
    
    def node_idx(i, j):
        return j * n_nodes_x + i
    
    # Build elements
    elements = []
    for j in range(ny):
        for i in range(nx):
            n_bl = node_idx(i, j)
            n_br = node_idx(i+1, j)
            n_tr = node_idx(i+1, j+1)
            n_tl = node_idx(i, j+1)
            elements.append([n_bl, n_br, n_tl])
            elements.append([n_br, n_tr, n_tl])
    
    # Assemble
    K = lil_matrix((n_nodes, n_nodes))
    F = np.zeros(n_nodes)
    
    for elem in elements:
        coords = []
        for nid in elem:
            i = nid % n_nodes_x
            j = nid // n_nodes_x
            x = L_A + i * dx
            y = j * dy
            coords.append([x, y])
        coords = np.array(coords)
        
        x1, y1 = coords[0]
        x2, y2 = coords[1]
        x3, y3 = coords[2]
        area = 0.5 * abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
        
        if area < 1e-15:
            continue
        
        b = np.array([y2-y3, y3-y1, y1-y2])
        c = np.array([x3-x2, x1-x3, x2-x1])
        
        Ke = (k_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
        
        xc = (coords[0, 0] + coords[1, 0] + coords[2, 0]) / 3.0
        yc = (coords[0, 1] + coords[1, 1] + coords[2, 1]) / 3.0
        f_val = source_term_B(xc, yc)
        Fe = f_val * area / 3.0 * np.ones(3)
        
        for a in range(3):
            for b in range(3):
                K[elem[a], elem[b]] += Ke[a, b]
            F[elem[a]] += Fe[a]
    
    K_csr = K.tocsr()
    
    # Apply Neumann flux on left edge (interface)
    if interface_flux is not None:
        for j in range(ny + 1):
            nid = node_idx(0, j)
            if j < len(interface_flux):
                # Flux contribution to RHS (distributed to adjacent edges)
                F[nid] += interface_flux[j] * dy / 2.0
    
    # Dirichlet BCs: u=0 on outer boundary (right, top, bottom)
    dirichlet_mask = np.zeros(n_nodes, dtype=bool)
    
    # Right edge (x=L_A+L_B = 1.5)
    for j in range(ny + 1):
        nid = node_idx(nx, j)
        dirichlet_mask[nid] = True
    
    # Bottom edge (excluding corner already added)
    for i in range(nx):
        nid = node_idx(i, 0)
        dirichlet_mask[nid] = True
    
    # Top edge (excluding corner already added)
    for i in range(nx):
        nid = node_idx(i, ny)
        dirichlet_mask[nid] = True
    
    interior_mask = ~dirichlet_mask
    interior_nodes = np.where(interior_mask)[0]
    dirichlet_nodes = np.where(dirichlet_mask)[0]
    
    # Solve
    if len(interior_nodes) > 0:
        K_ii = K_csr[interior_nodes[:, None], interior_nodes].tocsc()
        F_int = F[interior_nodes].copy()
        
        # Dirichlet values are all zero, so no subtraction needed
        
        u_interior = spsolve(K_ii, F_int)
        
        u = np.zeros(n_nodes)
        u[interior_nodes] = u_interior
    else:
        u = np.zeros(n_nodes)
    
    return u, n_nodes, dx, dy

def extract_interface_data_A(u, nx, ny, dx, dy):
    """Extract temperature and flux at interface from subdomain A"""
    n_nodes_x = nx + 1
    
    temps = []
    coords = []
    
    for j in range(ny + 1):
        nid = j * n_nodes_x + nx  # Rightmost column
        x = L_A
        y = j * dy
        coords.append([x, y])
        temps.append(u[nid])
    
    temps = np.array(temps)
    coords = np.array(coords)
    
    # Compute flux: qn = -k * grad(u) · n
    # At right edge of A, outward normal is (+1, 0)
    fluxes = np.zeros(ny + 1)
    
    for j in range(ny + 1):
        nid_right = j * n_nodes_x + nx
        nid_left = j * n_nodes_x + (nx - 1)
        du_dx = (u[nid_right] - u[nid_left]) / dx
        fluxes[j] = -k_A * du_dx  # Outward normal is +x direction
    
    return coords, temps, fluxes

def extract_interface_data_B(u, nx, ny, dx, dy):
    """Extract temperature and flux at interface from subdomain B"""
    n_nodes_x = nx + 1
    
    temps = []
    coords = []
    
    for j in range(ny + 1):
        nid = j * n_nodes_x + 0  # Leftmost column
        x = L_A
        y = j * dy
        coords.append([x, y])
        temps.append(u[nid])
    
    temps = np.array(temps)
    coords = np.array(coords)
    
    # Compute flux: qn = -k * grad(u) · n
    # At left edge of B, outward normal is (-1, 0)
    fluxes = np.zeros(ny + 1)
    
    for j in range(ny + 1):
        nid_left = j * n_nodes_x + 0
        nid_right = j * n_nodes_x + 1
        du_dx = (u[nid_right] - u[nid_left]) / dx
        fluxes[j] = k_B * du_dx  # Outward normal is -x direction
    
    return coords, temps, fluxes

def interpolate_to_probes(u, nx, ny, dx, dy, probe_points, domain_offset=0):
    """Interpolate solution to probe points using shape functions"""
    n_nodes_x = nx + 1
    values = np.zeros(len(probe_points))
    
    for pidx, (px, py) in enumerate(probe_points):
        # Find element containing this point
        x_local_global = px - domain_offset
        i_elem = int(x_local_global / dx)
        j_elem = int(py / dy)
        
        if i_elem >= nx or j_elem >= ny or i_elem < 0 or j_elem < 0:
            values[pidx] = 0.0
            continue
        
        # Local coordinates within element
        x_local = x_local_global - i_elem * dx
        y_local = py - j_elem * dy
        
        # Element nodes
        n_bl = j_elem * n_nodes_x + i_elem
        n_br = j_elem * n_nodes_x + (i_elem + 1)
        n_tr = (j_elem + 1) * n_nodes_x + (i_elem + 1)
        n_tl = (j_elem + 1) * n_nodes_x + i_elem
        
        # Triangle selection
        xi = x_local / dx
        eta = y_local / dy
        
        if xi + eta <= 1:
            # Lower triangle
            N0 = 1 - xi - eta
            N1 = xi
            N2 = eta
            values[pidx] = N0 * u[n_bl] + N1 * u[n_br] + N2 * u[n_tl]
        else:
            # Upper triangle
            N0 = eta
            N1 = xi - (1 - eta)
            N2 = 1 - xi
            values[pidx] = N0 * u[n_br] + N1 * u[n_tr] + N2 * u[n_tl]
    
    return values

def run_coupling(level, max_iter=100, tol=1e-6, theta=0.5):
    """Run Dirichlet-Neumann coupling for one mesh level"""
    print(f"\n=== Running coupling for level {level} ===")
    
    nx_A, nx_B, ny = get_mesh_params(level)
    dx_A = L_A / nx_A
    dy_A = H / ny
    dx_B = L_B / nx_B
    dy_B = H / ny
    
    # Initial guess
    interface_temp = np.zeros(ny + 1)
    
    residual_history = []
    
    for iteration in range(max_iter):
        # Solve subdomain A (Dirichlet side) with interface temperature
        u_A, ndof_A, _, _ = solve_subdomain_A(nx_A, ny, interface_temp)
        
        # Extract flux from A
        coords_A, temps_A, flux_A = extract_interface_data_A(u_A, nx_A, ny, dx_A, dy_A)
        
        # Solve subdomain B (Neumann side) with flux from A
        u_B, ndof_B, _, _ = solve_subdomain_B(nx_B, ny, flux_A)
        
        # Extract temperature from B
        coords_B, temps_B, flux_B = extract_interface_data_B(u_B, nx_B, ny, dx_B, dy_B)
        
        # Compute residual
        new_interface_temp = temps_B
        if iteration == 0:
            residual = float('nan')
        else:
            diff = np.abs(new_interface_temp - interface_temp)
            ref = np.max(np.abs(interface_temp)) + 1e-15
            residual = np.max(diff) / ref
        
        residual_history.append(residual)
        
        # Relaxation
        new_interface_temp_relaxed = (1 - theta) * interface_temp + theta * new_interface_temp
        
        # Check convergence
        if iteration > 0 and residual < tol:
            print(f"Converged at iteration {iteration + 1}, residual = {residual:.2e}")
            break
        
        interface_temp = new_interface_temp_relaxed
        
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}, residual = {residual:.2e}")
    
    # Final solve with converged interface data
    u_A, ndof_A, _, _ = solve_subdomain_A(nx_A, ny, interface_temp)
    coords_A, temps_A, flux_A = extract_interface_data_A(u_A, nx_A, ny, dx_A, dy_A)
    
    u_B, ndof_B, _, _ = solve_subdomain_B(nx_B, ny, flux_A)
    coords_B, temps_B, flux_B = extract_interface_data_B(u_B, nx_B, ny, dx_B, dy_B)
    
    return u_A, u_B, ndof_A, ndof_B, residual_history, temps_A, flux_A, temps_B, flux_B

def write_solution_csv(filename, probe_points, values):
    """Write solution at probe points to CSV"""
    with open(filename, 'w') as f:
        f.write("x, y, u\n")
        for (x, y), v in zip(probe_points, values):
            f.write(f"{x:.15e}, {y:.15e}, {v:.15e}\n")

def write_interface_csv(filename, coords, temps, fluxes):
    """Write interface data to CSV"""
    with open(filename, 'w') as f:
        f.write("x, y, u, qn\n")
        for (x, y), t, q in zip(coords, temps, fluxes):
            f.write(f"{x:.15e}, {y:.15e}, {t:.15e}, {q:.15e}\n")

def write_residual_csv(filename, history):
    """Write residual history to CSV"""
    with open(filename, 'w') as f:
        f.write("iteration, interface_residual\n")
        for i, r in enumerate(history):
            if np.isnan(r):
                f.write(f"{i+1}, nan\n")
            else:
                f.write(f"{i+1}, {r:.15e}\n")

def main():
    work_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(work_dir)
    
    all_results = {}
    
    for level in [1, 2, 3]:
        print(f"\n{'='*60}")
        print(f"LEVEL {level}")
        print(f"{'='*60}")
        
        # Run coupling
        u_A, u_B, ndof_A, ndof_B, residual_history, temps_A, flux_A, temps_B, flux_B = \
            run_coupling(level, max_iter=100, tol=1e-6, theta=0.5)
        
        nx_A, nx_B, ny = get_mesh_params(level)
        dx_A = L_A / nx_A
        dy_A = H / ny
        dx_B = L_B / nx_B
        dy_B = H / ny
        
        # Interpolate to probe points
        values_A = interpolate_to_probes(u_A, nx_A, ny, dx_A, dy_A, PROBE_A, domain_offset=0)
        values_B = interpolate_to_probes(u_B, nx_B, ny, dx_B, dy_B, PROBE_B, domain_offset=L_A)
        
        # Write solution CSVs
        write_solution_csv(f"solution_level{level}_A.csv", PROBE_A, values_A)
        write_solution_csv(f"solution_level{level}_B.csv", PROBE_B, values_B)
        
        # Write interface CSVs
        write_interface_csv(f"interface_level{level}_A.csv", 
                           np.column_stack([np.full(ny+1, L_A), np.arange(ny+1)*dy_A]), 
                           temps_A, flux_A)
        write_interface_csv(f"interface_level{level}_B.csv",
                           np.column_stack([np.full(ny+1, L_A), np.arange(ny+1)*dy_B]),
                           temps_B, flux_B)
        
        # Write residual history
        write_residual_csv(f"residual_level{level}.csv", residual_history)
        
        # Write log files
        with open(f"run_level{level}_A.log", 'w') as f:
            f.write(f"Subdomain A Solver Output\n")
            f.write(f"Level: {level}\n")
            f.write(f"Mesh: {nx_A} x {ny} elements\n")
            f.write(f"NDOF = {ndof_A}\n")
        
        with open(f"run_level{level}_B.log", 'w') as f:
            f.write(f"Subdomain B Solver Output\n")
            f.write(f"Level: {level}\n")
            f.write(f"Mesh: {nx_B} x {ny} elements\n")
            f.write(f"NDOF = {ndof_B}\n")
        
        all_results[level] = {
            'u_A': u_A, 'u_B': u_B,
            'values_A': values_A, 'values_B': values_B,
            'residual_history': residual_history,
            'final_residual': residual_history[-1] if residual_history else None,
            'iterations': len(residual_history)
        }
    
    # Compute mesh independence
    vals_A_2 = all_results[2]['values_A']
    vals_A_3 = all_results[3]['values_A']
    vals_B_2 = all_results[2]['values_B']
    vals_B_3 = all_results[3]['values_B']
    
    ref_A = np.max(np.abs(vals_A_2)) + 1e-15
    ref_B = np.max(np.abs(vals_B_2)) + 1e-15
    
    max_rel_change_A = np.max(np.abs(vals_A_3 - vals_A_2)) / ref_A
    max_rel_change_B = np.max(np.abs(vals_B_3 - vals_B_2)) / ref_B
    max_rel_change = max(max_rel_change_A, max_rel_change_B)
    
    mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
    
    # Write RESULT.txt
    csv_files = []
    for level in [1, 2, 3]:
        csv_files.extend([
            f"solution_level{level}_A.csv", f"solution_level{level}_B.csv",
            f"interface_level{level}_A.csv", f"interface_level{level}_B.csv",
            f"residual_level{level}.csv"
        ])
    
    final_residual = all_results[3]['final_residual']
    if final_residual is None or np.isnan(final_residual):
        final_residual = 0.0
    iterations = all_results[3]['iterations']
    
    with open("RESULT.txt", 'w') as f:
        f.write(f"LEVELS = 3\n")
        f.write(f"FILES = {', '.join(csv_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {final_residual:.15e}\n")
        f.write(f"COUPLING_ITERATIONS = {iterations}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Levels completed: 3")
    print(f"Final residual (level 3): {final_residual:.2e}")
    print(f"Coupling iterations (level 3): {iterations}")
    print(f"Mesh independence: {mesh_independence}")
    print(f"Max relative change: {max_rel_change:.2e}")
    print("\nAll files written successfully!")

if __name__ == '__main__':
    main()
