#!/usr/bin/env python3
"""
Coupled simulation using 4C (subdomain A) and Kratos (subdomain B)
with Dirichlet-Neumann iteration.

Subdomain A: (0, 0.625) x (0, 1), k = 1 - DIRICHLET side
Subdomain B: (0.625, 1.5) x (0, 1), k = 200 - NEUMANN side
Interface: x = 5/8 = 0.625
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path
import shutil

# Constants
INTERFACE_X = 5.0 / 8.0  # 0.625
L_A = 0.625  # Width of subdomain A
L_B = 0.875  # Width of subdomain B (1.5 - 0.625)
H = 1.0      # Height
K_A = 1.0    # Conductivity in A
K_B = 200.0  # Conductivity in B

# Source term functions
def f_A(x, y):
    """Source term in subdomain A"""
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def f_B(x, y):
    """Source term in subdomain B"""
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Probe points
def get_probe_points_A():
    """Generate probe points for subdomain A"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * L_A / 44
            y = 0 + (i_y + 0.5) * H / 44
            points.append((x, y))
    return points

def get_probe_points_B():
    """Generate probe points for subdomain B"""
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = INTERFACE_X + (i_x + 0.5) * L_B / 44
            y = 0 + (i_y + 0.5) * H / 44
            points.append((x, y))
    return points

def get_interface_probe_points():
    """Generate interface probe points (j = 11 to 32)"""
    points = []
    for j in range(11, 33):
        x = INTERFACE_X
        y = (j + 0.5) / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = get_probe_points_A()
PROBE_POINTS_B = get_probe_points_B()
INTERFACE_PROBE_POINTS = get_interface_probe_points()

# Mesh levels
MESH_LEVELS = [8, 16, 32]  # h = 1/8, 1/16, 1/32

def generate_gmsh_mesh(level, domain, output_dir):
    """Generate mesh using gmsh for a rectangular domain"""
    
    n_elem = level
    
    if domain == 'A':
        x_min, x_max = 0.0, L_A
    else:
        x_min, x_max = INTERFACE_X, 1.5
    
    y_min, y_max = 0.0, H
    
    # Create gmsh script
    mesh_script = f'''
SetFactory("OpenCASCADE");
lc = {min(x_max-x_min, y_max-y_min)/n_elem};

// Points
p1 = {{x_min, y_min, 0, lc}};
p2 = {{x_max, y_min, 0, lc}};
p3 = {{x_max, y_max, 0, lc}};
p4 = {{x_min, y_max, 0, lc}};

// Lines
l1 = {{p1, p2}};  // bottom
l2 = {{p2, p3}};  // right
l3 = {{p3, p4}};  // top
l4 = {{p4, p1}};  // left

// Line loop
ll1 = {{l1, l2, l3, l4}};

// Plane surface
s1 = {{ll1}};

// Physical groups
Physical Line("bottom") = {{l1}};
Physical Line("right") = {{l2}};
Physical Line("top") = {{l3}};
Physical Line("left") = {{l4}};
Physical Surface("domain") = {{s1}};

Mesh 2;
Export "{{output_dir}}/mesh.msh";
'''
    
    # Replace placeholders with actual values
    mesh_script = mesh_script.replace('x_min', str(x_min))
    mesh_script = mesh_script.replace('x_max', str(x_max))
    mesh_script = mesh_script.replace('y_min', str(y_min))
    mesh_script = mesh_script.replace('y_max', str(y_max))
    
    script_file = os.path.join(output_dir, 'mesh.geo')
    with open(script_file, 'w') as f:
        f.write(mesh_script)
    
    # Run gmsh
    cmd = ['gmsh', '-2', '-format', 'msh2', script_file, '-o', os.path.join(output_dir, 'mesh.msh')]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"gmsh error: {result.stderr}")
    
    return os.path.join(output_dir, 'mesh.msh')

def msh_to_cdf(msh_file, cdf_file):
    """Convert gmsh .msh file to 4C CDF format"""
    # This is complex - let's use a simpler approach
    # For now, we'll try to use exodus format which 4C might support
    pass

def main():
    print("=" * 60)
    print("COUPLED SIMULATION: 4C (A) + Kratos (B)")
    print("=" * 60)
    
    all_csv_files = []
    results_by_level = {}
    
    for level_idx, level in enumerate(MESH_LEVELS):
        level_num = level_idx + 1
        print(f"\n{'='*60}")
        print(f"MESH LEVEL {level_num}: h = 1/{level}")
        print(f"{'='*60}")
        
        h = 1.0 / level
        n_elem = level
        
        # Create working directory for this level
        work_dir = Path(f"level_{level}")
        work_dir.mkdir(exist_ok=True)
        
        # Generate meshes for both subdomains
        print("\nGenerating meshes...")
        mesh_a_file = generate_gmsh_mesh(level, 'A', work_dir)
        mesh_b_file = generate_gmsh_mesh(level, 'B', work_dir)
        
        print(f"  Subdomain A mesh: {mesh_a_file}")
        print(f"  Subdomain B mesh: {mesh_b_file}")
        
        # Initialize coupling iteration
        # Start with zero interface values
        n_interface = len(INTERFACE_PROBE_POINTS)
        interface_u_from_B = np.zeros(n_interface)
        interface_qn_from_A = np.zeros(n_interface)
        
        residual_history = []
        max_iterations = 100
        tolerance = 1e-6
        
        print(f"\nStarting Dirichlet-Neumann coupling iteration...")
        print(f"  Interface points: {n_interface}")
        print(f"  Tolerance: {tolerance}")
        
        converged = False
        final_iteration = 0
        
        for iteration in range(max_iterations):
            final_iteration = iteration + 1
            print(f"\n  Coupling iteration {iteration + 1}")
            
            # ===== SUBDOMAIN A (4C) - DIRICHLET SIDE =====
            # Receives u from B, imposes as Dirichlet on interface, returns qn
            
            print(f"    Solving subdomain A with 4C...")
            
            # Create 4C input file
            # We need to handle the interface boundary condition
            # The interface is at x = L_A (right boundary of A)
            
            # For now, let's create a simple test to see if 4C works
            # Then we'll add the coupling logic
            
            # Actually, given the complexity, let me try a different approach:
            # Use NGSolve or FEniCSx internally but call 4C and Kratos as required
            
            # Let me first check if we can make 4C work with a simple example
            pass
            
            # ===== SUBDOMAIN B (Kratos) - NEUMANN SIDE =====
            # Receives qn from A, applies as Neumann on interface, returns u
            
            print(f"    Solving subdomain B with Kratos...")
            
            # Similar approach for Kratos
            
            pass
            
            # Check convergence
            if iteration > 0:
                # Compute relative residual based on interface field
                residual = np.max(np.abs(interface_u_new - interface_u_old)) / (np.max(np.abs(interface_u_old)) + 1e-15)
                residual_history.append(residual)
                print(f"    Residual: {residual:.6e}")
                
                if residual < tolerance:
                    print(f"  Converged after {iteration + 1} iterations")
                    converged = True
                    break
        
        # Save residual history
        residual_file = f"residual_level{level_num}.csv"
        with open(residual_file, 'w') as f:
            f.write("iteration,interface_residual\n")
            for i, res in enumerate(residual_history):
                f.write(f"{i+1},{res}\n")
        all_csv_files.append(residual_file)
        
        results_by_level[level_num] = {
            'residual_history': residual_history,
            'final_residual': residual_history[-1] if residual_history else None,
            'iterations': final_iteration
        }
    
    # Write RESULT.txt
    print("\n" + "=" * 60)
    print("Writing RESULT.txt...")
    print("=" * 60)
    
    # Calculate mesh independence
    # Compare finest two levels
    if len(results_by_level) >= 2:
        # Need to compare solutions - placeholder
        max_rel_change = 0.0  # Will calculate properly
        mesh_independence = "NOT_CONVERGED"  # Placeholder
    else:
        max_rel_change = float('inf')
        mesh_independence = "NOT_CONVERGED"
    
    with open('RESULT.txt', 'w') as f:
        f.write(f"LEVELS = {len(MESH_LEVELS)}\n")
        f.write(f"FILES = {','.join(all_csv_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {results_by_level.get(len(MESH_LEVELS), {}).get('final_residual', 'N/A')}\n")
        f.write(f"COUPLING_ITERATIONS = {results_by_level.get(len(MESH_LEVELS), {}).get('iterations', 0)}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
