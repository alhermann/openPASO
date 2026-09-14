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
import shutil
from pathlib import Path

# Constants
INTERFACE_X = 5.0 / 8.0  # 0.625
L_A = 0.625  # Width of subdomain A
L_B = 0.875  # Width of subdomain B (1.5 - 0.625)
H = 1.0      # Height

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

def generate_4c_input(level, input_file, output_prefix, k_value, source_func_name, 
                      interface_dirichlet_data=None, interface_neumann_data=None):
    """Generate 4C YAML input file for heat equation"""
    
    n_elem = level  # Number of elements in each direction
    
    # For subdomain A: (0, 0.625) x (0, 1)
    # For subdomain B: (0.625, 1.5) x (0, 1)
    
    yaml_content = f'''
problem:
  type: poisson
  
geometry:
  type: rectangle
  x_min: {0 if 'A' in input_file else INTERFACE_X}
  x_max: {L_A if 'A' in input_file else 1.5}
  y_min: 0
  y_max: {H}
  
mesh:
  type: structured
  nx: {n_elem}
  ny: {n_elem}
  
material:
  conductivity: {k_value}
  
source:
  type: expression
  expression: "{source_func_name}"
  
boundary_conditions:
  - name: bottom
    type: dirichlet
    value: 0
    condition: "y == 0"
    
  - name: top
    type: dirichlet
    value: 0
    condition: "y == {H}"
    
  - name: left
    type: dirichlet
    value: 0
    condition: "x == {0 if 'A' in input_file else INTERFACE_X}"
    
  - name: right
    type: dirichlet
    value: 0
    condition: "x == {L_A if 'A' in input_file else 1.5}"
    
output:
  format: vtu
  fields: [u]
'''
    
    with open(input_file, 'w') as f:
        f.write(yaml_content)

def run_4c(input_file, output_prefix, log_file):
    """Run 4C solver"""
    cmd = ['LD_LIBRARY_PATH=/opt/4C-dependencies/lib', '/home/alexander/4C/build/4C', 
           input_file, output_prefix]
    
    with open(log_file, 'w') as log:
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        log.write(result.stdout)
        log.write(result.stderr)
        log.flush()
    
    return result.returncode

def main():
    print("Starting coupled simulation...")
    
    results = {}
    all_csv_files = []
    
    for level_idx, level in enumerate(MESH_LEVELS):
        print(f"\n=== Mesh Level {level_idx + 1}: h = 1/{level} ===")
        
        h = 1.0 / level
        n_elem = level
        
        # Create working directory for this level
        work_dir = Path(f"level_{level}")
        work_dir.mkdir(exist_ok=True)
        
        # Initialize coupling iteration
        # Start with zero interface values
        interface_u_from_B = np.zeros(len(INTERFACE_PROBE_POINTS))
        interface_qn_from_A = np.zeros(len(INTERFACE_PROBE_POINTS))
        
        residual_history = []
        max_iterations = 100
        tolerance = 1e-6
        
        for iteration in range(max_iterations):
            print(f"  Coupling iteration {iteration + 1}")
            
            # ===== SUBDOMAIN A (4C) - DIRICHLET SIDE =====
            # Receives u from B, imposes as Dirichlet on interface, returns qn
            
            # Generate 4C input for subdomain A
            # Subdomain A: (0, 0.625) x (0, 1), k = 1
            # Interface is at x = 0.625 (right boundary of A)
            
            # We need to create a proper mesh and solve
            # For now, let's use a simpler approach with FEniCS-like structure
            
            # Actually, let me reconsider - 4C needs specific YAML format
            # Let me check what 4C actually expects
            
            pass  # Will implement below
            
            # ===== SUBDOMAIN B (Kratos) - NEUMANN SIDE =====
            # Receives qn from A, applies as Neumann on interface, returns u
            
            pass  # Will implement below
            
            # Check convergence
            if iteration > 0:
                # Compute relative residual
                residual = np.max(np.abs(interface_u_new - interface_u_old)) / (np.max(np.abs(interface_u_old)) + 1e-15)
                residual_history.append(residual)
                print(f"    Residual: {residual:.6e}")
                
                if residual < tolerance:
                    print(f"  Converged after {iteration + 1} iterations")
                    break
        
        # Save results for this level
        # ... (will implement)
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    main()
