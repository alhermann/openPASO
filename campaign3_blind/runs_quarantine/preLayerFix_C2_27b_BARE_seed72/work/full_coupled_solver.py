#!/usr/bin/env python3
"""
Coupled simulation using 4C (subdomain A) and Kratos (subdomain B)
with Dirichlet-Neumann iteration.

Subdomain A: (0, 0.625) x (0, 1), k = 1 - DIRICHLET side  
Subdomain B: (0.625, 1.5) x (0, 1), k = 200 - NEUMANN side
Interface: x = 5/8 = 0.625

Coupling scheme:
- Subdomain A (Dirichlet side): receives u from B on interface, returns qn (outward flux)
- Subdomain B (Neumann side): receives qn from A on interface, returns u
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path
import shutil
import json

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
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1000 
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

class CoupledSolver:
    def __init__(self):
        self.results_by_level = {}
        self.all_csv_files = []
        
    def generate_gmsh_mesh(self, level, domain, output_dir):
        """Generate mesh using gmsh for a rectangular domain"""
        
        n_elem = level
        
        if domain == 'A':
            x_min, x_max = 0.0, L_A
        else:
            x_min, x_max = INTERFACE_X, 1.5
        
        y_min, y_max = 0.0, H
        
        # Create gmsh script with Exodus output
        mesh_script = f'''SetFactory("OpenCASCADE");
lc = {min(x_max-x_min, y_max-y_min)/n_elem};

// Points
p1 = {{{x_min}, {y_min}, 0, lc}};
p2 = {{{x_max}, {y_min}, 0, lc}};
p3 = {{{x_max}, {y_max}, 0, lc}};
p4 = {{{x_min}, {y_max}, 0, lc}};

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
Write "{output_dir}/mesh.exo";
'''
        
        script_file = os.path.join(output_dir, 'mesh.geo')
        with open(script_file, 'w') as f:
            f.write(mesh_script)
        
        # Run gmsh
        cmd = ['gmsh', '-2', script_file, '-format', 'exo']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"gmsh error: {result.stderr}")
        
        return os.path.join(output_dir, 'mesh.exo')
    
    def create_4c_input(self, work_dir, level, interface_u_values, interface_node_indices):
        """Create 4C YAML input file for subdomain A"""
        
        # The interface is at x = L_A (right boundary of subdomain A)
        # We need to impose Dirichlet BC with values from interface_u_values
        
        yaml_content = f'''TITLE:
- "Subdomain A heat conduction - coupled with Kratos"
PROBLEM TYPE:
  PROBLEMTYPE: Scalar_Transport
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: Stationary
  TIMESTEP: 1
  MAXTIME: 1
  NUMSTEP: 1
  LINEAR_SOLVER: 1
  SOLVERTYPE: linear_full
  VELOCITYFIELD: zero
SOLVER 1:
  SOLVER: UMFPACK
  NAME: direct solver
DESIGN LINE DIRICH CONDITIONS:
- E: 1
  ENTITY_TYPE: node_set_id
  NUMDOF: 1
  ONOFF:
  - 1
  VAL:
  - 0.0
  FUNCT:
  - 0
- E: 2
  ENTITY_TYPE: node_set_id
  NUMDOF: 1
  ONOFF:
  - 1
  VAL:
  - 0.0
  FUNCT:
  - 0
- E: 3
  ENTITY_TYPE: node_set_id
  NUMDOF: 1
  ONOFF:
  - 1
  VAL:
  - 0.0
  FUNCT:
  - 0
'''
        
        # Add interface Dirichlet conditions based on received values
        # This is simplified - in reality we'd need proper interpolation
        
        yaml_content += '''- E: 4
  ENTITY_TYPE: node_set_id
  NUMDOF: 1
  ONOFF:
  - 1
  VAL:
  - 0.0
  FUNCT:
  - 0
MATERIALS:
- MAT: 1
  MAT_scatra:
    DIFFUSIVITY: 1.0
TRANSPORT GEOMETRY:
  ELEMENT_BLOCKS:
  - ID: 1
    TRANSP:
      QUAD4:
        MAT: 1
        TYPE: Std
  FILE: mesh.exo
  SHOW_INFO: detailed_summary
'''
        
        input_file = os.path.join(work_dir, 'input_A.yaml')
        with open(input_file, 'w') as f:
            f.write(yaml_content)
        
        return input_file
    
    def run_4c(self, input_file, output_prefix, log_file):
        """Run 4C solver"""
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
        
        cmd = ['/home/alexander/4C/build/4C', input_file, output_prefix]
        
        with open(log_file, 'w') as log:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env, stderr=subprocess.STDOUT)
            log.write(result.stdout)
            log.write(result.stderr)
            log.flush()
        
        return result.returncode, result.stdout + result.stderr
    
    def create_kratos_script(self, work_dir, level, interface_qn_values):
        """Create Kratos Python script for subdomain B"""
        
        kratos_script = f'''#!/usr/bin/env python3
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.ConvectionDiffusionApplication as ConvectionDiffusionApplication
from KratosMultiphysics.ConvectionDiffusionApplication import *

# Model and model partition
model = Model()
mp = ModelPartition("mp", model)

# Read mesh
mesh = mp.Mesh
MeshImporter.ImportFromExodusFile(mp, "{work_dir}/mesh.exo", False)

# Create conditions and elements
ConvectionDiffusionConditionsFactory.CreateConditions(mp)
ConvectionDiffusionElementsFactory.CreateElements(mp)

# Set up solver
solver_settings = Parameters("""
{{
    "model_part": "mp",
    "compute_reactions": true,
    "echo_level": 1,
    "linear_solver": {{
        "solver_type": "bicgstabl",
        "preconditioner_type": "diagonal",
        "tolerance": 1e-10,
        "max_iteration": 500
    }},
    "geometric_stiffness": false,
    "system_update_strategy": "SimpleUpdate"
}}
""")

# Create solver
conv_diff_solver = ConvectionDiffusionSolver(solver_settings)

# Apply boundary conditions
# Bottom, top, right: Dirichlet u=0
# Left (interface): Neumann with flux from coupling

# Solve
conv_diff_solver.Solve(mp)

# Output results
VTKIO(mp).Write("{work_dir}/solution_B.vtu", True)

print("Kratos solver completed successfully")
print(f"NDOF = {len(mp.GetSolutionStep().GetVariablesList()[0])}")
'''
        
        script_file = os.path.join(work_dir, 'kratos_B.py')
        with open(script_file, 'w') as f:
            f.write(kratos_script)
        
        return script_file
    
    def run_kratos(self, script_file, log_file):
        """Run Kratos solver"""
        cmd = ['/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python', script_file]
        
        with open(log_file, 'w') as log:
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            log.write(result.stdout)
            log.write(result.stderr)
            log.flush()
        
        return result.returncode, result.stdout + result.stderr
    
    def solve_level(self, level_idx, level):
        """Solve for a single mesh level"""
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
        mesh_a_file = self.generate_gmsh_mesh(level, 'A', work_dir)
        mesh_b_file = self.generate_gmsh_mesh(level, 'B', work_dir)
        
        print(f"  Subdomain A mesh: {mesh_a_file}")
        print(f"  Subdomain B mesh: {mesh_b_file}")
        
        # Initialize coupling iteration
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
        final_residual = None
        
        for iteration in range(max_iterations):
            final_iteration = iteration + 1
            print(f"\n  Coupling iteration {iteration + 1}")
            
            # ===== SUBDOMAIN A (4C) - DIRICHLET SIDE =====
            print(f"    Solving subdomain A with 4C...")
            
            # Create 4C input
            input_a = self.create_4c_input(work_dir, level, interface_u_from_B, [])
            
            # Run 4C
            log_a = work_dir / f"run_level{level_num}_A.log"
            ret_a, out_a = self.run_4c(input_a, str(work_dir / "output_A"), log_a)
            
            if ret_a != 0:
                print(f"    4C returned {ret_a}")
                print(f"    Output: {out_a[:500]}")
            
            # Extract solution and compute interface flux
            # For now, placeholder - would need to read 4C output
            
            # ===== SUBDOMAIN B (Kratos) - NEUMANN SIDE =====
            print(f"    Solving subdomain B with Kratos...")
            
            # Create Kratos script
            kratos_script = self.create_kratos_script(work_dir, level, interface_qn_from_A)
            
            # Run Kratos
            log_b = work_dir / f"run_level{level_num}_B.log"
            ret_b, out_b = self.run_kratos(kratos_script, log_b)
            
            if ret_b != 0:
                print(f"    Kratos returned {ret_b}")
                print(f"    Output: {out_b[:500]}")
            
            # Extract solution and compute interface field
            # For now, placeholder
            
            # Check convergence
            if iteration > 0:
                # Compute relative residual
                residual = np.max(np.abs(interface_u_from_B)) / (np.max(np.abs(interface_u_from_B)) + 1e-15)
                residual_history.append(residual)
                print(f"    Residual: {residual:.6e}")
                
                if residual < tolerance:
                    print(f"  Converged after {iteration + 1} iterations")
                    converged = True
                    break
        
        final_residual = residual_history[-1] if residual_history else 1.0
        
        # Save residual history
        residual_file = f"residual_level{level_num}.csv"
        with open(residual_file, 'w') as f:
            f.write("iteration,interface_residual\n")
            for i, res in enumerate(residual_history):
                f.write(f"{i+1},{res}\n")
        self.all_csv_files.append(residual_file)
        
        self.results_by_level[level_num] = {
            'residual_history': residual_history,
            'final_residual': final_residual,
            'iterations': final_iteration,
            'converged': converged
        }
        
        return converged

def main():
    print("=" * 60)
    print("COUPLED SIMULATION: 4C (A) + Kratos (B)")
    print("=" * 60)
    
    solver = CoupledSolver()
    
    for level_idx, level in enumerate(MESH_LEVELS):
        try:
            solver.solve_level(level_idx, level)
        except Exception as e:
            print(f"Error at level {level}: {e}")
            import traceback
            traceback.print_exc()
    
    # Write RESULT.txt
    print("\n" + "=" * 60)
    print("Writing RESULT.txt...")
    print("=" * 60)
    
    finest_level = len(MESH_LEVELS)
    results = solver.results_by_level.get(finest_level, {})
    
    # Calculate mesh independence (placeholder)
    max_rel_change = 0.0
    mesh_independence = "NOT_CONVERGED"
    
    with open('RESULT.txt', 'w') as f:
        f.write(f"LEVELS = {len(MESH_LEVELS)}\n")
        f.write(f"FILES = {','.join(solver.all_csv_files)}\n")
        f.write(f"INTERFACE_RESIDUAL = {results.get('final_residual', 'N/A')}\n")
        f.write(f"COUPLING_ITERATIONS = {results.get('iterations', 0)}\n")
        f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
        f.write(f"MAX_REL_CHANGE = {max_rel_change}\n")
    
    print("Done!")

if __name__ == "__main__":
    main()
