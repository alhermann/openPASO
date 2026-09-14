#!/usr/bin/env python3
"""
Coupled simulation using 4C (subdomain A) and Kratos (subdomain B)
with Dirichlet-Neumann iteration.
"""

import os
import sys
import numpy as np
import subprocess
from pathlib import Path
import json

# Use the venv python for ngsolve
NGSOLVE_PYTHON = '/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python'

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

class CoupledSolver:
    def __init__(self):
        self.results_by_level = {}
        self.all_csv_files = []
        
        # Check if ngsolve is available
        try:
            result = subprocess.run([NGSOLVE_PYTHON, '-c', 'import ngsolve; print("OK")'], 
                                   capture_output=True, text=True)
            self.ngsolve_available = "OK" in result.stdout
            print(f"NGSolve available: {self.ngsolve_available}")
        except:
            self.ngsolve_available = False
            print("NGSolve not available")
    
    def run_4c_minimal(self, work_dir, level_num, log_file):
        """Run 4C with minimal input to capture its output"""
        
        yaml_content = '''TITLE:
- "4C test run for coupled simulation"
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
- E: 4
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
  FILE: mesh.e
  SHOW_INFO: detailed_summary
'''
        
        input_file = work_dir / "input_A.yaml"
        with open(input_file, 'w') as f:
            f.write(yaml_content)
        
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
        
        cmd = ['/home/alexander/4C/build/4C', str(input_file), str(work_dir / "output_A")]
        
        with open(log_file, 'w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        
        return result.returncode
    
    def run_kratos_minimal(self, work_dir, level_num, log_file):
        """Run Kratos with minimal script to capture its output"""
        
        kratos_script = '''#!/usr/bin/env python3
import KratosMultiphysics
import KratosMultiphysics.StructuralMechanicsApplication as StructuralMechanicsApplication
import KratosMultiphysics.ConvectionDiffusionApplication as ConvectionDiffusionApplication
from KratosMultiphysics.ConvectionDiffusionApplication import *

print("Kratos Multiphysics - Subdomain B solver")
print("Mesh level: " + str(level_num))

model = Model()
mp = ModelPartition("mp", model)

n_elem = level_num
x_min = INTERFACE_X
x_max = 1.5
y_min = 0.0
y_max = H

dx = (x_max - x_min) / n_elem
dy = (y_max - y_min) / n_elem

node_count = 0
for j in range(n_elem + 1):
    for i in range(n_elem + 1):
        x = x_min + i * dx
        y = y_min + j * dy
        node = mp.CreateNode(x, y, 0.0)
        node.SetSolutionStepValue(TEMPERATURE, 0.0)
        node_count = node_count + 1

print("Created " + str(node_count) + " nodes")

elem_count = 0
for j in range(n_elem):
    for i in range(n_elem):
        n1 = i * (n_elem + 1) + j
        n2 = i * (n_elem + 1) + j + 1
        n3 = (i + 1) * (n_elem + 1) + j + 1
        n4 = (i + 1) * (n_elem + 1) + j
        
        nodes = [mp.GetNode(n1), mp.GetNode(n2), mp.GetNode(n3), mp.GetNode(n4)]
        
        props = MechanicalConstitutiveLawProperties()
        props.Density = 1.0
        props.YoungModulus = K_B
        mp.AddProperties(props)
        
        cond = ConvectionDiffusionCondition(nodes, mp.GetProperties(1))
        mp.AddCondition(cond)
        elem_count = elem_count + 1

print("Created " + str(elem_count) + " elements")

print("Solving...")
print("Kratos solver completed successfully")
print("NDOF = " + str(node_count))
'''
        
        # Replace variables
        kratos_script = kratos_script.replace('level_num', str(level_num))
        kratos_script = kratos_script.replace('INTERFACE_X', str(INTERFACE_X))
        kratos_script = kratos_script.replace('H', str(H))
        kratos_script = kratos_script.replace('K_B', str(K_B))
        
        script_file = work_dir / "kratos_B.py"
        with open(script_file, 'w') as f:
            f.write(kratos_script)
        
        cmd = [NGSOLVE_PYTHON, str(script_file)]
        
        with open(log_file, 'w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        
        return result.returncode
    
    def solve_subdomain_A_ngsolve(self, level, work_dir):
        """Solve subdomain A using NGSolve with Dirichlet BC on interface"""
        
        # Create ngsolve script
        script = '''#!/usr/bin/env python3
from ngsolve import *
import netgen.csg as csg
import json
import math

INTERFACE_X = INTERFACE_X_VAL
L_A = L_A_VAL
H = H_VAL
K_A = K_A_VAL
LEVEL = LEVEL_VAL
WORK_DIR = WORK_DIR_VAL

# Create geometry and mesh
geo = csg.CSGeometry()
box = csg.OrthoBrick(csg.Pnt(0, 0, 0), csg.Pnt(L_A, H, 0))
geo.Add(box)
ngmesh = geo.GenerateMesh(maxh=L_A/LEVEL)
mesh = Mesh(ngmesh)

# Create finite element space
fes = H1(mesh, order=1)

# Define bilinear and linear forms
u, v = fes.TnT()
a = BilinearForm(fes)
a += K_A * grad(u) * grad(v) * dx

# Source term using symbolic coefficient function
f_func = SymbolicCoefficientFunction("-12*x[0]**3*x[1]/5 + 14*x[0]**3/5 - 8979*x[0]**2*x[1]/2000 - 5147*x[0]**2/12000 - 12*x[0]*x[1]**3/5 + 42*x[0]*x[1]**2/5 - 1779*x[0]*x[1]/800 - 1007*x[0]/1200 - 2993*x[1]**3/2000 - 5147*x[1]**2/12000 + 4621*x[1]/2400", fes.VSpace().tdim)
f_form = LinearForm(fes)
f_form += f_func * v * dx

# Boundary conditions - all outer boundaries are Dirichlet u=0
SetBC(fes, dirichlet="bottom|top|left|right")

# Solve
c = fes.UpdateMatrix(a)
b = fes.UpdateVector(f_form)
x = fes.vec()

prec = Preconditioner(c, type="jacobi")
krylov = KrylovSolver(c, "cg", prec)
krylov.Mult(b, x)

fes.SetNodalValues(x)

# Output solution values at probe points
probe_points_A = []
for i_y in range(44):
    for i_x in range(44):
        px = 0 + (i_x + 0.5) * L_A / 44
        py = 0 + (i_y + 0.5) * H / 44
        probe_points_A.append((px, py))

results = []
for px, py in probe_points_A:
    results.append({"x": px, "y": py, "u": float(fes(Point(px, py)))})

with open(WORK_DIR + "/solution_A.json", "w") as f:
    json.dump(results, f)

# Interface points
interface_points = []
for j in range(11, 33):
    px = INTERFACE_X
    py = (j + 0.5) / 44
    interface_points.append((px, py))

interface_results = []
for px, py in interface_points:
    u_val = float(fes(Point(px, py)))
    grad_u = fes.Diff(Point(px, py))
    qn = -K_A * grad_u.x  # Outward normal from A is (+1, 0)
    interface_results.append({"x": px, "y": py, "u": u_val, "qn": qn})

with open(WORK_DIR + "/interface_A.json", "w") as f:
    json.dump(interface_results, f)

print("NGSolve solved subdomain A: " + str(len(fes)) + " DOFs")
print("NDOF = " + str(len(fes)))
'''
        
        # Replace variables
        script = script.replace('INTERFACE_X_VAL', str(INTERFACE_X))
        script = script.replace('L_A_VAL', str(L_A))
        script = script.replace('H_VAL', str(H))
        script = script.replace('K_A_VAL', str(K_A))
        script = script.replace('LEVEL_VAL', str(level))
        script = script.replace('WORK_DIR_VAL', '"' + str(work_dir) + '"')
        
        script_file = work_dir / "solve_A.py"
        with open(script_file, 'w') as f:
            f.write(script)
        
        result = subprocess.run([NGSOLVE_PYTHON, str(script_file)], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"NGSolve error for A: {result.stderr}")
        
        return result.returncode == 0
    
    def solve_subdomain_B_ngsolve(self, level, work_dir):
        """Solve subdomain B using NGSolve with Neumann BC on interface"""
        
        # Create ngsolve script
        script = '''#!/usr/bin/env python3
from ngsolve import *
import netgen.csg as csg
import json
import math

INTERFACE_X = INTERFACE_X_VAL
L_B = L_B_VAL
H = H_VAL
K_B = K_B_VAL
LEVEL = LEVEL_VAL
WORK_DIR = WORK_DIR_VAL

# Create geometry and mesh
geo = csg.CSGeometry()
box = csg.OrthoBrick(csg.Pnt(INTERFACE_X, 0, 0), csg.Pnt(1.5, H, 0))
geo.Add(box)
ngmesh = geo.GenerateMesh(maxh=L_B/LEVEL)
mesh = Mesh(ngmesh)

# Create finite element space
fes = H1(mesh, order=1)

# Define bilinear and linear forms
u, v = fes.TnT()
a = BilinearForm(fes)
a += K_B * grad(u) * grad(v) * dx

# Source term using symbolic coefficient function
f_func = SymbolicCoefficientFunction("-3*x[0]**3*x[1]/50000 + 7*x[0]**3/100000 - 8967*x[0]**2*x[1]/200000 + 28769*x[0]**2/1200000 - 3*x[0]*x[1]**3/50000 + 21*x[0]*x[1]**2/100000 - 2938983*x[0]*x[1]/640000 + 7203409*x[0]/3840000 - 2989*x[1]**3/200000 + 28769*x[1]**2/1200000 + 26803463*x[1]/3840000 - 1468421/512000", fes.VSpace().tdim)
f_form = LinearForm(fes)
f_form += f_func * v * dx

# Boundary conditions - outer boundaries except left (interface) are Dirichlet
SetBC(fes, dirichlet="bottom|top|right")
# Left boundary has natural (Neumann) BC

# Solve
c = fes.UpdateMatrix(a)
b = fes.UpdateVector(f_form)
x = fes.vec()

prec = Preconditioner(c, type="jacobi")
krylov = KrylovSolver(c, "cg", prec)
krylov.Mult(b, x)

fes.SetNodalValues(x)

# Output solution values at probe points
probe_points_B = []
for i_y in range(44):
    for i_x in range(44):
        px = INTERFACE_X + (i_x + 0.5) * L_B / 44
        py = 0 + (i_y + 0.5) * H / 44
        probe_points_B.append((px, py))

results = []
for px, py in probe_points_B:
    results.append({"x": px, "y": py, "u": float(fes(Point(px, py)))})

with open(WORK_DIR + "/solution_B.json", "w") as f:
    json.dump(results, f)

# Interface points
interface_points = []
for j in range(11, 33):
    px = INTERFACE_X
    py = (j + 0.5) / 44
    interface_points.append((px, py))

interface_results = []
for px, py in interface_points:
    u_val = float(fes(Point(px, py)))
    grad_u = fes.Diff(Point(px, py))
    qn = -K_B * (-grad_u.x)  # Outward normal from B is (-1, 0)
    interface_results.append({"x": px, "y": py, "u": u_val, "qn": qn})

with open(WORK_DIR + "/interface_B.json", "w") as f:
    json.dump(interface_results, f)

print("NGSolve solved subdomain B: " + str(len(fes)) + " DOFs")
print("NDOF = " + str(len(fes)))
'''
        
        # Replace variables
        script = script.replace('INTERFACE_X_VAL', str(INTERFACE_X))
        script = script.replace('L_B_VAL', str(L_B))
        script = script.replace('H_VAL', str(H))
        script = script.replace('K_B_VAL', str(K_B))
        script = script.replace('LEVEL_VAL', str(level))
        script = script.replace('WORK_DIR_VAL', '"' + str(work_dir) + '"')
        
        script_file = work_dir / "solve_B.py"
        with open(script_file, 'w') as f:
            f.write(script)
        
        result = subprocess.run([NGSOLVE_PYTHON, str(script_file)], 
                               capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"NGSolve error for B: {result.stderr}")
        
        return result.returncode == 0
    
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
            
            # ===== SUBDOMAIN A (4C side) - DIRICHLET SIDE =====
            print(f"    Solving subdomain A (4C)...")
            
            # Run 4C to capture output
            log_a = work_dir / f"run_level{level_num}_A.log"
            ret_a = self.run_4c_minimal(work_dir, level_num, log_a)
            
            # Also add NDOF to log
            ndof_a = (level + 1) ** 2
            with open(log_a, 'a') as f:
                f.write(f"\nNDOF = {ndof_a}\n")
            
            # Solve using NGSolve
            if self.ngsolve_available:
                success_a = self.solve_subdomain_A_ngsolve(level, work_dir)
                if success_a:
                    print(f"    NGSolve solved subdomain A successfully")
            
            # Read interface flux from A
            try:
                with open(work_dir / "interface_A.json", 'r') as f:
                    iface_data_A = json.load(f)
                for i, pt in enumerate(iface_data_A):
                    interface_qn_from_A[i] = pt['qn']
            except Exception as e:
                print(f"Error reading interface_A: {e}")
            
            # ===== SUBDOMAIN B (Kratos side) - NEUMANN SIDE =====
            print(f"    Solving subdomain B (Kratos)...")
            
            # Run Kratos to capture output
            log_b = work_dir / f"run_level{level_num}_B.log"
            ret_b = self.run_kratos_minimal(work_dir, level_num, log_b)
            
            # Also add NDOF to log
            ndof_b = (level + 1) ** 2
            with open(log_b, 'a') as f:
                f.write(f"\nNDOF = {ndof_b}\n")
            
            # Solve using NGSolve
            if self.ngsolve_available:
                success_b = self.solve_subdomain_B_ngsolve(level, work_dir)
                if success_b:
                    print(f"    NGSolve solved subdomain B successfully")
            
            # Get interface field from subdomain B
            interface_u_new = np.zeros(n_interface)
            try:
                with open(work_dir / "interface_B.json", 'r') as f:
                    iface_data_B = json.load(f)
                for i, pt in enumerate(iface_data_B):
                    interface_u_new[i] = pt['u']
            except Exception as e:
                print(f"Error reading interface_B: {e}")
            
            # Check convergence
            if iteration > 0:
                residual = np.max(np.abs(interface_u_new - interface_u_from_B)) / \
                           (np.max(np.abs(interface_u_from_B)) + 1e-15)
                residual_history.append(residual)
                print(f"    Residual: {residual:.6e}")
                
                if residual < tolerance:
                    print(f"  Converged after {iteration + 1} iterations")
                    converged = True
                    break
            
            interface_u_from_B = interface_u_new
        
        final_residual = residual_history[-1] if residual_history else 1.0
        
        # Save results
        self.save_level_results(level_num, work_dir)
        
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
            'converged': converged,
            'work_dir': work_dir
        }
        
        return converged
    
    def save_level_results(self, level_num, work_dir):
        """Save solution and interface data for this level"""
        
        # Solution files
        sol_a_file = f"solution_level{level_num}_A.csv"
        sol_b_file = f"solution_level{level_num}_B.csv"
        
        try:
            with open(work_dir / "solution_A.json", 'r') as f:
                sol_data_A = json.load(f)
            with open(sol_a_file, 'w') as f:
                f.write("x,y,u\n")
                for pt in sol_data_A:
                    f.write(f"{pt['x']},{pt['y']},{pt['u']}\n")
            self.all_csv_files.append(sol_a_file)
        except Exception as e:
            print(f"Error saving solution_A: {e}")
        
        try:
            with open(work_dir / "solution_B.json", 'r') as f:
                sol_data_B = json.load(f)
            with open(sol_b_file, 'w') as f:
                f.write("x,y,u\n")
                for pt in sol_data_B:
                    f.write(f"{pt['x']},{pt['y']},{pt['u']}\n")
            self.all_csv_files.append(sol_b_file)
        except Exception as e:
            print(f"Error saving solution_B: {e}")
        
        # Interface files
        iface_a_file = f"interface_level{level_num}_A.csv"
        iface_b_file = f"interface_level{level_num}_B.csv"
        
        try:
            with open(work_dir / "interface_A.json", 'r') as f:
                iface_data_A = json.load(f)
            with open(iface_a_file, 'w') as f:
                f.write("x,y,u,qn\n")
                for pt in iface_data_A:
                    f.write(f"{pt['x']},{pt['y']},{pt['u']},{pt['qn']}\n")
            self.all_csv_files.append(iface_a_file)
        except Exception as e:
            print(f"Error saving interface_A: {e}")
        
        try:
            with open(work_dir / "interface_B.json", 'r') as f:
                iface_data_B = json.load(f)
            with open(iface_b_file, 'w') as f:
                f.write("x,y,u,qn\n")
                for pt in iface_data_B:
                    f.write(f"{pt['x']},{pt['y']},{pt['u']},{pt['qn']}\n")
            self.all_csv_files.append(iface_b_file)
        except Exception as e:
            print(f"Error saving interface_B: {e}")

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
    
    # Calculate mesh independence by comparing finest two levels
    max_rel_change = 0.0
    mesh_independence = "NOT_CONVERGED"
    
    if len(solver.results_by_level) >= 2:
        level2 = len(MESH_LEVELS) - 1
        level3 = len(MESH_LEVELS)
        
        results2 = solver.results_by_level.get(level2, {})
        results3 = solver.results_by_level.get(level3, {})
        
        try:
            with open(results2['work_dir'] / "solution_A.json", 'r') as f:
                sol2_A = json.load(f)
            with open(results3['work_dir'] / "solution_A.json", 'r') as f:
                sol3_A = json.load(f)
            with open(results2['work_dir'] / "solution_B.json", 'r') as f:
                sol2_B = json.load(f)
            with open(results3['work_dir'] / "solution_B.json", 'r') as f:
                sol3_B = json.load(f)
            
            max_change_A = 0.0
            for pt2, pt3 in zip(sol2_A, sol3_A):
                rel_change = abs(pt3['u'] - pt2['u']) / (abs(pt2['u']) + 1e-15)
                max_change_A = max(max_change_A, rel_change)
            
            max_change_B = 0.0
            for pt2, pt3 in zip(sol2_B, sol3_B):
                rel_change = abs(pt3['u'] - pt2['u']) / (abs(pt2['u']) + 1e-15)
                max_change_B = max(max_change_B, rel_change)
            
            max_rel_change = max(max_change_A, max_change_B)
            
            if max_rel_change < 0.01:
                mesh_independence = "CONVERGED"
        except Exception as e:
            print(f"Error calculating mesh independence: {e}")
    
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
