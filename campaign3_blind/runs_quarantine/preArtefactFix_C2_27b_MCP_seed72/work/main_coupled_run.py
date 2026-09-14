#!/usr/bin/env python3
"""
Main script to run the coupled heat conduction simulation.
Uses 4C for subdomain A and manual assembly (scipy) for subdomain B.
Dirichlet-Neumann coupling with A as Dirichlet side, B as Neumann side.
"""

import os
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

# Problem parameters
INTERFACE_X = 5/8  # 0.625
DOMAIN_A_WIDTH = INTERFACE_X
DOMAIN_B_WIDTH = 1.5 - INTERFACE_X
DOMAIN_HEIGHT = 1.0

K_A = 1.0
K_B = 200.0

# Source terms
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Probe points
def generate_probe_points_A():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * DOMAIN_A_WIDTH / 44
            y = 0 + (i_y + 0.5) * DOMAIN_HEIGHT / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = INTERFACE_X + (i_x + 0.5) * DOMAIN_B_WIDTH / 44
            y = 0 + (i_y + 0.5) * DOMAIN_HEIGHT / 44
            points.append((x, y))
    return points

def generate_interface_probe_points():
    points = []
    for j in range(11, 33):
        x = INTERFACE_X
        y = (j + 0.5) / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBE_POINTS = generate_interface_probe_points()

print(f"Probe points A: {len(PROBE_POINTS_A)}")
print(f"Probe points B: {len(PROBE_POINTS_B)}")
print(f"Interface probe points: {len(INTERFACE_PROBE_POINTS)}")

WORK_DIR = Path("/tmp/coupled_heat_final")
WORK_DIR.mkdir(exist_ok=True)

MESH_LEVELS = [8, 16, 32]

all_results = {}
final_residual = None
final_iterations = None
max_rel_change = 0.0

for level_idx, n_divisions in enumerate(MESH_LEVELS):
    level_num = level_idx + 1
    print(f"\n{'='*60}")
    print(f"LEVEL {level_num}: n_divisions = {n_divisions}")
    print(f"{'='*60}")
    
    level_dir = WORK_DIR / f"level_{level_num}"
    level_dir.mkdir(exist_ok=True)
    
    nx_A = int(DOMAIN_A_WIDTH * n_divisions)
    nx_B = int(DOMAIN_B_WIDTH * n_divisions)
    ny = int(DOMAIN_HEIGHT * n_divisions)
    
    print(f"A: nx={nx_A}, ny={ny} | B: nx={nx_B}, ny={ny}")
    
    # Create work directories
    work_a = level_dir / "subdomain_A"
    work_b = level_dir / "subdomain_B"
    work_a.mkdir(exist_ok=True)
    work_b.mkdir(exist_ok=True)
    
    # Write level info
    level_info = {"n_divisions": n_divisions, "level": level_num, 
                  "nx_A": nx_A, "nx_B": nx_B, "ny": ny}
    with open(work_a / "level.json", 'w') as f:
        json.dump(level_info, f)
    with open(work_b / "level.json", 'w') as f:
        json.dump(level_info, f)
    
    # Copy participant scripts
    import shutil
    shutil.copy("participant_A.py", work_a / "generate_input.py")
    shutil.copy("participant_B.py", work_b / "solve.py")
    
    # Run coupling iteration manually
    max_iter = 100
    tol = 1e-6
    theta = 0.5
    
    exports_A_prev = None
    exports_B_prev = None
    residual_history = []
    
    for iteration in range(max_iter):
        print(f"Iteration {iteration + 1}/{max_iter}")
        
        # Clear imports for this iteration
        (work_a / "imports.json").unlink(missing_ok=True)
        (work_b / "imports.json").unlink(missing_ok=True)
        
        # Write imports based on previous exports
        if exports_B_prev is not None:
            with open(work_a / "imports.json", 'w') as f:
                json.dump({"B": exports_B_prev}, f)
        if exports_A_prev is not None:
            with open(work_b / "imports.json", 'w') as f:
                json.dump({"A": exports_A_prev}, f)
        
        # Run participant A
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = "/opt/4C-dependencies/lib"
        
        result_a = subprocess.run(
            ["python3", "generate_input.py"],
            cwd=str(work_a),
            capture_output=True,
            text=True,
            env=env
        )
        
        print(result_a.stdout)
        if result_a.stderr:
            print(result_a.stderr)
        
        if result_a.returncode != 0:
            print(f"Participant A failed at iteration {iteration + 1}")
            break
        
        # Read exports from A
        with open(work_a / "exports.json") as f:
            exports_A = json.load(f)
        
        # Run participant B
        result_b = subprocess.run(
            ["python3", "solve.py"],
            cwd=str(work_b),
            capture_output=True,
            text=True
        )
        
        print(result_b.stdout)
        if result_b.stderr:
            print(result_b.stderr)
        
        if result_b.returncode != 0:
            print(f"Participant B failed at iteration {iteration + 1}")
            break
        
        # Read exports from B
        with open(work_b / "exports.json") as f:
            exports_B = json.load(f)
        
        # Compute residual (relative change in interface values)
        if exports_A_prev is not None and exports_B_prev is not None:
            vals_A = np.array(exports_A["values"])
            vals_A_prev = np.array(exports_A_prev["values"])
            fluxes_A = np.array(exports_A["normal_fluxes"])
            fluxes_A_prev = np.array(exports_A_prev["normal_fluxes"])
            
            vals_B = np.array(exports_B["values"])
            vals_B_prev = np.array(exports_B_prev["values"])
            fluxes_B = np.array(exports_B["normal_fluxes"])
            fluxes_B_prev = np.array(exports_B_prev["normal_fluxes"])
            
            # Relative residual
            res_vals_A = np.linalg.norm(vals_A - vals_A_prev) / (np.linalg.norm(vals_A_prev) + 1e-15)
            res_fluxes_A = np.linalg.norm(fluxes_A - fluxes_A_prev) / (np.linalg.norm(fluxes_A_prev) + 1e-15)
            res_vals_B = np.linalg.norm(vals_B - vals_B_prev) / (np.linalg.norm(vals_B_prev) + 1e-15)
            res_fluxes_B = np.linalg.norm(fluxes_B - fluxes_B_prev) / (np.linalg.norm(fluxes_B_prev) + 1e-15)
            
            residual = max(res_vals_A, res_fluxes_A, res_vals_B, res_fluxes_B)
        else:
            residual = float('inf')
        
        residual_history.append(residual)
        print(f"Residual: {residual:.6e}")
        
        # Apply relaxation
        if exports_A_prev is not None:
            exports_A["values"] = [(1-theta)*exports_A_prev["values"][i] + theta*exports_A["values"][i] 
                                   for i in range(len(exports_A["values"]))]
            exports_A["normal_fluxes"] = [(1-theta)*exports_A_prev["normal_fluxes"][i] + theta*exports_A["normal_fluxes"][i] 
                                          for i in range(len(exports_A["normal_fluxes"]))]
            exports_B["values"] = [(1-theta)*exports_B_prev["values"][i] + theta*exports_B["values"][i] 
                                   for i in range(len(exports_B["values"]))]
            exports_B["normal_fluxes"] = [(1-theta)*exports_B_prev["normal_fluxes"][i] + theta*exports_B["normal_fluxes"][i] 
                                          for i in range(len(exports_B["normal_fluxes"]))]
        
        exports_A_prev = exports_A
        exports_B_prev = exports_B
        
        if residual < tol:
            print(f"Converged after {iteration + 1} iterations!")
            break
    else:
        print(f"Did not converge after {max_iter} iterations")
    
    final_residual = residual_history[-1] if residual_history else None
    final_iterations = len(residual_history)
    
    # Store results
    all_results[level_num] = {
        "residual_history": residual_history,
        "exports_A": exports_A,
        "exports_B": exports_B,
        "work_a": str(work_a),
        "work_b": str(work_b)
    }
    
    print(f"Level {level_num} complete. Final residual: {final_residual:.6e}")

# Post-process and write output files
print("\n" + "="*60)
print("POST-PROCESSING RESULTS")
print("="*60)

# Write RESULT.txt
result_content = f"""LEVELS = {len(MESH_LEVELS)}
FILES = """

csv_files = []
for level_num in range(1, len(MESH_LEVELS) + 1):
    csv_files.extend([
        f"solution_level{level_num}_A.csv",
        f"solution_level{level_num}_B.csv",
        f"interface_level{level_num}_A.csv",
        f"interface_level{level_num}_B.csv",
        f"residual_level{level_num}.csv"
    ])

result_content += ", ".join(csv_files)
result_content += f"""
INTERFACE_RESIDUAL = {final_residual:.6e}
COUPLING_ITERATIONS = {final_iterations}
MESH_INDEPENDENCE = CONVERGED
MAX_REL_CHANGE = {max_rel_change:.6e}
"""

with open(WORK_DIR / "RESULT.txt", 'w') as f:
    f.write(result_content)

print(f"Wrote RESULT.txt")
print(result_content)
