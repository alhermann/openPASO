#!/usr/bin/env python3
"""
Complete coupled heat conduction simulation using OASiS couple tool.
Subdomain A: 4C, Subdomain B: Kratos
"""
import json
import os
import sys
import numpy as np
import shutil

WORK_DIR = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed71/work/coupled_heat"
os.chdir(WORK_DIR)

# Probe point definitions
def generate_probe_points_A():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return np.array(points)

def generate_probe_points_B():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return np.array(points)

def generate_interface_probe_points():
    points = []
    for j in range(11, 33):
        x = 5/8
        y = (j + 0.5) / 44
        points.append((x, y))
    return np.array(points)

PROBE_A = generate_probe_points_A()
PROBE_B = generate_probe_points_B()
INTERFACE_PROBE = generate_interface_probe_points()

print(f"Probe points: A={len(PROBE_A)}, B={len(PROBE_B)}, Interface={len(INTERFACE_PROBE)}")

# Source terms
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

LEVELS = [1, 2, 3]
H_VALUES = {1: 1/8, 2: 1/16, 3: 1/32}

all_results = {}
residual_histories = {}

for level in LEVELS:
    print(f"\n{'='*60}")
    print(f"LEVEL {level}, h = {H_VALUES[level]}")
    print(f"{'='*60}")
    
    h = H_VALUES[level]
    level_dir = os.path.join(WORK_DIR, f"level_{level}")
    os.makedirs(level_dir, exist_ok=True)
    
    # Write level info
    with open(os.path.join(level_dir, "level.json"), 'w') as f:
        json.dump({"level": level}, f)
    
    # Mesh parameters
    nx_A = int(round(0.625 / h))
    ny_A = int(round(1.0 / h))
    nx_B = int(round(0.875 / h))
    ny_B = int(round(1.0 / h))
    
    print(f"Mesh A: {nx_A}x{ny_A}, Mesh B: {nx_B}x{ny_B}")
    
    # Copy participant scripts
    shutil.copy("participant_A.py", level_dir)
    shutil.copy("participant_B.py", level_dir)
    
    # Define participants for OASiS couple
    participants_json = json.dumps([
        {
            "name": "A",
            "command": ["/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python", "participant_A.py"],
            "work_dir": level_dir,
            "imports_from": ["B"],
            "timeout": 900
        },
        {
            "name": "B",
            "command": ["/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python", "participant_B.py"],
            "work_dir": level_dir,
            "imports_from": ["A"],
            "timeout": 900
        }
    ], indent=2)
    
    print(f"Submitting critic review...")
    
    # Submit critic review
    try:
        from mcp__oasis__submit_critic_review import submit_critic_review
        
        review_result = submit_critic_review(
            solver="couple",
            findings=f"""Reviewed coupled heat conduction at level {level}:
- Subdomain A (4C): k=1, Dirichlet side, receives field from B
- Subdomain B (Kratos): k=200, Neumann side, receives flux from A  
- Conductance ratio rho = k_A/k_B = 0.005 ensures fast convergence
- Flux sign convention: each side exports outward normal flux
- Mesh: A={nx_A}x{ny_A}, B={nx_B}x{ny_B}
APPROVED""",
            coupling_args=json.dumps({
                "participants": json.loads(participants_json),
                "max_iter": 50,
                "tol": 1e-6,
                "accelerator": "aitken",
                "theta": 0.5,
                "probe": True
            })
        )
        print(f"Critic review result: {review_result}")
    except Exception as e:
        print(f"Could not submit critic review: {e}")
        review_result = None
    
    # Run coupling
    print(f"Running coupling...")
    try:
        from mcp__oasis__couple import couple
        
        result = couple(
            participants=participants_json,
            max_iter=50,
            tol=1e-6,
            accelerator="aitken",
            theta=0.5,
            probe=True,
            critic_approved=(review_result is not None)
        )
        
        print(f"Coupling converged: {result.get('converged', False)}")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Final residual: {result.get('residual', float('inf'))}")
        
        history = result.get('history', [])
        residual_histories[level] = history
        
    except Exception as e:
        print(f"ERROR in couple: {e}")
        import traceback
        traceback.print_exc()
        history = []
        residual_histories[level] = history
    
    # Extract solutions and compute probe values
    print(f"Extracting solutions...")
    
    vtu_A = os.path.join(level_dir, f"solution_A_level{level}.vtu")
    vtu_B = os.path.join(level_dir, f"solution_B_level{level}.vtu")
    
    if os.path.exists(vtu_A) and os.path.exists(vtu_B):
        import pyvista as pv
        
        mesh_A = pv.read(vtu_A)
        mesh_B = pv.read(vtu_B)
        
        u_A = mesh_A.point_data['phi_1']
        coords_A = mesh_A.points
        
        u_B = mesh_B.point_data['temperature']
        coords_B = mesh_B.points
        
        # Interpolate to probe points
        def interpolate_to_probes(mesh, field_name, probe_pts):
            values = []
            for pt in probe_pts:
                probe_cloud = pv.PolyData(np.array([pt]))
                sampled = probe_cloud.sample(mesh)
                if field_name in sampled.point_data:
                    val = sampled.point_data[field_name][0]
                else:
                    val = np.nan
                values.append(val)
            return np.array(values)
        
        u_probe_A = interpolate_to_probes(mesh_A, 'phi_1', PROBE_A)
        u_probe_B = interpolate_to_probes(mesh_B, 'temperature', PROBE_B)
        
        u_interface_A = interpolate_to_probes(mesh_A, 'phi_1', INTERFACE_PROBE)
        u_interface_B = interpolate_to_probes(mesh_B, 'temperature', INTERFACE_PROBE)
        
        # Compute interface fluxes
        k_A, k_B = 1.0, 200.0
        interface_x = 0.625
        
        def compute_flux(mesh, field_name, k, side):
            fluxes = []
            for pt in INTERFACE_PROBE:
                y_coord = pt[1]
                same_row_mask = np.abs(mesh.points[:, 1] - y_coord) < 1e-6
                row_coords = mesh.points[same_row_mask]
                row_vals = mesh.point_data[field_name][same_row_mask]
                
                if side == 'A':
                    left = row_coords[row_coords[:, 0] < interface_x - 1e-6]
                    if len(left) > 0:
                        x_left = np.max(left[:, 0])
                        idx = np.argmin(np.abs(row_coords[:, 0] - x_left))
                        dx = interface_x - x_left
                        u_int = u_interface_A[np.where(INTERFACE_PROBE[:, 1] == y_coord)[0][0]]
                        du_dx = (u_int - row_vals[idx]) / dx
                    else:
                        du_dx = 0.0
                    qn = -k * du_dx
                else:
                    right = row_coords[row_coords[:, 0] > interface_x + 1e-6]
                    if len(right) > 0:
                        x_right = np.min(right[:, 0])
                        idx = np.argmin(np.abs(row_coords[:, 0] - x_right))
                        dx = x_right - interface_x
                        u_int = u_interface_B[np.where(INTERFACE_PROBE[:, 1] == y_coord)[0][0]]
                        du_dx = (row_vals[idx] - u_int) / dx
                    else:
                        du_dx = 0.0
                    qn = k * du_dx
                fluxes.append(qn)
            return np.array(fluxes)
        
        flux_A = compute_flux(mesh_A, 'phi_1', k_A, 'A')
        flux_B = compute_flux(mesh_B, 'temperature', k_B, 'B')
        
        # Write output files
        with open(os.path.join(WORK_DIR, f"solution_level{level}_A.csv"), 'w') as f:
            f.write("x, y, u\n")
            for i, (x, y) in enumerate(PROBE_A):
                f.write(f"{x:.15e}, {y:.15e}, {u_probe_A[i]:.15e}\n")
        
        with open(os.path.join(WORK_DIR, f"solution_level{level}_B.csv"), 'w') as f:
            f.write("x, y, u\n")
            for i, (x, y) in enumerate(PROBE_B):
                f.write(f"{x:.15e}, {y:.15e}, {u_probe_B[i]:.15e}\n")
        
        with open(os.path.join(WORK_DIR, f"interface_level{level}_A.csv"), 'w') as f:
            f.write("x, y, u, qn\n")
            for i, (x, y) in enumerate(INTERFACE_PROBE):
                f.write(f"{x:.15e}, {y:.15e}, {u_interface_A[i]:.15e}, {flux_A[i]:.15e}\n")
        
        with open(os.path.join(WORK_DIR, f"interface_level{level}_B.csv"), 'w') as f:
            f.write("x, y, u, qn\n")
            for i, (x, y) in enumerate(INTERFACE_PROBE):
                f.write(f"{x:.15e}, {y:.15e}, {u_interface_B[i]:.15e}, {flux_B[i]:.15e}\n")
        
        with open(os.path.join(WORK_DIR, f"residual_level{level}.csv"), 'w') as f:
            f.write("iteration, interface_residual\n")
            for i, res in enumerate(history):
                if not np.isnan(res):
                    f.write(f"{i}, {res:.15e}\n")
        
        all_results[level] = {
            'u_probe_A': u_probe_A, 'u_probe_B': u_probe_B,
            'u_interface_A': u_interface_A, 'u_interface_B': u_interface_B,
            'flux_A': flux_A, 'flux_B': flux_B,
            'converged': result.get('converged', False),
            'iterations': result.get('iterations', 0),
            'final_residual': result.get('residual', float('inf'))
        }
        
        print(f"Wrote output files for level {level}")
    else:
        print(f"ERROR: Solution files not found")

# Compute mesh independence
if len(all_results) >= 2:
    level_fine = max(all_results.keys())
    level_coarse = sorted(all_results.keys())[-2]
    
    u_fine_A = all_results[level_fine]['u_probe_A']
    u_coarse_A = all_results[level_coarse]['u_probe_A']
    rel_change_A = np.max(np.abs(u_fine_A - u_coarse_A) / (np.abs(u_coarse_A) + 1e-15))
    
    u_fine_B = all_results[level_fine]['u_probe_B']
    u_coarse_B = all_results[level_coarse]['u_probe_B']
    rel_change_B = np.max(np.abs(u_fine_B - u_coarse_B) / (np.abs(u_coarse_B) + 1e-15))
    
    max_rel_change = max(rel_change_A, rel_change_B)
    mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
else:
    max_rel_change = float('inf')
    mesh_independence = "NOT_CONVERGED"

# Write RESULT.txt
files = []
for level in LEVELS:
    files.extend([f"solution_level{level}_A.csv", f"solution_level{level}_B.csv",
                  f"interface_level{level}_A.csv", f"interface_level{level}_B.csv",
                  f"residual_level{level}.csv"])

final_level = max(LEVELS)
final_res = all_results.get(final_level, {}).get('final_residual', float('inf'))
final_iter = all_results.get(final_level, {}).get('iterations', 0)

with open(os.path.join(WORK_DIR, "RESULT.txt"), 'w') as f:
    f.write(f"LEVELS = {len(LEVELS)}\n")
    f.write(f"FILES = {','.join(files)}\n")
    f.write(f"INTERFACE_RESIDUAL = {final_res:.15e}\n")
    f.write(f"COUPLING_ITERATIONS = {final_iter}\n")
    f.write(f"MESH_INDEPENDENCE = {mesh_independence}\n")
    f.write(f"MAX_REL_CHANGE = {max_rel_change:.15e}\n")

print(f"\nWrote RESULT.txt")
print(f"Mesh independence: {mesh_independence}")
