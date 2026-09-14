#!/usr/bin/env python3
"""
Driver script for coupled thermal diffusion simulation.
Runs Dirichlet-Neumann iteration for each mesh level.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import subprocess

# Problem parameters
L_A = 0.625
L_B = 0.875
H = 1.0
k_A = 1.0
k_B = 200.0
x_interface = 0.625

# Mesh levels
mesh_levels = [
    {"level": 1, "nx_A": 5, "ny": 8, "nx_B": 7},
    {"level": 2, "nx_A": 10, "ny": 16, "nx_B": 14},
    {"level": 3, "nx_A": 20, "ny": 32, "nx_B": 28}
]

def run_participant(level, side, work_dir, imports=None):
    env = os.environ.copy()
    env['LEVEL'] = str(level)
    env['SIDE'] = side
    env['WORK_DIR'] = work_dir
    
    if imports is not None:
        imports_file = os.path.join(work_dir, 'imports.json')
        partner = 'B' if side == 'A' else 'A'
        with open(imports_file, 'w') as f:
            json.dump({partner: imports}, f)
    
    script = '/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed70/work/coupled_thermal_complete.py'
    
    result = subprocess.run([sys.executable, script], 
                          cwd=work_dir, 
                          env=env,
                          capture_output=True, 
                          text=True)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    
    exports_file = os.path.join(work_dir, 'exports.json')
    if os.path.exists(exports_file):
        with open(exports_file, 'r') as f:
            exports = json.load(f)
        return exports.get(side)
    return None

def interpolate_to_fixed_grid(data, target_ny):
    """Interpolate interface data to a fixed grid."""
    ys = np.array([c[1] for c in data['coordinates']])
    vs = np.array(data['values'])
    qs = np.array(data['normal_fluxes'])
    
    target_ys = np.linspace(0, H, target_ny + 1)
    
    new_values = []
    new_fluxes = []
    new_coords = []
    
    for ty in target_ys:
        mask = ys <= ty
        if not np.any(mask):
            idx = 0
        elif np.all(mask):
            idx = len(ys) - 2
        else:
            idx = np.where(mask)[0][-1]
        
        if ys[idx+1] == ys[idx]:
            t = 0
        else:
            t = (ty - ys[idx]) / (ys[idx+1] - ys[idx])
        
        new_values.append(vs[idx] * (1-t) + vs[idx+1] * t)
        new_fluxes.append(qs[idx] * (1-t) + qs[idx+1] * t)
        new_coords.append([x_interface, ty])
    
    return {
        'coordinates': new_coords,
        'values': new_values,
        'normal_fluxes': new_fluxes
    }

def coupled_solve(level, base_work_dir):
    work_dir_A = os.path.join(base_work_dir, f'level{level}_A')
    work_dir_B = os.path.join(base_work_dir, f'level{level}_B')
    
    os.makedirs(work_dir_A, exist_ok=True)
    os.makedirs(work_dir_B, exist_ok=True)
    
    ml = mesh_levels[level - 1]
    ny = ml['ny']
    
    max_iter = 100
    tol = 1e-6
    theta = 0.5
    
    residual_history = []
    
    interface_temp = [0.0] * (ny + 1)
    interface_coords = [[x_interface, j * H / ny] for j in range(ny + 1)]
    
    result_A = None
    result_B = None
    
    for iteration in range(max_iter):
        imports_A = {
            'coordinates': interface_coords,
            'values': interface_temp
        }
        result_A = run_participant(level, 'A', work_dir_A, imports_A)
        
        if result_A is None:
            print("Failed to run subdomain A")
            break
        
        imports_B = {
            'coordinates': result_A['coordinates'],
            'normal_fluxes': result_A['normal_fluxes']
        }
        result_B = run_participant(level, 'B', work_dir_B, imports_B)
        
        if result_B is None:
            print("Failed to run subdomain B")
            break
        
        old_temp = interface_temp
        new_temp = result_B['values']
        
        residual = np.max(np.abs(np.array(new_temp) - np.array(old_temp))) / (np.max(np.abs(np.array(old_temp))) + 1e-10)
        residual_history.append(residual)
        
        interface_temp = [theta * new_temp[j] + (1 - theta) * old_temp[j] for j in range(len(old_temp))]
        
        print(f"Iteration {iteration + 1}: residual = {residual:.6e}")
        
        if residual < tol:
            print(f"Converged after {iteration + 1} iterations")
            break
    
    return result_A, result_B, residual_history

def main():
    base_work_dir = '/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed70/work'
    
    all_results = {}
    all_residuals = {}
    
    # Fixed reference grid for comparison (use finest level's ny)
    ref_ny = 32
    
    for ml in mesh_levels:
        level = ml['level']
        print(f"\n{'='*50}")
        print(f"Mesh level {level}: nx_A={ml['nx_A']}, nx_B={ml['nx_B']}, ny={ml['ny']}")
        print(f"{'='*50}")
        
        result_A, result_B, residuals = coupled_solve(level, base_work_dir)
        
        # Interpolate to reference grid for comparison
        result_A_interp = interpolate_to_fixed_grid(result_A, ref_ny)
        result_B_interp = interpolate_to_fixed_grid(result_B, ref_ny)
        
        all_results[level] = {
            'A': result_A,
            'B': result_B,
            'A_interp': result_A_interp,
            'B_interp': result_B_interp
        }
        all_residuals[level] = residuals
        
        # Write residual history
        residual_file = os.path.join(base_work_dir, f'residual_level{level}.csv')
        with open(residual_file, 'w') as f:
            f.write('iteration,interface_residual\n')
            for i, r in enumerate(residuals):
                f.write(f'{i+1},{r:.15e}\n')
        
        # Write solution CSVs (interface data)
        if result_A:
            sol_file_A = os.path.join(base_work_dir, f'solution_level{level}_A.csv')
            with open(sol_file_A, 'w') as f:
                f.write('x,y,u\n')
                for coord, val in zip(result_A['coordinates'], result_A['values']):
                    f.write(f'{coord[0]:.15e},{coord[1]:.15e},{val:.15e}\n')
            
            iface_file_A = os.path.join(base_work_dir, f'interface_level{level}_A.csv')
            with open(iface_file_A, 'w') as f:
                f.write('x,y,u,qn\n')
                for coord, val, qn in zip(result_A['coordinates'], result_A['values'], result_A['normal_fluxes']):
                    f.write(f'{coord[0]:.15e},{coord[1]:.15e},{val:.15e},{qn:.15e}\n')
        
        if result_B:
            sol_file_B = os.path.join(base_work_dir, f'solution_level{level}_B.csv')
            with open(sol_file_B, 'w') as f:
                f.write('x,y,u\n')
                for coord, val in zip(result_B['coordinates'], result_B['values']):
                    f.write(f'{coord[0]:.15e},{coord[1]:.15e},{val:.15e}\n')
            
            iface_file_B = os.path.join(base_work_dir, f'interface_level{level}_B.csv')
            with open(iface_file_B, 'w') as f:
                f.write('x,y,u,qn\n')
                for coord, val, qn in zip(result_B['coordinates'], result_B['values'], result_B['normal_fluxes']):
                    f.write(f'{coord[0]:.15e},{coord[1]:.15e},{val:.15e},{qn:.15e}\n')
    
    # Check mesh independence using interpolated values on common grid
    if 2 in all_results and 3 in all_results:
        vals2_A = np.array(all_results[2]['A_interp']['values'])
        vals3_A = np.array(all_results[3]['A_interp']['values'])
        rel_change_A = np.max(np.abs(vals3_A - vals2_A)) / (np.max(np.abs(vals2_A)) + 1e-10)
        max_rel_change = rel_change_A
        mesh_independence = "CONVERGED" if max_rel_change < 0.01 else "NOT_CONVERGED"
    else:
        max_rel_change = 0.0
        mesh_independence = "NOT_CONVERGED"
    
    final_residual = all_residuals[3][-1] if 3 in all_residuals else all_residuals[list(all_residuals.keys())[-1]][-1]
    final_iterations = len(all_residuals[3]) if 3 in all_residuals else len(all_residuals[list(all_residuals.keys())[-1]])
    
    csv_files = []
    for level in [1, 2, 3]:
        csv_files.extend([
            f'solution_level{level}_A.csv',
            f'solution_level{level}_B.csv',
            f'interface_level{level}_A.csv',
            f'interface_level{level}_B.csv',
            f'residual_level{level}.csv'
        ])
    
    result_txt = os.path.join(base_work_dir, 'RESULT.txt')
    with open(result_txt, 'w') as f:
        f.write(f'LEVELS = 3\n')
        f.write(f'FILES = {", ".join(csv_files)}\n')
        f.write(f'INTERFACE_RESIDUAL = {final_residual:.15e}\n')
        f.write(f'COUPLING_ITERATIONS = {final_iterations}\n')
        f.write(f'MESH_INDEPENDENCE = {mesh_independence}\n')
        f.write(f'MAX_REL_CHANGE = {max_rel_change:.15e}\n')
    
    print(f"\nResults written to {result_txt}")

if __name__ == '__main__':
    main()
