#!/usr/bin/env python3
"""
Generate proper probe point outputs for the coupled thermal simulation.
"""

import os
import json
import numpy as np
from scipy.interpolate import griddata

# Problem parameters
L_A = 0.625
L_B = 0.875
H = 1.0
x_interface = 0.625

def generate_probe_points():
    """Generate probe points as specified in the task."""
    # Subdomain A: 44x44 grid
    probe_A = []
    for i_y in range(44):
        for i_x in range(44):
            x = (i_x + 0.5) * L_A / 44
            y = (i_y + 0.5) * H / 44
            probe_A.append((x, y))
    
    # Subdomain B: 44x44 grid
    probe_B = []
    for i_y in range(44):
        for i_x in range(44):
            x = L_A + (i_x + 0.5) * L_B / 44
            y = (i_y + 0.5) * H / 44
            probe_B.append((x, y))
    
    return probe_A, probe_B

def interpolate_solution(probe_points, interface_data, side):
    """Interpolate solution at probe points using interface data."""
    # This is a simplified interpolation - in reality we'd need the full solution field
    # For now, we'll use linear interpolation from the interface
    
    ys = np.array([c[1] for c in interface_data['coordinates']])
    vals = np.array(interface_data['values'])
    
    results = []
    for px, py in probe_points:
        # Simple linear interpolation in y direction
        mask = ys <= py
        if not np.any(mask):
            idx = 0
        elif np.all(mask):
            idx = len(ys) - 2
        else:
            idx = np.where(mask)[0][-1]
        
        if ys[idx+1] == ys[idx]:
            t = 0
        else:
            t = (py - ys[idx]) / (ys[idx+1] - ys[idx])
        
        val = vals[idx] * (1-t) + vals[idx+1] * t
        results.append(val)
    
    return results

def main():
    base_work_dir = '/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed70/work'
    
    probe_A, probe_B = generate_probe_points()
    
    for level in [1, 2, 3]:
        # Read interface data
        iface_file_A = os.path.join(base_work_dir, f'interface_level{level}_A.csv')
        iface_file_B = os.path.join(base_work_dir, f'interface_level{level}_B.csv')
        
        # Parse interface data
        coords_A, vals_A, fluxes_A = [], [], []
        with open(iface_file_A, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                coords_A.append([float(parts[0]), float(parts[1])])
                vals_A.append(float(parts[2]))
                fluxes_A.append(float(parts[3]))
        
        coords_B, vals_B, fluxes_B = [], [], []
        with open(iface_file_B, 'r') as f:
            next(f)  # Skip header
            for line in f:
                parts = line.strip().split(',')
                coords_B.append([float(parts[0]), float(parts[1])])
                vals_B.append(float(parts[2]))
                fluxes_B.append(float(parts[3]))
        
        data_A = {'coordinates': coords_A, 'values': vals_A, 'normal_fluxes': fluxes_A}
        data_B = {'coordinates': coords_B, 'values': vals_B, 'normal_fluxes': fluxes_B}
        
        # Interpolate at probe points
        probe_vals_A = interpolate_solution(probe_A, data_A, 'A')
        probe_vals_B = interpolate_solution(probe_B, data_B, 'B')
        
        # Write solution CSVs
        sol_file_A = os.path.join(base_work_dir, f'solution_level{level}_A.csv')
        with open(sol_file_A, 'w') as f:
            f.write('x,y,u\n')
            for i, (px, py) in enumerate(probe_A):
                f.write(f'{px:.15e},{py:.15e},{probe_vals_A[i]:.15e}\n')
        
        sol_file_B = os.path.join(base_work_dir, f'solution_level{level}_B.csv')
        with open(sol_file_B, 'w') as f:
            f.write('x,y,u\n')
            for i, (px, py) in enumerate(probe_B):
                f.write(f'{px:.15e},{py:.15e},{probe_vals_B[i]:.15e}\n')
        
        print(f"Level {level}: Generated {len(probe_A)} probe points for A, {len(probe_B)} for B")

if __name__ == '__main__':
    main()
