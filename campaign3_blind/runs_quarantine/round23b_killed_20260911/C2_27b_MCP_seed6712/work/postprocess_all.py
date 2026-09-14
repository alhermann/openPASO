#!/usr/bin/env python3
"""Post-process all levels to write files at prescribed probe points."""
import numpy as np
from scipy.interpolate import griddata
import json
import os

def get_probe_points_A():
    """Prescribed probe points for subdomain A."""
    probes = []
    for iy in range(44):
        for ix in range(44):
            px = 0.0 + (ix + 0.5) * 0.625 / 44
            py = 0.0 + (iy + 0.5) * 1.0 / 44
            probes.append((px, py))
    return np.array(probes)

def get_probe_points_B():
    """Prescribed probe points for subdomain B."""
    probes = []
    for iy in range(44):
        for ix in range(44):
            px = 0.625 + (ix + 0.5) * 0.875 / 44
            py = 0.0 + (iy + 0.5) * 1.0 / 44
            probes.append((px, py))
    return np.array(probes)

def get_interface_probes():
    """Prescribed interface probe points."""
    probes = []
    for i in range(44):
        px = 5/8
        py = 1/4 + (i + 0.5) * (1/2) / 44
        probes.append((px, py))
    return np.array(probes)

probe_A = get_probe_points_A()
probe_B = get_probe_points_B()
interface_probes = get_interface_probes()

for LEVEL in [1, 2, 3]:
    print(f"Processing level {LEVEL}...")
    
    # Read side A field data from participant output
    a_field_file = f'side_A/field_level{LEVEL}_A.csv'
    if not os.path.exists(a_field_file):
        print(f"  Warning: {a_field_file} not found")
        continue
    
    a_data = np.loadtxt(a_field_file, delimiter=',', skiprows=1)
    a_coords = a_data[:, :2]
    a_values = a_data[:, 2]
    
    # Read side B field data
    b_field_file = f'side_B/field_level{LEVEL}_B.csv'
    if not os.path.exists(b_field_file):
        print(f"  Warning: {b_field_file} not found")
        continue
    
    b_data = np.loadtxt(b_field_file, delimiter=',', skiprows=1)
    b_coords = b_data[:, :2]
    b_values = b_data[:, 2]
    
    # Interpolate side A solution at probe points
    u_A = griddata(a_coords, a_values, probe_A, method='linear')
    
    # Write field_level{LEVEL}_A.csv with prescribed probes
    with open(f'field_level{LEVEL}_A.csv', 'w') as f:
        f.write('x,y,u\n')
        for (px, py), u_val in zip(probe_A, u_A):
            if np.isnan(u_val):
                print(f"  Warning: NaN at {px}, {py}")
            f.write(f'{px:.11e},{py:.11e},{u_val:.11e}\n')
    
    # Interpolate side B solution at probe points
    u_B = griddata(b_coords, b_values, probe_B, method='linear')
    
    # Write field_level{LEVEL}_B.csv with prescribed probes
    with open(f'field_level{LEVEL}_B.csv', 'w') as f:
        f.write('x,y,u\n')
        for (px, py), u_val in zip(probe_B, u_B):
            if np.isnan(u_val):
                print(f"  Warning: NaN at {px}, {py}")
            f.write(f'{px:.11e},{py:.11e},{u_val:.11e}\n')
    
    # For interface files, read exports.json from both sides
    exp_A_file = f'side_A/exports.json'
    exp_B_file = f'side_B/exports.json'
    
    if os.path.exists(exp_A_file) and os.path.exists(exp_B_file):
        with open(exp_A_file) as f:
            exp_A = json.load(f)
        with open(exp_B_file) as f:
            exp_B = json.load(f)
        
        # Side A exports fluxes at its interface nodes
        a_iface_coords = np.array(exp_A['coordinates'])
        a_iface_fluxes = np.array(exp_A['normal_fluxes'])
        
        # Side B exports values and fluxes at its interface nodes
        b_iface_coords = np.array(exp_B['coordinates'])
        b_iface_values = np.array(exp_B['values'])
        b_iface_fluxes = np.array(exp_B['normal_fluxes'])
        
        # Interpolate side A fluxes at prescribed interface probes
        q_A = griddata(a_iface_coords[:, 1], a_iface_fluxes, interface_probes[:, 1], method='linear')
        
        # Interpolate side B values and fluxes at prescribed interface probes
        u_B_iface = griddata(b_iface_coords[:, 1], b_iface_values, interface_probes[:, 1], method='linear')
        q_B = griddata(b_iface_coords[:, 1], b_iface_fluxes, interface_probes[:, 1], method='linear')
        
        # Also need side A values at interface - interpolate from field data
        u_A_iface = griddata(a_coords, a_values, interface_probes, method='linear')
        
        # Write interface_level{LEVEL}_A.csv
        with open(f'interface_level{LEVEL}_A.csv', 'w') as f:
            f.write('x,y,u,qn\n')
            for (px, py), u_val, q_val in zip(interface_probes, u_A_iface, q_A):
                f.write(f'{px:.11e},{py:.11e},{u_val:.11e},{q_val:.11e}\n')
        
        # Write interface_level{LEVEL}_B.csv
        with open(f'interface_level{LEVEL}_B.csv', 'w') as f:
            f.write('x,y,u,qn\n')
            for (px, py), u_val, q_val in zip(interface_probes, u_B_iface, q_B):
                f.write(f'{px:.11e},{py:.11e},{u_val:.11e},{q_val:.11e}\n')
        
        # Check interface agreement
        rel_jump = np.abs(u_A_iface - u_B_iface).max() / max(np.abs(u_A_iface).max(), np.abs(u_B_iface).max())
        flux_balance = np.abs(q_A + q_B).max() / max(np.abs(q_A).max(), np.abs(q_B).max())
        print(f"  Level {LEVEL}: rel field jump = {rel_jump:.6e}, flux balance = {flux_balance:.6e}")
    
    print(f"  Level {LEVEL} complete")

print("All levels processed")
