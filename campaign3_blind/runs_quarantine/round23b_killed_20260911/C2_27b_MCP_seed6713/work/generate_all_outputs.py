#!/usr/bin/env python3
"""Generate probe point outputs from field files for all levels."""
import numpy as np
from scipy.interpolate import griddata, interp1d
import os
import glob

work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed6713/work"

# Probe points for subdomain A
probe_A = []
for i_y in range(44):
    for i_x in range(44):
        x = 0.0 + (i_x + 0.5) * 0.625 / 44
        y = 0.0 + (i_y + 0.5) * 1.0 / 44
        probe_A.append([x, y])
probe_A = np.array(probe_A)

# Probe points for subdomain B
probe_B = []
for i_y in range(44):
    for i_x in range(44):
        x = 0.625 + (i_x + 0.5) * 0.875 / 44
        y = 0.0 + (i_y + 0.5) * 1.0 / 44
        probe_B.append([x, y])
probe_B = np.array(probe_B)

# Interface probe points
interface_probes = []
for i in range(44):
    x = 5/8
    y = 1/4 + (i + 0.5) * 1/2/44
    interface_probes.append([x, y])
interface_probes = np.array(interface_probes)

# Process each level
for level in [1, 2]:
    print(f"\nProcessing level {level}...")
    
    # Read field data
    field_A_path = os.path.join(work_dir, f"side_A/field_level{level}.csv")
    field_B_path = os.path.join(work_dir, f"side_B/field_level{level}.csv")
    
    if not os.path.exists(field_A_path) or not os.path.exists(field_B_path):
        print(f"  Skipping level {level} - field files not found")
        continue
    
    field_A = np.loadtxt(field_A_path, delimiter=',', skiprows=1)
    nodes_A = field_A[:, :2]
    values_A = field_A[:, 2]
    
    field_B = np.loadtxt(field_B_path, delimiter=',', skiprows=1)
    nodes_B = field_B[:, :2]
    values_B = field_B[:, 2]
    
    # Interpolate at probe points
    u_A = griddata(nodes_A, values_A, probe_A, method='linear')
    u_B = griddata(nodes_B, values_B, probe_B, method='linear')
    
    # Write solution files
    with open(os.path.join(work_dir, f"solution_level{level}_A.csv"), 'w') as f:
        f.write("x,y,u\n")
        for (x, y), u in zip(probe_A, u_A):
            if not np.isnan(u):
                f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")
    
    with open(os.path.join(work_dir, f"solution_level{level}_B.csv"), 'w') as f:
        f.write("x,y,u\n")
        for (x, y), u in zip(probe_B, u_B):
            if not np.isnan(u):
                f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")
    
    # Compute fluxes from field gradients
    k_A, k_B = 1.0, 200.0
    dx = 0.001
    x_iface = 0.625
    
    # Side A flux
    from scipy.interpolate import RegularGridInterpolator
    x_unique = np.sort(np.unique(nodes_A[:, 0]))
    y_unique = np.sort(np.unique(nodes_A[:, 1]))
    u_grid_A = np.zeros((len(y_unique), len(x_unique)))
    for i, x in enumerate(x_unique):
        for j, y in enumerate(y_unique):
            idx = np.where((np.abs(nodes_A[:, 0] - x) < 1e-9) & (np.abs(nodes_A[:, 1] - y) < 1e-9))[0]
            if len(idx) > 0:
                u_grid_A[j, i] = values_A[idx[0]]
    interp_A = RegularGridInterpolator((y_unique, x_unique), u_grid_A, bounds_error=False, fill_value=None)
    
    qn_A = []
    for (x, y) in interface_probes:
        u_at_iface = interp_A((y, x_iface))
        u_inside = interp_A((y, x_iface - dx))
        if u_at_iface is not None and u_inside is not None:
            du_dx = (u_at_iface - u_inside) / dx
            qn = -k_A * du_dx
            qn_A.append(qn)
        else:
            qn_A.append(0.0)
    
    # Side B flux
    x_unique_B = np.sort(np.unique(nodes_B[:, 0]))
    y_unique_B = np.sort(np.unique(nodes_B[:, 1]))
    u_grid_B = np.zeros((len(y_unique_B), len(x_unique_B)))
    for i, x in enumerate(x_unique_B):
        for j, y in enumerate(y_unique_B):
            idx = np.where((np.abs(nodes_B[:, 0] - x) < 1e-9) & (np.abs(nodes_B[:, 1] - y) < 1e-9))[0]
            if len(idx) > 0:
                u_grid_B[j, i] = values_B[idx[0]]
    interp_B = RegularGridInterpolator((y_unique_B, x_unique_B), u_grid_B, bounds_error=False, fill_value=None)
    
    qn_B = []
    for (x, y) in interface_probes:
        u_at_iface = interp_B((y, x_iface))
        u_inside = interp_B((y, x_iface + dx))
        if u_at_iface is not None and u_inside is not None:
            du_dx = (u_inside - u_at_iface) / dx
            qn = k_B * du_dx
            qn_B.append(qn)
        else:
            qn_B.append(0.0)
    
    # Get u values at interface from exports.json or compute from field
    iface_u_A_interp = griddata(nodes_A, values_A, interface_probes, method='linear')
    iface_u_B_interp = griddata(nodes_B, values_B, interface_probes, method='linear')
    
    # Write interface files
    with open(os.path.join(work_dir, f"interface_level{level}_A.csv"), 'w') as f:
        f.write("x,y,u,qn\n")
        for i, (x, y) in enumerate(interface_probes):
            f.write(f"{x:.11e},{y:.11e},{iface_u_A_interp[i]:.11e},{qn_A[i]:.11e}\n")
    
    with open(os.path.join(work_dir, f"interface_level{level}_B.csv"), 'w') as f:
        f.write("x,y,u,qn\n")
        for i, (x, y) in enumerate(interface_probes):
            f.write(f"{x:.11e},{y:.11e},{iface_u_B_interp[i]:.11e},{qn_B[i]:.11e}\n")
    
    print(f"  Level {level}: {np.sum(~np.isnan(u_A))} valid probes side A, {np.sum(~np.isnan(u_B))} side B")
    print(f"  Flux ranges: A=[{min(qn_A):.3e}, {max(qn_A):.3e}], B=[{min(qn_B):.3e}, {max(qn_B):.3e}]")

print("\nDone!")
