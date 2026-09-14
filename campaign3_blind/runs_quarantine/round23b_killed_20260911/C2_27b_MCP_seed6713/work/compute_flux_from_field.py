#!/usr/bin/env python3
"""Compute interface flux from field data using finite differences."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import os

work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed6713/work"

# Read field data from side A
field_A = np.loadtxt(os.path.join(work_dir, "side_A/field_level1.csv"), delimiter=',', skiprows=1)
nodes_A = field_A[:, :2]
values_A = field_A[:, 2]

# Read field data from side B  
field_B = np.loadtxt(os.path.join(work_dir, "side_B/field_level1.csv"), delimiter=',', skiprows=1)
nodes_B = field_B[:, :2]
values_B = field_B[:, 2]

# Interface probe points
interface_probes = []
for i in range(44):
    x = 5/8
    y = 1/4 + (i + 0.5) * 1/2/44
    interface_probes.append([x, y])
interface_probes = np.array(interface_probes)

# For side A: compute du/dx at the interface (x=0.625)
# Use nodes near the interface to estimate gradient
k_A = 1.0  # conductivity in subdomain A

# Find nodes in side A near the interface (right edge)
x_max_A = np.max(nodes_A[:, 0])
near_interface_A = nodes_A[np.abs(nodes_A[:, 0] - x_max_A) < 0.01]
values_near_A = values_A[np.abs(nodes_A[:, 0] - x_max_A) < 0.01]

# Create interpolator for side A
if len(np.unique(nodes_A[:, 0])) > 1 and len(np.unique(nodes_A[:, 1])) > 1:
    x_unique = np.sort(np.unique(nodes_A[:, 0]))
    y_unique = np.sort(np.unique(nodes_A[:, 1]))
    
    # Create grid
    u_grid_A = np.zeros((len(y_unique), len(x_unique)))
    for i, x in enumerate(x_unique):
        for j, y in enumerate(y_unique):
            idx = np.where((np.abs(nodes_A[:, 0] - x) < 1e-9) & (np.abs(nodes_A[:, 1] - y) < 1e-9))[0]
            if len(idx) > 0:
                u_grid_A[j, i] = values_A[idx[0]]
    
    interp_A = RegularGridInterpolator((y_unique, x_unique), u_grid_A, bounds_error=False, fill_value=None)
    
    # Compute du/dx at interface by evaluating at x=0.625 and x=0.625-dx
    dx = 0.001
    x_iface = 0.625
    
    qn_A = []
    for (x, y) in interface_probes:
        # Evaluate at interface and slightly inside
        u_at_iface = interp_A((y, x_iface))
        u_inside = interp_A((y, x_iface - dx))
        
        if u_at_iface is not None and u_inside is not None:
            du_dx = (u_at_iface - u_inside) / dx
            # Outward normal for side A is (1, 0) pointing right
            # q_n = -k * grad(u) . n = -k * du/dx
            qn = -k_A * du_dx
            qn_A.append(qn)
        else:
            qn_A.append(0.0)
else:
    qn_A = [0.0] * len(interface_probes)

# For side B: compute du/dx at the interface (x=0.625)
k_B = 200.0  # conductivity in subdomain B

# Find nodes in side B near the interface (left edge)
x_min_B = np.min(nodes_B[:, 0])

# Create interpolator for side B
if len(np.unique(nodes_B[:, 0])) > 1 and len(np.unique(nodes_B[:, 1])) > 1:
    x_unique_B = np.sort(np.unique(nodes_B[:, 0]))
    y_unique_B = np.sort(np.unique(nodes_B[:, 1]))
    
    # Create grid
    u_grid_B = np.zeros((len(y_unique_B), len(x_unique_B)))
    for i, x in enumerate(x_unique_B):
        for j, y in enumerate(y_unique_B):
            idx = np.where((np.abs(nodes_B[:, 0] - x) < 1e-9) & (np.abs(nodes_B[:, 1] - y) < 1e-9))[0]
            if len(idx) > 0:
                u_grid_B[j, i] = values_B[idx[0]]
    
    interp_B = RegularGridInterpolator((y_unique_B, x_unique_B), u_grid_B, bounds_error=False, fill_value=None)
    
    # Compute du/dx at interface
    dx = 0.001
    x_iface = 0.625
    
    qn_B = []
    for (x, y) in interface_probes:
        # Evaluate at interface and slightly inside (to the right for side B)
        u_at_iface = interp_B((y, x_iface))
        u_inside = interp_B((y, x_iface + dx))
        
        if u_at_iface is not None and u_inside is not None:
            du_dx = (u_inside - u_at_iface) / dx
            # Outward normal for side B is (-1, 0) pointing left
            # q_n = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
            qn = k_B * du_dx
            qn_B.append(qn)
        else:
            qn_B.append(0.0)
else:
    qn_B = [0.0] * len(interface_probes)

# Read existing interface data for u values
interface_A = np.loadtxt(os.path.join(work_dir, "side_A/interface_level1.csv"), delimiter=',', skiprows=1)
interface_B = np.loadtxt(os.path.join(work_dir, "side_B/interface_level1.csv"), delimiter=',', skiprows=1)

# Interpolate u values to interface probes (1D along y)
from scipy.interpolate import interp1d

iface_y_A = interface_A[:, 1]
iface_u_A = interface_A[:, 2]
idx_A = np.argsort(iface_y_A)
f_u_A = interp1d(iface_y_A[idx_A], iface_u_A[idx_A], kind='linear', fill_value="extrapolate")

iface_y_B = interface_B[:, 1]
iface_u_B = interface_B[:, 2]
idx_B = np.argsort(iface_y_B)
f_u_B = interp1d(iface_y_B[idx_B], iface_u_B[idx_B], kind='linear', fill_value="extrapolate")

iface_y_probes = interface_probes[:, 1]
iface_u_A_interp = f_u_A(iface_y_probes)
iface_u_B_interp = f_u_B(iface_y_probes)

# Write updated interface files with computed fluxes
with open(os.path.join(work_dir, "interface_level1_A.csv"), 'w') as f:
    f.write("x,y,u,qn\n")
    for i, (x, y) in enumerate(interface_probes):
        f.write(f"{x:.11e},{y:.11e},{iface_u_A_interp[i]:.11e},{qn_A[i]:.11e}\n")

with open(os.path.join(work_dir, "interface_level1_B.csv"), 'w') as f:
    f.write("x,y,u,qn\n")
    for i, (x, y) in enumerate(interface_probes):
        f.write(f"{x:.11e},{y:.11e},{iface_u_B_interp[i]:.11e},{qn_B[i]:.11e}\n")

print("Computed interface fluxes from field gradients")
print(f"Side A flux range: [{min(qn_A):.6e}, {max(qn_A):.6e}]")
print(f"Side B flux range: [{min(qn_B):.6e}, {max(qn_B):.6e}]")
