#!/usr/bin/env python3
"""Generate probe point outputs from field files."""
import numpy as np
from scipy.interpolate import griddata, interp1d
import os

work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C2_27b_MCP_seed6713/work"

# Probe points for subdomain A: x = 0 + (i_x+0.5)*0.625/44; y = 0 + (i_y+0.5)*1/44
probe_A = []
for i_y in range(44):
    for i_x in range(44):
        x = 0.0 + (i_x + 0.5) * 0.625 / 44
        y = 0.0 + (i_y + 0.5) * 1.0 / 44
        probe_A.append([x, y])
probe_A = np.array(probe_A)

# Probe points for subdomain B: x = 0.625 + (i_x+0.5)*0.875/44; y = 0 + (i_y+0.5)*1/44
probe_B = []
for i_y in range(44):
    for i_x in range(44):
        x = 0.625 + (i_x + 0.5) * 0.875 / 44
        y = 0.0 + (i_y + 0.5) * 1.0 / 44
        probe_B.append([x, y])
probe_B = np.array(probe_B)

# Interface probe points: (x, y) = (5/8, 1/4 + (i+0.5)*1/2/44) for i = 0, ..., 43
interface_probes = []
for i in range(44):
    x = 5/8
    y = 1/4 + (i + 0.5) * 1/2/44
    interface_probes.append([x, y])
interface_probes = np.array(interface_probes)

# Read field data from side A
field_A = np.loadtxt(os.path.join(work_dir, "side_A/field_level1.csv"), delimiter=',', skiprows=1)
nodes_A = field_A[:, :2]
values_A = field_A[:, 2]

# Read field data from side B
field_B = np.loadtxt(os.path.join(work_dir, "side_B/field_level1.csv"), delimiter=',', skiprows=1)
nodes_B = field_B[:, :2]
values_B = field_B[:, 2]

# Interpolate at probe points
u_A = griddata(nodes_A, values_A, probe_A, method='linear')
u_B = griddata(nodes_B, values_B, probe_B, method='linear')

# Write solution files
with open(os.path.join(work_dir, "solution_level1_A.csv"), 'w') as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(probe_A, u_A):
        if not np.isnan(u):
            f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")

with open(os.path.join(work_dir, "solution_level1_B.csv"), 'w') as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(probe_B, u_B):
        if not np.isnan(u):
            f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")

# Read interface data
interface_A = np.loadtxt(os.path.join(work_dir, "side_A/interface_level1.csv"), delimiter=',', skiprows=1)
interface_B = np.loadtxt(os.path.join(work_dir, "side_B/interface_level1.csv"), delimiter=',', skiprows=1)

# For interface, use 1D interpolation along y (all points have x=0.625)
iface_y_A = interface_A[:, 1]
iface_u_A = interface_A[:, 2]
iface_q_A = interface_A[:, 3]

iface_y_B = interface_B[:, 1]
iface_u_B = interface_B[:, 2]
iface_q_B = interface_B[:, 3]

iface_y_probes = interface_probes[:, 1]

# Sort by y and interpolate
idx_A = np.argsort(iface_y_A)
f_u_A = interp1d(iface_y_A[idx_A], iface_u_A[idx_A], kind='linear', fill_value="extrapolate")
f_q_A = interp1d(iface_y_A[idx_A], iface_q_A[idx_A], kind='linear', fill_value="extrapolate")

idx_B = np.argsort(iface_y_B)
f_u_B = interp1d(iface_y_B[idx_B], iface_u_B[idx_B], kind='linear', fill_value="extrapolate")
f_q_B = interp1d(iface_y_B[idx_B], iface_q_B[idx_B], kind='linear', fill_value="extrapolate")

iface_u_A_interp = f_u_A(iface_y_probes)
iface_q_A_interp = f_q_A(iface_y_probes)
iface_u_B_interp = f_u_B(iface_y_probes)
iface_q_B_interp = f_q_B(iface_y_probes)

# Write interface files
with open(os.path.join(work_dir, "interface_level1_A.csv"), 'w') as f:
    f.write("x,y,u,qn\n")
    for (x, y), u, q in zip(interface_probes, iface_u_A_interp, iface_q_A_interp):
        f.write(f"{x:.11e},{y:.11e},{u:.11e},{q:.11e}\n")

with open(os.path.join(work_dir, "interface_level1_B.csv"), 'w') as f:
    f.write("x,y,u,qn\n")
    for (x, y), u, q in zip(interface_probes, iface_u_B_interp, iface_q_B_interp):
        f.write(f"{x:.11e},{y:.11e},{u:.11e},{q:.11e}\n")

print("Generated probe point outputs for level 1")
print(f"Side A: {np.sum(~np.isnan(u_A))} valid probe points out of {len(u_A)}")
print(f"Side B: {np.sum(~np.isnan(u_B))} valid probe points out of {len(u_B)}")
