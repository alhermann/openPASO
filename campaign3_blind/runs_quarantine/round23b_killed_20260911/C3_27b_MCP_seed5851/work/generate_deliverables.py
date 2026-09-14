#!/usr/bin/env python3
"""Generate deliverable files for level 1."""

import numpy as np
from scipy.interpolate import griddata, interp1d
import os

work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C3_27b_MCP_seed5851/work"

# Step 1: Generate probe points for subdomain A (1936 points)
# x = (i_x+0.5)/44, y = (i_y+0.5)*0.625/44 for i_x, i_y = 0..43
# Order: last index (i_y) varies fastest
probe_A = []
for i_x in range(44):
    for i_y in range(44):
        x = (i_x + 0.5) / 44.0
        y = (i_y + 0.5) * 0.625 / 44.0
        probe_A.append((x, y))
print(f"Generated {len(probe_A)} probe points for subdomain A")

# Step 2: Generate probe points for subdomain B (1936 points)
# x = (i_x+0.5)/44, y = 0.625 + (i_y+0.5)*0.875/44 for i_x, i_y = 0..43
# Order: last index (i_y) varies fastest
probe_B = []
for i_x in range(44):
    for i_y in range(44):
        x = (i_x + 0.5) / 44.0
        y = 0.625 + (i_y + 0.5) * 0.875 / 44.0
        probe_B.append((x, y))
print(f"Generated {len(probe_B)} probe points for subdomain B")

# Step 3: Generate interface probe points (44 points)
# x = 1/4 + (i+0.5)*1/2/44, y = 5/8 for i = 0..43
interface_probe = []
for i in range(44):
    x = 1/4 + (i + 0.5) * (1/2) / 44.0
    y = 5/8
    interface_probe.append((x, y))
print(f"Generated {len(interface_probe)} interface probe points")

# Step 4: Read field_level1.csv from side_A and side_B, interpolate to probe points
def read_field_csv(filepath):
    """Read field CSV file and return arrays of x, y, u."""
    x_vals = []
    y_vals = []
    u_vals = []
    with open(filepath, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
                u_vals.append(float(parts[2]))
    return np.array(x_vals), np.array(y_vals), np.array(u_vals)

# Read side_A field data
x_A, y_A, u_A = read_field_csv(os.path.join(work_dir, "side_A/field_level1.csv"))
print(f"Read {len(x_A)} nodes from side_A field")

# Read side_B field data
x_B, y_B, u_B = read_field_csv(os.path.join(work_dir, "side_B/field_level1.csv"))
print(f"Read {len(x_B)} nodes from side_B field")

# Interpolate to probe points for subdomain A
points_A = np.column_stack([x_A, y_A])
u_at_probe_A = griddata(points_A, u_A, probe_A, method='linear')

# Write solution_level1_A.csv
with open(os.path.join(work_dir, "solution_level1_A.csv"), 'w') as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(probe_A, u_at_probe_A):
        f.write(f"{x:.15e},{y:.15e},{u:.15e}\n")
print("Written solution_level1_A.csv")

# Interpolate to probe points for subdomain B
points_B = np.column_stack([x_B, y_B])
u_at_probe_B = griddata(points_B, u_B, probe_B, method='linear')

# Write solution_level1_B.csv
with open(os.path.join(work_dir, "solution_level1_B.csv"), 'w') as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(probe_B, u_at_probe_B):
        f.write(f"{x:.15e},{y:.15e},{u:.15e}\n")
print("Written solution_level1_B.csv")

# Step 5: Read interface data and interpolate to interface probe points
def read_interface_csv(filepath):
    """Read interface CSV file and return arrays of x, y, u, qn."""
    x_vals = []
    y_vals = []
    u_vals = []
    qn_vals = []
    with open(filepath, 'r') as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 4:
                x_vals.append(float(parts[0]))
                y_vals.append(float(parts[1]))
                u_vals.append(float(parts[2]))
                qn_vals.append(float(parts[3]))
    return np.array(x_vals), np.array(y_vals), np.array(u_vals), np.array(qn_vals)

# Read side_A interface data
xi_A, yi_A, ui_A, qni_A = read_interface_csv(os.path.join(work_dir, "side_A/interface_level1.csv"))
print(f"Read {len(xi_A)} interface nodes from side_A")

# Read side_B interface data
xi_B, yi_B, ui_B, qni_B = read_interface_csv(os.path.join(work_dir, "side_B/interface_level1.csv"))
print(f"Read {len(xi_B)} interface nodes from side_B")

# Interface points are all on y=0.625, so we use 1D interpolation along x
# Sort by x coordinate
idx_A = np.argsort(xi_A)
xi_A_sorted = xi_A[idx_A]
ui_A_sorted = ui_A[idx_A]
qni_A_sorted = qni_A[idx_A]

idx_B = np.argsort(xi_B)
xi_B_sorted = xi_B[idx_B]
ui_B_sorted = ui_B[idx_B]
qni_B_sorted = qni_B[idx_B]

# Create 1D interpolators for side_A
interp_u_A = interp1d(xi_A_sorted, ui_A_sorted, kind='linear', fill_value='extrapolate')
interp_qn_A = interp1d(xi_A_sorted, qni_A_sorted, kind='linear', fill_value='extrapolate')

# Create 1D interpolators for side_B
interp_u_B = interp1d(xi_B_sorted, ui_B_sorted, kind='linear', fill_value='extrapolate')
interp_qn_B = interp1d(xi_B_sorted, qni_B_sorted, kind='linear', fill_value='extrapolate')

# Extract x coordinates from interface probe points
interface_x = [p[0] for p in interface_probe]
interface_y = [p[1] for p in interface_probe]

# Interpolate at interface probe points
u_at_iface_A = interp_u_A(interface_x)
qn_at_iface_A = interp_qn_A(interface_x)

u_at_iface_B = interp_u_B(interface_x)
qn_at_iface_B = interp_qn_B(interface_x)

# Write interface_level1_A.csv
with open(os.path.join(work_dir, "interface_level1_A.csv"), 'w') as f:
    f.write("x,y,u,qn\n")
    for (x, y), u, qn in zip(interface_probe, u_at_iface_A, qn_at_iface_A):
        f.write(f"{x:.15e},{y:.15e},{u:.15e},{qn:.15e}\n")
print("Written interface_level1_A.csv")

# Write interface_level1_B.csv
with open(os.path.join(work_dir, "interface_level1_B.csv"), 'w') as f:
    f.write("x,y,u,qn\n")
    for (x, y), u, qn in zip(interface_probe, u_at_iface_B, qn_at_iface_B):
        f.write(f"{x:.15e},{y:.15e},{u:.15e},{qn:.15e}\n")
print("Written interface_level1_B.csv")

# Step 6: Copy participant_output.log and add NDOF line at beginning
# For side_A: NDOF = 54
with open(os.path.join(work_dir, "side_A/participant_output.log"), 'r') as f:
    content_A = f.read()
with open(os.path.join(work_dir, "run_level1_A.log"), 'w') as f:
    f.write("NDOF = 54\n")
    f.write(content_A)
print("Written run_level1_A.log")

# For side_B: NDOF = 72
with open(os.path.join(work_dir, "side_B/participant_output.log"), 'r') as f:
    content_B = f.read()
with open(os.path.join(work_dir, "run_level1_B.log"), 'w') as f:
    f.write("NDOF = 72\n")
    f.write(content_B)
print("Written run_level1_B.log")

print("\nAll deliverables generated successfully!")
