#!/usr/bin/env python3
"""Generate level 2 deliverable files."""

import numpy as np
from scipy.interpolate import griddata, interp1d
import os

work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C3_27b_MCP_seed5851/work"

# Read field data for side A
field_A_data = []
with open(os.path.join(work_dir, "side_A", "field_level2.csv")) as f:
    header = f.readline()  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 3:
            x, y, u = float(parts[0]), float(parts[1]), float(parts[2])
            field_A_data.append((x, y, u))

# Read field data for side B
field_B_data = []
with open(os.path.join(work_dir, "side_B", "field_level2.csv")) as f:
    header = f.readline()  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 3:
            x, y, u = float(parts[0]), float(parts[1]), float(parts[2])
            field_B_data.append((x, y, u))

print(f"Loaded {len(field_A_data)} points from side_A field")
print(f"Loaded {len(field_B_data)} points from side_B field")

# Generate probe points for subdomain A (1936 points)
# x = (i_x+0.5)/44, y = (i_y+0.5)*0.625/44 for i_x, i_y = 0..43
# Order: last index (i_y) varies fastest
probe_points_A = []
for i_x in range(44):
    for i_y in range(44):
        x = (i_x + 0.5) / 44.0
        y = (i_y + 0.5) * 0.625 / 44.0
        probe_points_A.append((x, y))

print(f"Generated {len(probe_points_A)} probe points for subdomain A")

# Generate probe points for subdomain B (1936 points)
# x = (i_x+0.5)/44, y = 0.625 + (i_y+0.5)*0.875/44 for i_x, i_y = 0..43
# Order: last index (i_y) varies fastest
probe_points_B = []
for i_x in range(44):
    for i_y in range(44):
        x = (i_x + 0.5) / 44.0
        y = 0.625 + (i_y + 0.5) * 0.875 / 44.0
        probe_points_B.append((x, y))

print(f"Generated {len(probe_points_B)} probe points for subdomain B")

# Interpolate field A to probe points A using linear interpolation
points_A = np.array([[p[0], p[1]] for p in field_A_data])
values_A = np.array([p[2] for p in field_A_data])

probe_coords_A = np.array(probe_points_A)
u_probe_A = griddata(points_A, values_A, probe_coords_A, method='linear')

# Write solution_level2_A.csv
with open(os.path.join(work_dir, "solution_level2_A.csv"), "w") as f:
    f.write("x,y,u\n")
    for i, (x, y) in enumerate(probe_points_A):
        f.write(f"{x:.15e},{y:.15e},{u_probe_A[i]:.15e}\n")

print("Written solution_level2_A.csv")

# Interpolate field B to probe points B using linear interpolation
points_B = np.array([[p[0], p[1]] for p in field_B_data])
values_B = np.array([p[2] for p in field_B_data])

probe_coords_B = np.array(probe_points_B)
u_probe_B = griddata(points_B, values_B, probe_coords_B, method='linear')

# Write solution_level2_B.csv
with open(os.path.join(work_dir, "solution_level2_B.csv"), "w") as f:
    f.write("x,y,u\n")
    for i, (x, y) in enumerate(probe_points_B):
        f.write(f"{x:.15e},{y:.15e},{u_probe_B[i]:.15e}\n")

print("Written solution_level2_B.csv")

# Read interface data for side A
interface_A_data = []
with open(os.path.join(work_dir, "side_A", "interface_level2.csv")) as f:
    header = f.readline()  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 4:
            x, y, u, qn = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            interface_A_data.append((x, y, u, qn))

print(f"Loaded {len(interface_A_data)} points from side_A interface")

# Read interface data for side B
interface_B_data = []
with open(os.path.join(work_dir, "side_B", "interface_level2.csv")) as f:
    header = f.readline()  # skip header
    for line in f:
        parts = line.strip().split(",")
        if len(parts) == 4:
            x, y, u, qn = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
            interface_B_data.append((x, y, u, qn))

print(f"Loaded {len(interface_B_data)} points from side_B interface")

# Generate interface probe points (44 points)
# x = 1/4 + (i+0.5)*1/2/44, y = 5/8 for i = 0..43
interface_probe_points = []
for i in range(44):
    x = 1/4 + (i + 0.5) * (1/2) / 44.0
    y = 5/8
    interface_probe_points.append((x, y))

print(f"Generated {len(interface_probe_points)} interface probe points")

# Interface data is 1D (all points have y=0.625), so use 1D interpolation
# Sort by x coordinate
interface_A_sorted = sorted(interface_A_data, key=lambda p: p[0])
interface_B_sorted = sorted(interface_B_data, key=lambda p: p[0])

x_interface_A = np.array([p[0] for p in interface_A_sorted])
u_interface_A = np.array([p[2] for p in interface_A_sorted])
qn_interface_A = np.array([p[3] for p in interface_A_sorted])

x_interface_B = np.array([p[0] for p in interface_B_sorted])
u_interface_B = np.array([p[2] for p in interface_B_sorted])
qn_interface_B = np.array([p[3] for p in interface_B_sorted])

x_probe = np.array([p[0] for p in interface_probe_points])

# Create 1D interpolators with linear method
interp_u_A = interp1d(x_interface_A, u_interface_A, kind='linear', fill_value='extrapolate')
interp_qn_A = interp1d(x_interface_A, qn_interface_A, kind='linear', fill_value='extrapolate')

interp_u_B = interp1d(x_interface_B, u_interface_B, kind='linear', fill_value='extrapolate')
interp_qn_B = interp1d(x_interface_B, qn_interface_B, kind='linear', fill_value='extrapolate')

u_interface_probe_A = interp_u_A(x_probe)
qn_interface_probe_A = interp_qn_A(x_probe)

u_interface_probe_B = interp_u_B(x_probe)
qn_interface_probe_B = interp_qn_B(x_probe)

# Write interface_level2_A.csv
with open(os.path.join(work_dir, "interface_level2_A.csv"), "w") as f:
    f.write("x,y,u,qn\n")
    for i, (x, y) in enumerate(interface_probe_points):
        f.write(f"{x:.15e},{y:.15e},{u_interface_probe_A[i]:.15e},{qn_interface_probe_A[i]:.15e}\n")

print("Written interface_level2_A.csv")

# Write interface_level2_B.csv
with open(os.path.join(work_dir, "interface_level2_B.csv"), "w") as f:
    f.write("x,y,u,qn\n")
    for i, (x, y) in enumerate(interface_probe_points):
        f.write(f"{x:.15e},{y:.15e},{u_interface_probe_B[i]:.15e},{qn_interface_probe_B[i]:.15e}\n")

print("Written interface_level2_B.csv")

# Copy participant_output.log from side_A and add NDOF line at beginning
with open(os.path.join(work_dir, "side_A", "participant_output.log")) as f:
    content_A = f.read()

# Add NDOF line at the beginning
log_A = "NDOF = 187\n" + content_A
with open(os.path.join(work_dir, "run_level2_A.log"), "w") as f:
    f.write(log_A)

print("Written run_level2_A.log")

# Copy participant_output.log from side_B and add NDOF line at beginning
with open(os.path.join(work_dir, "side_B", "participant_output.log")) as f:
    content_B = f.read()

# Add NDOF line at the beginning
log_B = "NDOF = 255\n" + content_B
with open(os.path.join(work_dir, "run_level2_B.log"), "w") as f:
    f.write(log_B)

print("Written run_level2_B.log")

print("\nAll files generated successfully!")
