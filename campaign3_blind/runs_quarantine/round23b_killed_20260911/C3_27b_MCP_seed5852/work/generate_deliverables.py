#!/usr/bin/env python3
"""Generate all level 1 deliverable files."""

import numpy as np
from scipy.interpolate import griddata, interp1d

# Read side_A field data
def read_csv(filepath):
    """Read CSV file and return arrays."""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split(',')
    data_lines = [l.strip().split(',') for l in lines[1:] if l.strip()]
    
    x = []
    y = []
    u = []
    qn = []
    
    for row in data_lines:
        x.append(float(row[0]))
        y.append(float(row[1]))
        u.append(float(row[2]))
        if len(row) > 3:
            qn.append(float(row[3]))
    
    return np.array(x), np.array(y), np.array(u), np.array(qn) if qn else None

# Generate solution probe points for subdomain A
def generate_solution_probes_A():
    """Generate 1936 solution probe points for subdomain A."""
    probe_x = []
    probe_y = []
    for i_x in range(44):
        for i_y in range(44):
            px = (i_x + 0.5) / 44
            py = (i_y + 0.5) * 0.625 / 44
            probe_x.append(px)
            probe_y.append(py)
    return np.array(probe_x), np.array(probe_y)

# Generate solution probe points for subdomain B
def generate_solution_probes_B():
    """Generate 1936 solution probe points for subdomain B."""
    probe_x = []
    probe_y = []
    for i_x in range(44):
        for i_y in range(44):
            px = (i_x + 0.5) / 44
            py = 0.625 + (i_y + 0.5) * 0.875 / 44
            probe_x.append(px)
            probe_y.append(py)
    return np.array(probe_x), np.array(probe_y)

# Generate interface probe points
def generate_interface_probes():
    """Generate 44 interface probe points."""
    probe_x = []
    probe_y = []
    for i in range(44):
        px = 1/4 + (i + 0.5) * (1/2) / 44
        py = 5/8
        probe_x.append(px)
        probe_y.append(py)
    return np.array(probe_x), np.array(probe_y)

# Interpolate on 1D line (interface data)
def interpolate_1d(x_data, y_data, x_probe):
    """Interpolate 1D data using linear interpolation."""
    # Sort by x coordinate
    sorted_idx = np.argsort(x_data)
    x_sorted = x_data[sorted_idx]
    y_sorted = y_data[sorted_idx]
    
    # Create interpolation function
    f = interp1d(x_sorted, y_sorted, kind='linear', fill_value='extrapolate')
    return f(x_probe)

# Main execution
print("Reading input files...")

# Read side_A field data
x_A, y_A, u_A, _ = read_csv('side_A/field_level1.csv')
print(f"Side A field: {len(x_A)} nodes")

# Read side_B field data
x_B, y_B, u_B, _ = read_csv('side_B/field_level1.csv')
print(f"Side B field: {len(x_B)} nodes")

# Read side_A interface data
x_int_A, y_int_A, u_int_A, qn_int_A = read_csv('side_A/interface_level1.csv')
print(f"Side A interface: {len(x_int_A)} nodes")

# Read side_B interface data
x_int_B, y_int_B, u_int_B, qn_int_B = read_csv('side_B/interface_level1.csv')
print(f"Side B interface: {len(x_int_B)} nodes")

# Generate solution probes for A
probe_x_A, probe_y_A = generate_solution_probes_A()
print(f"Solution probes A: {len(probe_x_A)} points")

# Interpolate solution at probe points for A
u_probe_A = griddata((x_A, y_A), u_A, (probe_x_A, probe_y_A), method='linear')

# Write solution_level1_A.csv
with open('solution_level1_A.csv', 'w') as f:
    f.write('x,y,u\n')
    for i in range(len(probe_x_A)):
        f.write(f'{probe_x_A[i]:.11e},{probe_y_A[i]:.11e},{u_probe_A[i]:.11e}\n')
print("Written solution_level1_A.csv")

# Generate solution probes for B
probe_x_B, probe_y_B = generate_solution_probes_B()
print(f"Solution probes B: {len(probe_x_B)} points")

# Interpolate solution at probe points for B
u_probe_B = griddata((x_B, y_B), u_B, (probe_x_B, probe_y_B), method='linear')

# Write solution_level1_B.csv
with open('solution_level1_B.csv', 'w') as f:
    f.write('x,y,u\n')
    for i in range(len(probe_x_B)):
        f.write(f'{probe_x_B[i]:.11e},{probe_y_B[i]:.11e},{u_probe_B[i]:.11e}\n')
print("Written solution_level1_B.csv")

# Generate interface probes
int_probe_x, int_probe_y = generate_interface_probes()
print(f"Interface probes: {len(int_probe_x)} points")

# Interpolate u and qn at interface probe points from side_A (1D interpolation)
u_int_probe_A = interpolate_1d(x_int_A, u_int_A, int_probe_x)
qn_int_probe_A = interpolate_1d(x_int_A, qn_int_A, int_probe_x)

# Write interface_level1_A.csv
with open('interface_level1_A.csv', 'w') as f:
    f.write('x,y,u,qn\n')
    for i in range(len(int_probe_x)):
        f.write(f'{int_probe_x[i]:.11e},{int_probe_y[i]:.11e},{u_int_probe_A[i]:.11e},{qn_int_probe_A[i]:.11e}\n')
print("Written interface_level1_A.csv")

# Interpolate u and qn at interface probe points from side_B (1D interpolation)
u_int_probe_B = interpolate_1d(x_int_B, u_int_B, int_probe_x)
qn_int_probe_B = interpolate_1d(x_int_B, qn_int_B, int_probe_x)

# Write interface_level1_B.csv
with open('interface_level1_B.csv', 'w') as f:
    f.write('x,y,u,qn\n')
    for i in range(len(int_probe_x)):
        f.write(f'{int_probe_x[i]:.11e},{int_probe_y[i]:.11e},{u_int_probe_B[i]:.11e},{qn_int_probe_B[i]:.11e}\n')
print("Written interface_level1_B.csv")

# Copy solver logs with NDOF line prepended
# Side A log
with open('side_A/participant_output.log', 'r') as f:
    log_a_content = f.read()

# Extract stdout section from side_A log
stdout_start = log_a_content.find('--- stdout ---\n')
stdout_end = log_a_content.find('--- stderr ---\n')
if stdout_start != -1 and stdout_end != -1:
    stdout_a = log_a_content[stdout_start + len('--- stdout ---\n'):stdout_end].strip()
else:
    stdout_a = log_a_content

# Prepend NDOF line
run_log_a = "NDOF = 54\n" + stdout_a

with open('run_level1_A.log', 'w') as f:
    f.write(run_log_a)
print("Written run_level1_A.log")

# Side B log
with open('side_B/participant_output.log', 'r') as f:
    log_b_content = f.read()

# Extract stdout section from side_B log
stdout_start = log_b_content.find('--- stdout ---\n')
stdout_end = log_b_content.find('--- stderr ---\n')
if stdout_start != -1 and stdout_end != -1:
    stdout_b = log_b_content[stdout_start + len('--- stdout ---\n'):stdout_end].strip()
else:
    stdout_b = log_b_content

# Prepend NDOF line
run_log_b = "NDOF = 72\n" + stdout_b

with open('run_level1_B.log', 'w') as f:
    f.write(run_log_b)
print("Written run_level1_B.log")

print("\nAll 8 files created successfully!")
