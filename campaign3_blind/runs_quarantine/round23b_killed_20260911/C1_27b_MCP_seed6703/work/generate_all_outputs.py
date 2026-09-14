#!/usr/bin/env python3
"""Generate required output files for all levels of the coupled thermo-elastic simulation."""
import numpy as np
from scipy.interpolate import griddata, interp1d
import os

work_dir = "/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/runs/C1_27b_MCP_seed6703/work"

def generate_probe_points_side_A():
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0 + (i_x + 0.5) * 0.625 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return np.array(points)

def generate_probe_points_side_B():
    points = []
    for i_y in range(44):
        for i_x in range(44):
            x = 0.625 + (i_x + 0.5) * 0.875 / 44
            y = 0 + (i_y + 0.5) * 1 / 44
            points.append((x, y))
    return np.array(points)

def generate_interface_probe_points():
    points = []
    for i in range(44):
        x = 5/8
        y = 1/4 + (i + 0.5) * 1/2 / 44
        points.append((x, y))
    return np.array(points)

def read_field_file(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    coords = data[:, :2]
    T = data[:, 2]
    ux = data[:, 3]
    uy = data[:, 4]
    return coords, T, ux, uy

def interpolate_to_probes(coords, values, probes):
    return griddata(coords, values, probes, method='linear', fill_value=np.nan)

def write_solution_file(filepath, probes, T, ux, uy):
    with open(filepath, 'w') as f:
        f.write("x,y,T,ux,uy\n")
        for i in range(len(probes)):
            x, y = probes[i]
            f.write(f"{x:.11e},{y:.11e},{T[i]:.11e},{ux[i]:.11e},{uy[i]:.11e}\n")

def read_interface_file(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return data

def interpolate_interface_data_1d(side_data, interface_probes):
    ys = side_data[:, 1]
    T_vals = side_data[:, 2]
    ux_vals = side_data[:, 3]
    uy_vals = side_data[:, 4]
    qn_vals = side_data[:, 5]
    tx_vals = side_data[:, 6]
    ty_vals = side_data[:, 7]
    
    idx = np.argsort(ys)
    ys_sorted = ys[idx]
    T_sorted = T_vals[idx]
    ux_sorted = ux_vals[idx]
    uy_sorted = uy_vals[idx]
    qn_sorted = qn_vals[idx]
    tx_sorted = tx_vals[idx]
    ty_sorted = ty_vals[idx]
    
    f_T = interp1d(ys_sorted, T_sorted, kind='linear', fill_value='extrapolate')
    f_ux = interp1d(ys_sorted, ux_sorted, kind='linear', fill_value='extrapolate')
    f_uy = interp1d(ys_sorted, uy_sorted, kind='linear', fill_value='extrapolate')
    f_qn = interp1d(ys_sorted, qn_sorted, kind='linear', fill_value='extrapolate')
    f_tx = interp1d(ys_sorted, tx_sorted, kind='linear', fill_value='extrapolate')
    f_ty = interp1d(ys_sorted, ty_sorted, kind='linear', fill_value='extrapolate')
    
    probe_ys = interface_probes[:, 1]
    return f_T(probe_ys), f_ux(probe_ys), f_uy(probe_ys), f_qn(probe_ys), f_tx(probe_ys), f_ty(probe_ys)

def write_interface_file(filepath, probes, T, ux, uy, qn, tx, ty):
    with open(filepath, 'w') as f:
        f.write("x,y,T,ux,uy,qn,tx,ty\n")
        for i in range(len(probes)):
            x, y = probes[i]
            f.write(f"{x:.11e},{y:.11e},{T[i]:.11e},{ux[i]:.11e},{uy[i]:.11e},"
                    f"{qn[i]:.11e},{tx[i]:.11e},{ty[i]:.11e}\n")

def copy_run_log(src, dst, ndof):
    with open(src, 'r') as f:
        content = f.read()
    with open(dst, 'w') as f:
        f.write(f"NDOF = {ndof}\n")
        f.write(content)

# Generate probe points once
probes_A = generate_probe_points_side_A()
probes_B = generate_probe_points_side_B()
interface_probes = generate_interface_probe_points()

print(f"Probe points A: {len(probes_A)}, B: {len(probes_B)}, Interface: {len(interface_probes)}")

# Process each level
for level in [1, 2, 3]:
    print(f"\nProcessing level {level}...")
    
    # Read field files
    try:
        coords_A, T_A, ux_A, uy_A = read_field_file(f"{work_dir}/side_A/field_level{level}.csv")
        coords_B, T_B, ux_B, uy_B = read_field_file(f"{work_dir}/side_B/field_level{level}.csv")
        print(f"  Side A nodes: {len(coords_A)}, Side B nodes: {len(coords_B)}")
    except FileNotFoundError as e:
        print(f"  Skipping level {level}: {e}")
        continue
    
    # Interpolate to probe points
    T_A_probe = interpolate_to_probes(coords_A, T_A, probes_A)
    ux_A_probe = interpolate_to_probes(coords_A, ux_A, probes_A)
    uy_A_probe = interpolate_to_probes(coords_A, uy_A, probes_A)
    
    T_B_probe = interpolate_to_probes(coords_B, T_B, probes_B)
    ux_B_probe = interpolate_to_probes(coords_B, ux_B, probes_B)
    uy_B_probe = interpolate_to_probes(coords_B, uy_B, probes_B)
    
    # Write solution files
    write_solution_file(f"{work_dir}/solution_level{level}_A.csv", probes_A, T_A_probe, ux_A_probe, uy_A_probe)
    write_solution_file(f"{work_dir}/solution_level{level}_B.csv", probes_B, T_B_probe, ux_B_probe, uy_B_probe)
    print(f"  Solution files written for level {level}")
    
    # Read interface files
    iface_A = read_interface_file(f"{work_dir}/side_A/interface_level{level}.csv")
    iface_B = read_interface_file(f"{work_dir}/side_B/interface_level{level}.csv")
    print(f"  Side A interface points: {len(iface_A)}, Side B: {len(iface_B)}")
    
    # Interpolate interface data
    T_A_iface, ux_A_iface, uy_A_iface, qn_A_iface, tx_A_iface, ty_A_iface = interpolate_interface_data_1d(iface_A, interface_probes)
    T_B_iface, ux_B_iface, uy_B_iface, qn_B_iface, tx_B_iface, ty_B_iface = interpolate_interface_data_1d(iface_B, interface_probes)
    
    # Write interface files
    write_interface_file(f"{work_dir}/interface_level{level}_A.csv", interface_probes, T_A_iface, ux_A_iface, uy_A_iface, qn_A_iface, tx_A_iface, ty_A_iface)
    write_interface_file(f"{work_dir}/interface_level{level}_B.csv", interface_probes, T_B_iface, ux_B_iface, uy_B_iface, qn_B_iface, tx_B_iface, ty_B_iface)
    print(f"  Interface files written for level {level}")
    
    # Copy run logs with NDOF
    ndof_A = 3 * len(coords_A)
    ndof_B = 3 * len(coords_B)
    copy_run_log(f"{work_dir}/side_A/participant_output.log", f"{work_dir}/run_level{level}_A.log", ndof_A)
    copy_run_log(f"{work_dir}/side_B/participant_output.log", f"{work_dir}/run_level{level}_B.log", ndof_B)
    print(f"  Run logs copied for level {level} (NDOF A={ndof_A}, B={ndof_B})")

print("\nAll output generation complete!")
