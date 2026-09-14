#!/usr/bin/env python3
"""Post-process level 1 results to write files at prescribed probe points."""
import numpy as np
from scipy.interpolate import griddata
import json

LEVEL = 1

# Prescribed probe points for subdomain A
# x = 0 + (i_x+0.5)*0.625/44; y = 0 + (i_y+0.5)*1/44 for i_x,i_y = 0..43
probe_A = []
for iy in range(44):
    for ix in range(44):
        px = 0.0 + (ix + 0.5) * 0.625 / 44
        py = 0.0 + (iy + 0.5) * 1.0 / 44
        probe_A.append((px, py))

# Prescribed probe points for subdomain B  
# x = 0.625 + (i_x+0.5)*0.875/44; y = 0 + (i_y+0.5)*1/44 for i_x,i_y = 0..43
probe_B = []
for iy in range(44):
    for ix in range(44):
        px = 0.625 + (ix + 0.5) * 0.875 / 44
        py = 0.0 + (iy + 0.5) * 1.0 / 44
        probe_B.append((px, py))

# Prescribed interface probe points
# (x, y) = (5/8, 1/4 + (i+0.5)*1/2/44) for i = 0..43
interface_probes = []
for i in range(44):
    px = 5/8
    py = 1/4 + (i + 0.5) * (1/2) / 44
    interface_probes.append((px, py))

# Read side A field data
a_data = np.loadtxt('side_A/field_level1_A.csv', delimiter=',', skiprows=1)
a_coords = a_data[:, :2]
a_values = a_data[:, 2]

# Read side B field data
b_data = np.loadtxt('side_B/field_level1_B.csv', delimiter=',', skiprows=1)
b_coords = b_data[:, :2]
b_values = b_data[:, 2]

# Interpolate side A solution at probe points
u_A = griddata(a_coords, a_values, probe_A, method='linear')

# Write field_level1_A.csv with prescribed probes
with open(f'field_level1_A.csv', 'w') as f:
    f.write('x,y,u\n')
    for (px, py), u_val in zip(probe_A, u_A):
        if np.isnan(u_val):
            print(f"Warning: NaN at {px}, {py}")
        f.write(f'{px:.11e},{py:.11e},{u_val:.11e}\n')

# Interpolate side B solution at probe points
u_B = griddata(b_coords, b_values, probe_B, method='linear')

# Write field_level1_B.csv with prescribed probes
with open(f'field_level1_B.csv', 'w') as f:
    f.write('x,y,u\n')
    for (px, py), u_val in zip(probe_B, u_B):
        if np.isnan(u_val):
            print(f"Warning: NaN at {px}, {py}")
        f.write(f'{px:.11e},{py:.11e},{u_val:.11e}\n')

# For interface files, we need to interpolate both solutions at the interface probes
# and compute fluxes. This is more complex - let's read the interface data from exports.json

# Read exports from both sides
with open('side_A/exports.json') as f:
    exp_A = json.load(f)
with open('side_B/exports.json') as f:
    exp_B = json.load(f)

# Side A exports fluxes at its interface nodes
a_iface_coords = np.array(exp_A['coordinates'])
a_iface_fluxes = np.array(exp_A['normal_fluxes'])

# Side B exports values and fluxes at its interface nodes
b_iface_coords = np.array(exp_B['coordinates'])
b_iface_values = np.array(exp_B['values'])
b_iface_fluxes = np.array(exp_B['normal_fluxes'])

# Interpolate side A fluxes at prescribed interface probes
q_A = griddata(a_iface_coords[:, 1], a_iface_fluxes, [p[1] for p in interface_probes], method='linear')

# Interpolate side B values and fluxes at prescribed interface probes
u_B_iface = griddata(b_iface_coords[:, 1], b_iface_values, [p[1] for p in interface_probes], method='linear')
q_B = griddata(b_iface_coords[:, 1], b_iface_fluxes, [p[1] for p in interface_probes], method='linear')

# Also need side A values at interface - interpolate from field data
u_A_iface = griddata(a_coords, a_values, interface_probes, method='linear')

# Write interface_level1_A.csv
with open(f'interface_level1_A.csv', 'w') as f:
    f.write('x,y,u,qn\n')
    for (px, py), u_val, q_val in zip(interface_probes, u_A_iface, q_A):
        f.write(f'{px:.11e},{py:.11e},{u_val:.11e},{q_val:.11e}\n')

# Write interface_level1_B.csv
with open(f'interface_level1_B.csv', 'w') as f:
    f.write('x,y,u,qn\n')
    for (px, py), u_val, q_val in zip(interface_probes, u_B_iface, q_B):
        f.write(f'{px:.11e},{py:.11e},{u_val:.11e},{q_val:.11e}\n')

print("Post-processing complete for level 1")
print(f"Field A: {len(probe_A)} points, Field B: {len(probe_B)} points")
print(f"Interface probes: {len(interface_probes)} points")
