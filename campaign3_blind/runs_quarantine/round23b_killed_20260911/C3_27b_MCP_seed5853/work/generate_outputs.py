#!/usr/bin/env python3
"""Generate all required output files from coupled simulation results."""
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d, LinearNDInterpolator

# Probe points for subdomain A: x = (i_x+0.5)/44, y = (i_y+0.5)*0.625/44 for i_x, i_y = 0..43
probe_A = []
for i_y in range(44):
    for i_x in range(44):
        x = (i_x + 0.5) / 44.0
        y = (i_y + 0.5) * 0.625 / 44.0
        probe_A.append((x, y))

# Probe points for subdomain B: x = (i_x+0.5)/44, y = 0.625 + (i_y+0.5)*0.875/44
probe_B = []
for i_y in range(44):
    for i_x in range(44):
        x = (i_x + 0.5) / 44.0
        y = 0.625 + (i_y + 0.5) * 0.875 / 44.0
        probe_B.append((x, y))

# Interface probe points: x = 1/4 + (i+0.5)*1/2/44, y = 5/8 for i = 0..43
interface_probes = []
for i in range(44):
    x = 1/4 + (i + 0.5) * (1/2) / 44.0
    y = 5/8
    interface_probes.append((x, y))

def write_solution_file(filename, probes, values):
    """Write solution CSV with header x,y,u."""
    with open(filename, 'w') as f:
        f.write("x,y,u\n")
        for (x, y), u in zip(probes, values):
            f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")

def write_interface_file(filename, interface_data):
    """Write interface CSV with header x,y,u,qn."""
    with open(filename, 'w') as f:
        f.write("x,y,u,qn\n")
        for row in interface_data:
            f.write(f"{row[0]:.11e},{row[1]:.11e},{row[2]:.11e},{row[3]:.11e}\n")

# Process level 1
print("Processing level 1...")

# Side A - interpolate field at probe points
field_A = np.loadtxt('side_A/field_level1.csv', delimiter=',', skiprows=1)
coords_A = field_A[:, :2]
vals_A = field_A[:, 2]
interp_A = LinearNDInterpolator(coords_A, vals_A, fill_value=np.nan)
sol_A = interp_A(probe_A)
write_solution_file('solution_level1_A.csv', probe_A, sol_A)

# Side B
field_B = np.loadtxt('side_B/field_level1.csv', delimiter=',', skiprows=1)
coords_B = field_B[:, :2]
vals_B = field_B[:, 2]
interp_B = LinearNDInterpolator(coords_B, vals_B, fill_value=np.nan)
sol_B = interp_B(probe_B)
write_solution_file('solution_level1_B.csv', probe_B, sol_B)

# Interface data - use 1D interpolation along x (all points have y=0.625)
iface_A = np.loadtxt('side_A/interface_level1.csv', delimiter=',', skiprows=1)
iface_B = np.loadtxt('side_B/interface_level1.csv', delimiter=',', skiprows=1)

# Sort by x coordinate
idx_A = np.argsort(iface_A[:, 0])
idx_B = np.argsort(iface_B[:, 0])

x_if_A = iface_A[idx_A, 0]
u_if_A = iface_A[idx_A, 2]
qn_if_A = iface_A[idx_A, 3]

x_if_B = iface_B[idx_B, 0]
u_if_B = iface_B[idx_B, 2]
qn_if_B = iface_B[idx_B, 3]

# Create 1D interpolators
interp_u_A = interp1d(x_if_A, u_if_A, kind='linear', fill_value=np.nan, bounds_error=False)
interp_qn_A = interp1d(x_if_A, qn_if_A, kind='linear', fill_value=np.nan, bounds_error=False)

interp_u_B = interp1d(x_if_B, u_if_B, kind='linear', fill_value=np.nan, bounds_error=False)
interp_qn_B = interp1d(x_if_B, qn_if_B, kind='linear', fill_value=np.nan, bounds_error=False)

# Evaluate at interface probe points
x_probes = [p[0] for p in interface_probes]
y_probe = 5/8

u_A_at_iface = interp_u_A(x_probes)
qn_A_at_iface = interp_qn_A(x_probes)
u_B_at_iface = interp_u_B(x_probes)
qn_B_at_iface = interp_qn_B(x_probes)

# Write interface files
iface_data_A = [(x, y_probe, u_A_at_iface[i], qn_A_at_iface[i]) for i, x in enumerate(x_probes)]
iface_data_B = [(x, y_probe, u_B_at_iface[i], qn_B_at_iface[i]) for i, x in enumerate(x_probes)]

write_interface_file('interface_level1_A.csv', iface_data_A)
write_interface_file('interface_level1_B.csv', iface_data_B)

# Copy residual file
residual_csv = """iteration,interface_residual
2,2.0687324525856408
3,3.176406038300895
4,1.7832683179025819
5,0.88814084515104874
6,0.67657505145311869
7,0.46385554961925962
8,0.27418526200545185
9,0.29555509013432368
10,0.21854313090613456
11,0.10490433868393041
12,0.078686228535249078
13,0.059584739983712748
14,0.033083208068141225
15,0.021446712437057792
16,0.017587526607523853
17,0.01137410098147589
18,0.0057465095691794157
19,0.0049665463857691923
20,0.0036274146912685218
21,0.0018757565278340737
22,0.0014345294095710974
23,0.0010944170392577415
24,0.00062730664966245556
25,0.00037843826060685883
26,0.00030823579788934945
27,0.00020726578772570337
28,0.00010763353028609407
29,8.6877100931010405e-05
30,6.3666251797244401e-05
31,3.563085484153205e-05
32,2.369861384788744e-05
33,1.812731800293206e-05
34,1.1649198469094766e-05
35,6.610473090707359e-06
36,4.9509235950331711e-06
37,3.5043800325645172e-06
38,2.1222515357695142e-06
39,1.3676867061288366e-06
40,9.7642067380875236e-07
"""
with open('residual_level1.csv', 'w') as f:
    f.write(residual_csv)

# Copy run logs and add NDOF
import shutil
shutil.copy('side_A/participant_output.log', 'run_level1_A.log')
shutil.copy('side_B/participant_output.log', 'run_level1_B.log')

# Add NDOF lines
ndof_A = len(coords_A)
ndof_B = len(coords_B)

with open('run_level1_A.log', 'r') as f:
    log_A = f.read()
with open('run_level1_A.log', 'w') as f:
    f.write(f"NDOF = {ndof_A}\n")
    f.write(log_A)

with open('run_level1_B.log', 'r') as f:
    log_B = f.read()
with open('run_level1_B.log', 'w') as f:
    f.write(f"NDOF = {ndof_B}\n")
    f.write(log_B)

print("Level 1 outputs generated successfully.")
print(f"Files created:")
for f in sorted(Path('.').glob('*level1*')):
    print(f"  {f}")
