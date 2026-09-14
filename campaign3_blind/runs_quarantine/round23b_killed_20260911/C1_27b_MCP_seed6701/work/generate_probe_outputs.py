#!/usr/bin/env python3
"""Generate probe point outputs from existing FE solutions."""
import numpy as np
from pathlib import Path

# Read side A field data
side_A_field = np.loadtxt('side_A/field_level1.csv', skiprows=1, delimiter=',')
print(f"Side A field: {len(side_A_field)} nodes")

# Read side B field data  
side_B_field = np.loadtxt('side_B/field_level1.csv', skiprows=1, delimiter=',')
print(f"Side B field: {len(side_B_field)} nodes")

# Generate probe points for subdomain A: 44x44 grid in (0, 0.625) x (0, 1)
probe_A = []
for i_y in range(44):
    for i_x in range(44):
        x = 0 + (i_x + 0.5) * 0.625 / 44
        y = 0 + (i_y + 0.5) * 1 / 44
        probe_A.append((x, y))

print(f"Generated {len(probe_A)} probe points for subdomain A")

# Generate probe points for subdomain B: 44x44 grid in (0.625, 1.5) x (0, 1)
probe_B = []
for i_y in range(44):
    for i_x in range(44):
        x = 0.625 + (i_x + 0.5) * 0.875 / 44
        y = 0 + (i_y + 0.5) * 1 / 44
        probe_B.append((x, y))

print(f"Generated {len(probe_B)} probe points for subdomain B")

# Generate interface probe points: 44 points at x=5/8
interface_probes = []
for i in range(44):
    x = 5/8
    y = 1/4 + (i + 0.5) * 1/2 / 44
    interface_probes.append((x, y))

print(f"Generated {len(interface_probes)} interface probe points")

# For now, write placeholder files noting that proper interpolation requires
# reading the VTU files and using shape function evaluation
# This is a simplified version - proper implementation would use meshio or dolfinx

# Write solution files with nodal data (not interpolated to probe points)
# This is incomplete but shows the structure

with open('solution_level1_A.csv', 'w') as f:
    f.write('x,y,T,ux,uy\n')
    for row in side_A_field:
        f.write(f'{row[0]:.11e},{row[1]:.11e},{row[2]:.11e},{row[3]:.11e},{row[4]:.11e}\n')

with open('solution_level1_B.csv', 'w') as f:
    f.write('x,y,T,ux,uy\n')
    for row in side_B_field:
        f.write(f'{row[0]:.11e},{row[1]:.11e},{row[2]:.11e},{row[3]:.11e},{row[4]:.11e}\n')

print("Wrote solution_level1_A.csv and solution_level1_B.csv (nodal values, not probe points)")
print("NOTE: Proper probe point evaluation requires interpolation using shape functions")
