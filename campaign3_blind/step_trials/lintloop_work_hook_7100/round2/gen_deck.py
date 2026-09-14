#!/usr/bin/env python3
"""
Generate 4C deck for coupled heat problem - 4C side standalone run
Domain: (0, 0.8) x (0, 1)
Steady heat conduction, k=2
Body source: f_T = 2*pi^2*x*sin(pi*y)
Outer BCs (x=0, y=0, y=1): T=0
Interface (x=0.8): per-node Dirichlet values
Mesh: 8x10 QUAD4 elements (99 nodes)
"""

import numpy as np

# Problem parameters
nx_elem = 8      # elements in x direction
ny_elem = 10     # elements in y direction
nx_node = nx_elem + 1   # nodes in x direction
ny_node = ny_elem + 1   # nodes in y direction

x_max = 0.8
y_max = 1.0

k_value = 2.0

# Interface temperature values (given per node at x=0.8, y from 0 to 1 in steps of 0.1)
interface_temps = [
    0.0000000000,  # y=0.0
    0.2472135955,  # y=0.1
    0.4702282018,  # y=0.2
    0.6472135955,  # y=0.3
    0.7608452130,  # y=0.4
    0.8000000000,  # y=0.5
    0.7608452130,  # y=0.6
    0.6472135955,  # y=0.7
    0.4702282018,  # y=0.8
    0.2472135955,  # y=0.9
    0.0000000000,  # y=1.0
]

# Generate node coordinates (1-based indexing)
node_coords = []
for j in range(ny_node):
    for i in range(nx_node):
        x = i * (x_max / nx_elem)
        y = j * (y_max / ny_elem)
        node_id = j * nx_node + i + 1  # 1-based
        node_coords.append((node_id, x, y))

# Generate element connectivity (TRANSP QUAD4, counter-clockwise)
elements = []
elem_id = 1
for j in range(ny_elem):
    for i in range(nx_elem):
        n1 = j * nx_node + i + 1           # bottom-left
        n2 = j * nx_node + i + 2           # bottom-right
        n3 = (j + 1) * nx_node + i + 2     # top-right
        n4 = (j + 1) * nx_node + i + 1     # top-left
        elements.append((elem_id, n1, n2, n3, n4))
        elem_id += 1

# Identify boundary nodes by coordinate
tol = 1e-10
left_boundary_nodes = [nc[0] for nc in node_coords if abs(nc[1]) < tol]
bottom_boundary_nodes = [nc[0] for nc in node_coords if abs(nc[2]) < tol]
top_boundary_nodes = [nc[0] for nc in node_coords if abs(nc[2] - 1.0) < tol]

# Interface nodes (x=0.8, right edge)
interface_nodes = [(j * nx_node + nx_node, j) for j in range(ny_node)]

# Start building the deck
deck_lines = []

# TITLE
deck_lines.append('TITLE:')
deck_lines.append('  - "Coupled Heat Problem - 4C Side"')
deck_lines.append('')

# PROBLEM SIZE
deck_lines.append('PROBLEM SIZE:')
deck_lines.append(f'  ELEMENTS: {len(elements)}')
deck_lines.append(f'  NODES: {len(node_coords)}')
deck_lines.append('')

# PROBLEM TYPE - Scalar_Transport for flux output
deck_lines.append('PROBLEM TYPE:')
deck_lines.append('  PROBLEMTYPE: "Scalar_Transport"')
deck_lines.append('')

# SCALAR TRANSPORT DYNAMIC (not THERMAL DYNAMIC!)
deck_lines.append('SCALAR TRANSPORT DYNAMIC:')
deck_lines.append('  TIMEINTEGR: "Stationary"')
deck_lines.append('  SOLVERTYPE: "linear_full"')
deck_lines.append('  NUMSTEP: 1')
deck_lines.append('  TIMESTEP: 1.0')
deck_lines.append('  MAXTIME: 1.0')
deck_lines.append('  LINEAR_SOLVER: 1')
deck_lines.append('  CALCFLUX_BOUNDARY: "diffusive"')
deck_lines.append('')

# SOLVER 1
deck_lines.append('SOLVER 1:')
deck_lines.append('  SOLVER: "UMFPACK"')
deck_lines.append('')

# MATERIALS - MAT_scatra with DIFFUSIVITY (not MAT_Fourier/CONDUCTIVITY)
deck_lines.append('MATERIALS:')
deck_lines.append('  - MAT: 1')
deck_lines.append('    MAT_scatra:')
deck_lines.append(f'      DIFFUSIVITY: {k_value}')
deck_lines.append('')

# FUNCT1 - body source function (use ^ not **)
pi_val = 3.14159265358979
deck_lines.append('FUNCT1:')
deck_lines.append(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*({pi_val}^2)*x*sin({pi_val}*y)"')
deck_lines.append('')

# DESIGN LINE DIRICH CONDITIONS - outer boundaries (3 separate lines)
deck_lines.append('DESIGN LINE DIRICH CONDITIONS:')
deck_lines.append('  - E: 1')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('  - E: 2')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('  - E: 3')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('')

# DESIGN SURF TRANSPORT NEUMANN CONDITIONS - body source (2D = SURF)
deck_lines.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
deck_lines.append('  - E: 1')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [1.0]')
deck_lines.append('    FUNCT: [1]')
deck_lines.append('')

# DESIGN POINT DIRICH CONDITIONS - interface (per-node, exact values)
deck_lines.append('DESIGN POINT DIRICH CONDITIONS:')
for j, (node_id, _) in enumerate(interface_nodes):
    deck_lines.append(f'  - E: {j + 1}')
    deck_lines.append('    NUMDOF: 1')
    deck_lines.append('    ONOFF: [1]')
    deck_lines.append(f'    VAL: [{interface_temps[j]:.15g}]')
    deck_lines.append('    FUNCT: [0]')
deck_lines.append('')

# DLINE-NODE TOPOLOGY - outer boundaries + interface line for flux calc
deck_lines.append('DLINE-NODE TOPOLOGY:')
for node_id in left_boundary_nodes:
    deck_lines.append(f'  - "NODE {node_id} DLINE 1"')
for node_id in bottom_boundary_nodes:
    deck_lines.append(f'  - "NODE {node_id} DLINE 2"')
for node_id in top_boundary_nodes:
    deck_lines.append(f'  - "NODE {node_id} DLINE 3"')
# Interface nodes (right edge) for flux calculation
for node_id, _ in interface_nodes:
    deck_lines.append(f'  - "NODE {node_id} DLINE 4"')
deck_lines.append('')

# DSURF-NODE TOPOLOGY - all nodes for body source
deck_lines.append('DSURF-NODE TOPOLOGY:')
for node_id, _, _ in node_coords:
    deck_lines.append(f'  - "NODE {node_id} DSURFACE 1"')
deck_lines.append('')

# DNODE-NODE TOPOLOGY - interface nodes
deck_lines.append('DNODE-NODE TOPOLOGY:')
for j, (node_id, _) in enumerate(interface_nodes):
    deck_lines.append(f'  - "NODE {node_id} DNODE {j + 1}"')
deck_lines.append('')

# NODE COORDS
deck_lines.append('NODE COORDS:')
for node_id, x, y in node_coords:
    deck_lines.append(f'  - "NODE {node_id} COORD {x:.15g} {y:.15g} 0.0"')
deck_lines.append('')

# TRANSPORT ELEMENTS (not THERMO ELEMENTS!)
deck_lines.append('TRANSPORT ELEMENTS:')
for elem_data in elements:
    elem_id, n1, n2, n3, n4 = elem_data
    deck_lines.append(f'  - "{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
deck_lines.append('')

# SCATRA FLUX CALC LINE CONDITIONS - for flux output on interface
deck_lines.append('SCATRA FLUX CALC LINE CONDITIONS:')
deck_lines.append('  - E: 4')
deck_lines.append('')

# Write to file
with open('slab_T.4C.yaml', 'w') as f:
    f.write('\n'.join(deck_lines))

print("Deck written to slab_T.4C.yaml")
