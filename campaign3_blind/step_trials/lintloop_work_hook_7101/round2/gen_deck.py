#!/usr/bin/env python3
"""
Generate 4C deck for coupled heat conduction problem.
Steady heat conduction on rectangle (0, 0.8) x (0, 1) with k=2.
Using Scalar_Transport for flux output capability (no IO section).
"""
import numpy as np

# Problem parameters
nx, ny = 8, 10  # Number of elements in x and y directions
Lx, Ly = 0.8, 1.0  # Domain dimensions
k = 2.0  # Thermal conductivity (DIFFUSIVITY in MAT_scatra)

# Mesh parameters
nx_nodes, ny_nodes = nx + 1, ny + 1  # Nodes in each direction
h_x, h_y = Lx / nx, Ly / ny  # Element sizes

# Generate node coordinates
nodes_x = np.linspace(0, Lx, nx_nodes)
nodes_y = np.linspace(0, Ly, ny_nodes)

node_coords = []
for j in range(ny_nodes):
    for i in range(nx_nodes):
        node_id = i + j * nx_nodes + 1  # 1-based indexing, row-major
        node_coords.append(f"NODE {node_id} COORD {nodes_x[i]} {nodes_y[j]} 0.0")

# Generate TRANSP QUAD4 elements
elements = []
elem_id = 1
for j in range(ny):
    for i in range(nx):
        n1 = i + j * nx_nodes + 1          # bottom-left
        n2 = i + 1 + j * nx_nodes + 1      # bottom-right
        n3 = i + 1 + (j + 1) * nx_nodes + 1  # top-right
        n4 = i + (j + 1) * nx_nodes + 1    # top-left
        elements.append(f"{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
        elem_id += 1

# Interface nodes at x = 0.8 (right boundary, column index nx)
interface_node_ids = [nx + j * nx_nodes + 1 for j in range(ny_nodes)]

# Partner's temperature values at interface nodes (given in problem)
interface_temps = [
    0.0000000000, 0.2472135955, 0.4702282018, 0.6472135955, 0.7608452130,
    0.8000000000, 0.7608452130, 0.6472135955, 0.4702282018, 0.2472135955,
    0.0000000000
]

# Outer boundary nodes (T=0): left (x=0), bottom (y=0), top (y=1)
left_boundary_nodes = [1 + j * nx_nodes for j in range(ny_nodes)]
bottom_boundary_nodes = list(range(1, nx_nodes + 1))
top_boundary_nodes = list(range((ny_nodes - 1) * nx_nodes + 1, ny_nodes * nx_nodes + 1))

# Combine outer boundary nodes (avoid duplicates at corners)
outer_boundary_nodes = sorted(set(left_boundary_nodes + bottom_boundary_nodes + top_boundary_nodes))

# Design entity IDs for outer boundaries (DLINE 1-3)
dline_left_id, dline_bottom_id, dline_top_id = 1, 2, 3

# Design entity ID for interface flux calculation (DLINE 4)
dline_interface_id = 4

# Write the deck
deck_filename = "slab_T.4C.yaml"

with open(deck_filename, 'w') as f:
    # Title
    f.write("TITLE:\n")
    f.write('  - "Steady heat conduction slab"\n')
    f.write("\n")
    
    # Problem size
    f.write("PROBLEM SIZE:\n")
    f.write(f"  ELEMENTS: {nx * ny}\n")
    f.write(f"  NODES: {nx_nodes * ny_nodes}\n")
    f.write("\n")
    
    # Problem type - Scalar_Transport for flux capability
    f.write("PROBLEM TYPE:\n")
    f.write('  PROBLEMTYPE: "Scalar_Transport"\n')
    f.write("\n")
    
    # Scalar transport dynamic settings
    f.write("SCALAR TRANSPORT DYNAMIC:\n")
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write("  NUMSTEP: 1\n")
    f.write("  TIMESTEP: 1.0\n")
    f.write("  MAXTIME: 1.0\n")
    f.write("  LINEAR_SOLVER: 1\n")
    f.write('  CALCFLUX_BOUNDARY: "diffusive"\n')
    f.write("\n")
    
    # Solver
    f.write("SOLVER 1:\n")
    f.write('  SOLVER: "UMFPACK"\n')
    f.write("\n")
    
    # Materials - MAT_scatra is correct for Scalar_Transport
    f.write("MATERIALS:\n")
    f.write("  - MAT: 1\n")
    f.write("    MAT_scatra:\n")
    f.write(f"      DIFFUSIVITY: {k}\n")
    f.write("\n")
    
    # Function for body source (convert ** to ^ for 4C syntax)
    f.write("FUNCT1:\n")
    f.write('  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi^2*x*sin(pi*y)"\n')
    f.write("\n")
    
    # Body source: SURF NEUMANN in 2D for Scalar_Transport
    f.write("DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n")
    f.write("  - E: 1\n")
    f.write("    NUMDOF: 1\n")
    f.write("    ONOFF: [1]\n")
    f.write("    VAL: [1.0]\n")
    f.write("    FUNCT: [1]\n")
    f.write("\n")
    
    # DSURF topology for body source (all nodes in domain)
    f.write("DSURF-NODE TOPOLOGY:\n")
    for node_id in range(1, nx_nodes * ny_nodes + 1):
        f.write(f'  - "NODE {node_id} DSURFACE 1"\n')
    f.write("\n")
    
    # Outer boundary Dirichlet conditions (T=0) using LINE conditions
    f.write("DESIGN LINE TRANSPORT DIRICH CONDITIONS:\n")
    f.write(f"  - E: {dline_left_id}\n")
    f.write("    NUMDOF: 1\n")
    f.write("    ONOFF: [1]\n")
    f.write("    VAL: [0.0]\n")
    f.write("    FUNCT: [0]\n")
    f.write(f"  - E: {dline_bottom_id}\n")
    f.write("    NUMDOF: 1\n")
    f.write("    ONOFF: [1]\n")
    f.write("    VAL: [0.0]\n")
    f.write("    FUNCT: [0]\n")
    f.write(f"  - E: {dline_top_id}\n")
    f.write("    NUMDOF: 1\n")
    f.write("    ONOFF: [1]\n")
    f.write("    VAL: [0.0]\n")
    f.write("    FUNCT: [0]\n")
    f.write("\n")
    
    # Interface Dirichlet conditions using POINT conditions (one per node)
    f.write("DESIGN POINT TRANSPORT DIRICH CONDITIONS:\n")
    for idx, (node_id, temp_val) in enumerate(zip(interface_node_ids, interface_temps)):
        point_id = 1 + idx  # E ids start at 1
        f.write(f"  - E: {point_id}\n")
        f.write("    NUMDOF: 1\n")
        f.write("    ONOFF: [1]\n")
        f.write(f"    VAL: [{temp_val}]\n")
        f.write("    FUNCT: [0]\n")
    f.write("\n")
    
    # Node coordinates
    f.write("NODE COORDS:\n")
    for coord_line in node_coords:
        f.write(f'  - "{coord_line}"\n')
    f.write("\n")
    
    # TRANSP elements
    f.write("TRANSPORT ELEMENTS:\n")
    for elem_line in elements:
        f.write(f'  - "{elem_line}"\n')
    f.write("\n")
    
    # DLINE topology for outer boundaries
    f.write("DLINE-NODE TOPOLOGY:\n")
    for node_id in left_boundary_nodes:
        f.write(f'  - "NODE {node_id} DLINE {dline_left_id}"\n')
    for node_id in bottom_boundary_nodes:
        f.write(f'  - "NODE {node_id} DLINE {dline_bottom_id}"\n')
    for node_id in top_boundary_nodes:
        f.write(f'  - "NODE {node_id} DLINE {dline_top_id}"\n')
    # Also add interface line for flux calculation
    for node_id in interface_node_ids:
        f.write(f'  - "NODE {node_id} DLINE {dline_interface_id}"\n')
    f.write("\n")
    
    # DNODE topology for interface Dirichlet points
    f.write("DNODE-NODE TOPOLOGY:\n")
    for idx, node_id in enumerate(interface_node_ids):
        point_id = 1 + idx
        f.write(f'  - "NODE {node_id} DNODE {point_id}"\n')
    f.write("\n")
    
    # SCATRA FLUX CALC LINE CONDITIONS for interface flux output
    f.write("SCATRA FLUX CALC LINE CONDITIONS:\n")
    f.write(f"  - E: {dline_interface_id}\n")
    f.write("\n")

print(f"Deck written to {deck_filename}")
print(f"Elements: {nx * ny}, Nodes: {nx_nodes * ny_nodes}")
print(f"Interface nodes: {len(interface_node_ids)}")
print(f"Outer boundary nodes (T=0): {len(outer_boundary_nodes)}")
