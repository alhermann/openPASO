import numpy as np

# Mesh parameters
nx = 8      # elements in x-direction
ny = 10     # elements in y-direction
nx_nodes = nx + 1   # 9 nodes in x
ny_nodes = ny + 1   # 11 nodes in y

Lx = 0.8
Ly = 1.0
hx = Lx / nx
hy = Ly / ny

# Generate node coordinates (column-major order: vary x first within each row)
nodes = []
for j in range(ny_nodes):
    for i in range(nx_nodes):
        x = i * hx
        y = j * hy
        nodes.append((x, y))

# Generate element connectivity (QUAD4, counter-clockwise)
elements = []
elem_id = 1
for j in range(ny):
    for i in range(nx):
        n1 = i + j * nx_nodes + 1      # bottom-left
        n2 = i + 1 + j * nx_nodes + 1  # bottom-right
        n3 = i + 1 + (j + 1) * nx_nodes + 1  # top-right
        n4 = i + (j + 1) * nx_nodes + 1     # top-left
        elements.append(f"{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
        elem_id += 1

# Identify boundary nodes
left_boundary = [i * nx_nodes + 1 for i in range(ny_nodes)]          # x = 0
bottom_boundary = list(range(1, nx_nodes + 1))                        # y = 0
top_boundary = list(range((ny_nodes - 1) * nx_nodes + 1, len(nodes) + 1))  # y = 1
right_boundary = [(nx_nodes - 1) + i * nx_nodes + 1 for i in range(ny_nodes)]  # x = 0.8

# Interface temperatures at right boundary (y from 0 to 1)
interface_temps = [
    0.0000000000,
    0.2472135955,
    0.4702282018,
    0.6472135955,
    0.7608452130,
    0.8000000000,
    0.7608452130,
    0.6472135955,
    0.4702282018,
    0.2472135955,
    0.0000000000
]

# Write deck file
with open("slab_T.4C.yaml", "w") as f:
    f.write('TITLE:\n')
    f.write('  - "Steady Heat Conduction"\n\n')
    
    f.write('PROBLEM SIZE:\n')
    f.write(f'  ELEMENTS: {len(elements)}\n')
    f.write(f'  NODES: {len(nodes)}\n\n')
    
    f.write('PROBLEM TYPE:\n')
    f.write('  PROBLEMTYPE: "Scalar_Transport"\n\n')
    
    f.write('SCALAR TRANSPORT DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n')
    f.write('  TIMESTEP: 1.0\n')
    f.write('  MAXTIME: 1.0\n')
    f.write('  LINEAR_SOLVER: 1\n')
    f.write('  CALCFLUX_BOUNDARY: "diffusive"\n\n')
    
    f.write('SOLVER 1:\n')
    f.write('  SOLVER: "UMFPACK"\n\n')
    
    f.write('MATERIALS:\n')
    f.write('  - MAT: 1\n')
    f.write('    MAT_scatra:\n')
    f.write('      DIFFUSIVITY: 2.0\n\n')
    
    f.write('FUNCT1:\n')
    f.write('  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi^2*x*sin(pi*y)"\n\n')
    
    f.write('NODE COORDS:\n')
    for idx, (x, y) in enumerate(nodes, 1):
        f.write(f'  - "NODE {idx} COORD {x:.16f} {y:.16f} 0.0"\n')
    f.write('\n')
    
    f.write('TRANSPORT ELEMENTS:\n')
    for elem in elements:
        f.write(f'  - "{elem}"\n')
    f.write('\n')
    
    # DSURF-NODE TOPOLOGY: all nodes belong to surface subdomain for body source
    f.write('DSURF-NODE TOPOLOGY:\n')
    for idx in range(1, len(nodes) + 1):
        f.write(f'  - "NODE {idx} DSURFACE 1"\n')
    f.write('\n')
    
    # DESIGN SURF TRANSPORT NEUMANN CONDITIONS: body source term
    f.write('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n')
    f.write('  - E: 1\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [1.0]\n')
    f.write('    FUNCT: [1]\n\n')
    
    # DLINE-NODE TOPOLOGY: all boundary lines (left=1, bottom=2, top=3, right=4)
    f.write('DLINE-NODE TOPOLOGY:\n')
    for node_id in left_boundary:
        f.write(f'  - "NODE {node_id} DLINE 1"\n')
    for node_id in bottom_boundary:
        f.write(f'  - "NODE {node_id} DLINE 2"\n')
    for node_id in top_boundary:
        f.write(f'  - "NODE {node_id} DLINE 3"\n')
    for node_id in right_boundary:
        f.write(f'  - "NODE {node_id} DLINE 4"\n')
    f.write('\n')
    
    # DESIGN LINE DIRICH CONDITIONS: outer boundaries (left, bottom, top) with T=0
    f.write('DESIGN LINE DIRICH CONDITIONS:\n')
    for e_id in [1, 2, 3]:
        f.write(f'  - E: {e_id}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write('    VAL: [0.0]\n')
        f.write('    FUNCT: [0]\n')
    f.write('\n')
    
    # DESIGN POINT DIRICH CONDITIONS: interface nodes with per-node values
    f.write('DESIGN POINT DIRICH CONDITIONS:\n')
    for i, (node_id, temp) in enumerate(zip(right_boundary, interface_temps)):
        dnode_num = i + 4
        f.write(f'  - E: {dnode_num}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write(f'    VAL: [{temp}]\n')
        f.write('    FUNCT: [0]\n')
    f.write('\n')
    
    # DNODE-NODE TOPOLOGY: map interface nodes to point conditions
    f.write('DNODE-NODE TOPOLOGY:\n')
    for i, node_id in enumerate(right_boundary):
        dnode_num = i + 4
        f.write(f'  - "NODE {node_id} DNODE {dnode_num}"\n')
    f.write('\n')
    
    # SCATRA FLUX CALC LINE CONDITIONS: flux on interface line
    f.write('SCATRA FLUX CALC LINE CONDITIONS:\n')
    f.write('  - E: 4\n\n')
