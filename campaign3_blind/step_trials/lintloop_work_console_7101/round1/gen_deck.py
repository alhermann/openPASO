import numpy as np

# Problem parameters
nx_elem = 8          # elements in x direction
ny_elem = 10         # elements in y direction
Lx = 0.8             # domain length in x
Ly = 1.0             # domain length in y
k = 2.0              # thermal conductivity / diffusivity

# Mesh parameters
nx_nodes = nx_elem + 1    # 9 nodes in x
ny_nodes = ny_elem + 1    # 11 nodes in y
total_nodes = nx_nodes * ny_nodes  # 99 nodes
total_elems = nx_elem * ny_elem    # 80 elements

# Interface temperature values (given per node, y from 0 to 1 in steps of 0.1)
interface_T = [
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

# Generate node coordinates (row-major: i varies fastest)
nodes = []
node_id = 1
for j in range(ny_nodes):
    for i in range(nx_nodes):
        x = i * Lx / nx_elem
        y = j * Ly / ny_elem
        nodes.append((node_id, x, y, 0.0))
        node_id += 1

# Generate element connectivity (QUAD4, counter-clockwise)
elements = []
elem_id = 1
for j in range(ny_elem):
    for i in range(nx_elem):
        n1 = i + 1 + j * nx_nodes           # bottom-left
        n2 = i + 2 + j * nx_nodes           # bottom-right
        n3 = i + 2 + (j + 1) * nx_nodes     # top-right
        n4 = i + 1 + (j + 1) * nx_nodes     # top-left
        elements.append((elem_id, n1, n2, n3, n4))
        elem_id += 1

# Design line topology for outer boundaries (Dirichlet T=0)
# DLINE 1: left (x=0)
dline1_nodes = [i + 1 + j * nx_nodes for j in range(ny_nodes)]  # i=0
# DLINE 2: bottom (y=0)
dline2_nodes = [i + 1 for i in range(nx_nodes)]  # j=0
# DLINE 3: top (y=1)
dline3_nodes = [i + 1 + (ny_nodes - 1) * nx_nodes for i in range(nx_nodes)]  # j=ny_nodes-1

# Design line topology for interface (Neumann/flux calculation)
# DLINE 4: right (x=0.8)
dline4_nodes = [nx_nodes + j * nx_nodes for j in range(ny_nodes)]  # i=nx_nodes-1

# DNODE topology for interface point conditions
dnode_nodes = [(j + 1, dline4_nodes[j]) for j in range(ny_nodes)]  # DNODE 1..11

# Write the deck
with open('slab_T.4C.yaml', 'w') as f:
    f.write('TITLE:\n')
    f.write('  - "Coupled Heat Conduction - 4C Standalone"\n\n')
    
    f.write('PROBLEM SIZE:\n')
    f.write(f'  ELEMENTS: {total_elems}\n')
    f.write(f'  NODES: {total_nodes}\n\n')
    
    f.write('PROBLEM TYPE:\n')
    f.write('  PROBLEMTYPE: "Scalar_Transport"\n\n')
    
    f.write('SCALAR TRANSPORT DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n')
    f.write('  TIMESTEP: 1.0\n')
    f.write('  MAXTIME: 1.0\n')
    f.write('  LINEAR_SOLVER: 1\n\n')
    
    f.write('SOLVER 1:\n')
    f.write('  SOLVER: "UMFPACK"\n\n')
    
    f.write('MATERIALS:\n')
    f.write('  - MAT: 1\n')
    f.write('    MAT_scatra:\n')
    f.write(f'      DIFFUSIVITY: {k}\n\n')
    
    f.write('FUNCT1:\n')
    f.write('  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi^2*x*sin(pi*y)"\n\n')
    
    f.write('DESIGN LINE DIRICH CONDITIONS:\n')
    for e_id in [1, 2, 3]:  # left, bottom, top boundaries
        f.write(f'  - E: {e_id}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write('    VAL: [0.0]\n')
        f.write('    FUNCT: [0]\n\n')
    
    f.write('DESIGN POINT DIRICH CONDITIONS:\n')
    for idx, T_val in enumerate(interface_T):
        e_id = idx + 1
        f.write(f'  - E: {e_id}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write(f'    VAL: [{T_val}]\n')
        f.write('    FUNCT: [0]\n\n')
    
    f.write('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n')
    f.write('  - E: 1\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [1.0]\n')
    f.write('    FUNCT: [1]\n\n')
    
    f.write('SCATRA FLUX CALC LINE CONDITIONS:\n')
    f.write('  - E: 4\n\n')
    
    f.write('NODE COORDS:\n')
    for nid, x, y, z in nodes:
        f.write(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} {z:.10f}"\n\n')
    
    f.write('TRANSPORT ELEMENTS:\n')
    for eid, n1, n2, n3, n4 in elements:
        f.write(f'  - "{eid} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n\n')
    
    f.write('DLINE-NODE TOPOLOGY:\n')
    for node_idx, nid in enumerate(dline1_nodes):
        f.write(f'  - "NODE {nid} DLINE 1"\n')
    for node_idx, nid in enumerate(dline2_nodes):
        f.write(f'  - "NODE {nid} DLINE 2"\n')
    for node_idx, nid in enumerate(dline3_nodes):
        f.write(f'  - "NODE {nid} DLINE 3"\n')
    for node_idx, nid in enumerate(dline4_nodes):
        f.write(f'  - "NODE {nid} DLINE 4"\n\n')
    
    f.write('DSURF-NODE TOPOLOGY:\n')
    for nid, _, _, _ in nodes:
        f.write(f'  - "NODE {nid} DSURFACE 1"\n\n')
    
    f.write('DNODE-NODE TOPOLOGY:\n')
    for dnode_id, nid in dnode_nodes:
        f.write(f'  - "NODE {nid} DNODE {dnode_id}"\n\n')

print(f"Deck generated: slab_T.4C.yaml")
print(f"  Nodes: {total_nodes}")
print(f"  Elements: {total_elems}")
print(f"  Interface nodes: {len(interface_T)}")
