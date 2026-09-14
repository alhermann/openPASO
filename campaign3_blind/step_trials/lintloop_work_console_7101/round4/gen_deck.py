import numpy as np

# Problem parameters
nx_elem = 8          
ny_elem = 10         
Lx = 0.8             
Ly = 1.0             
k = 2.0              

# Mesh parameters
nx_nodes = nx_elem + 1    
ny_nodes = ny_elem + 1    
total_nodes = nx_nodes * ny_nodes  
total_elems = nx_elem * ny_elem    

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
        z = 0.0
        nodes.append((node_id, x, y, z))
        node_id += 1

# Generate element connectivity (QUAD4, counter-clockwise)
elements = []
elem_id = 1
for j in range(ny_elem):
    for i in range(nx_elem):
        n1 = i + 1 + j * nx_nodes           
        n2 = i + 2 + j * nx_nodes           
        n3 = i + 2 + (j + 1) * nx_nodes     
        n4 = i + 1 + (j + 1) * nx_nodes     
        elements.append((elem_id, n1, n2, n3, n4))
        elem_id += 1

# Design line topology for outer boundaries (Dirichlet T=0)
# DLINE 1: left (x=0)
dline1_nodes = [i + 1 + j * nx_nodes for j in range(ny_nodes)]  
# DLINE 2: bottom (y=0)
dline2_nodes = [i + 1 for i in range(nx_nodes)]                  
# DLINE 3: top (y=1)
dline3_nodes = [i + 1 + (ny_nodes - 1) * nx_nodes for i in range(nx_nodes)]  

# Design line topology for interface (x=0.8)
# DLINE 4: right
dline4_nodes = [nx_nodes + j * nx_nodes for j in range(ny_nodes)]  

# DNODE topology for interface point conditions
dnode_nodes = [(j + 1, dline4_nodes[j]) for j in range(ny_nodes)]  

# Write the deck
with open('slab_T.4C.yaml', 'w') as f:
    f.write('TITLE:\n')
    f.write('  - "Coupled Heat Conduction - 4C Standalone"\n\n')
    
    f.write('PROBLEM SIZE:\n')
    f.write(f'  ELEMENTS: {total_elems}\n')
    f.write(f'  NODES: {total_nodes}\n\n')
    
    f.write('PROBLEM TYPE:\n')
    f.write('  PROBLEMTYPE: "Thermo"\n\n')
    
    f.write('THERMAL DYNAMIC:\n')
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
    f.write('    MAT_Fourier:\n')
    f.write(f'      CONDUCTIVITY: {k}\n\n')
    
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
    
    f.write('DESIGN SURF NEUMANN CONDITIONS:\n')
    f.write('  - E: 1\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [1.0]\n')
    f.write('    FUNCT: [1]\n\n')
    
    f.write('IO:\n')
    f.write('  VERBOSITY: "Standard"\n')
    f.write('  RUNTIME VTK OUTPUT:\n')
    f.write('    OUTPUT_DATA_FORMAT: ascii\n\n')
    
    f.write('NODE COORDS:\n')
    for nid, x, y, z in nodes:
        f.write(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} {z:.10f}"\n\n')
    
    f.write('THERMO ELEMENTS:\n')
    for eid, n1, n2, n3, n4 in elements:
        f.write(f'  - "{eid} THERMO QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n\n')
    
    f.write('DLINE-NODE TOPOLOGY:\n')
    for nid in dline1_nodes:
        f.write(f'  - "NODE {nid} DLINE 1"\n')
    for nid in dline2_nodes:
        f.write(f'  - "NODE {nid} DLINE 2"\n')
    for nid in dline3_nodes:
        f.write(f'  - "NODE {nid} DLINE 3"\n')
    for nid in dline4_nodes:
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
