import numpy as np

def main():
    # Problem Parameters
    x_max = 0.8
    y_max = 1.0
    nx = 8
    ny = 10
    
    # Mesh Generation
    # Nodes: (nx+1) x (ny+1) = 9 x 11 = 99
    # Elements: nx x ny = 8 x 10 = 80
    nodes = []
    node_map = {} # (i, j) -> global_id (1-based)
    
    global_id = 1
    for i in range(nx + 1):
        for j in range(ny + 1):
            x = i * (x_max / nx)
            y = j * (y_max / ny)
            nodes.append((global_id, x, y, 0.0))
            node_map[(i, j)] = global_id
            global_id += 1
            
    elements = []
    elem_id = 1
    for i in range(nx):
        for j in range(ny):
            n1 = node_map[(i, j)]         # Bottom-Left
            n2 = node_map[(i+1, j)]       # Bottom-Right
            n3 = node_map[(i+1, j+1)]     # Top-Right
            n4 = node_map[(i, j+1)]       # Top-Left
            # Format: "id THERMO QUAD4 n1 n2 n3 n4 MAT 1 TYPE Std"
            elements.append(f"{elem_id} THERMO QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
            elem_id += 1
            
    # Interface Values (x=0.8, i=8)
    # y from 0.0 to 1.0 step 0.1 (11 nodes)
    interface_values = [
        0.0000000000, 0.2472135955, 0.4702282018, 0.6472135955, 0.7608452130,
        0.8000000000, 0.7608452130, 0.6472135955, 0.4702282018, 0.2472135955,
        0.0000000000
    ]
    
    # Topology Generation
    # DLINE 1: Left (x=0, i=0)
    # DLINE 2: Bottom (y=0, j=0)
    # DLINE 3: Top (y=1, j=10)
    # DNODE 1..11: Interface (x=0.8, i=8, j=0..10)
    # DSURFACE 1: All nodes (for Volume Source)
    
    dline_topology = []
    # Left
    for j in range(ny + 1):
        nid = node_map[(0, j)]
        dline_topology.append(f"NODE {nid} DLINE 1")
    # Bottom
    for i in range(nx + 1):
        nid = node_map[(i, 0)]
        dline_topology.append(f"NODE {nid} DLINE 2")
    # Top
    for i in range(nx + 1):
        nid = node_map[(i, ny)]
        dline_topology.append(f"NODE {nid} DLINE 3")
        
    dnode_topology = []
    for j in range(ny + 1):
        nid = node_map[(nx, j)]
        dnode_id = j + 1
        dnode_topology.append(f"NODE {nid} DNODE {dnode_id}")
        
    dsurf_topology = []
    # All nodes belong to the source domain
    for nid, x, y, z in nodes:
        dsurf_topology.append(f"NODE {nid} DSURFACE 1")
        
    # Build Deck
    deck = []
    deck.append("TITLE:")
    deck.append('  - "Standalone Heat Conduction"')
    deck.append("PROBLEM SIZE:")
    deck.append(f"  ELEMENTS: {nx * ny}")
    deck.append(f"  NODES: {(nx + 1) * (ny + 1)}")
    deck.append("PROBLEM TYPE:")
    deck.append('  PROBLEMTYPE: "Thermo"')
    deck.append("THERMO DYNAMIC:")
    deck.append('  TIMEINTEGR: "Stationary"')
    deck.append('  SOLVERTYPE: "linear_full"')
    deck.append("  NUMSTEP: 1")
    deck.append("  TIMESTEP: 1.0")
    deck.append("  MAXTIME: 1.0")
    deck.append("  LINEAR_SOLVER: 1")
    deck.append("SOLVER 1:")
    deck.append('  SOLVER: "UMFPACK"')
    deck.append("MATERIALS:")
    deck.append("  - MAT: 1")
    deck.append("    MAT_Fourier:")
    deck.append("      CONDUCTIVITY: 2.0")
    deck.append("      INITTEMP: 0.0")
    deck.append("FUNCT1:")
    deck.append('  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi^2*x*sin(pi*y)"')
    
    # Conditions
    deck.append("DESIGN LINE DIRICH CONDITIONS:")
    # Left (DLINE 1)
    deck.append("  - E: 1")
    deck.append("    NUMDOF: 1")
    deck.append("    ONOFF: [1]")
    deck.append("    VAL: [0.0]")
    deck.append("    FUNCT: [0]")
    # Bottom (DLINE 2)
    deck.append("  - E: 2")
    deck.append("    NUMDOF: 1")
    deck.append("    ONOFF: [1]")
    deck.append("    VAL: [0.0]")
    deck.append("    FUNCT: [0]")
    # Top (DLINE 3)
    deck.append("  - E: 3")
    deck.append("    NUMDOF: 1")
    deck.append("    ONOFF: [1]")
    deck.append("    VAL: [0.0]")
    deck.append("    FUNCT: [0]")
    
    deck.append("DESIGN POINT DIRICH CONDITIONS:")
    # Interface (DNODE 1..11)
    for idx, val in enumerate(interface_values):
        dnode_id = idx + 1
        deck.append(f"  - E: {dnode_id}")
        deck.append("    NUMDOF: 1")
        deck.append("    ONOFF: [1]")
        deck.append(f"    VAL: [{val}]")
        deck.append("    FUNCT: [0]")
        
    deck.append("DESIGN SURF NEUMANN CONDITIONS:")
    deck.append("  - E: 1")
    deck.append("    NUMDOF: 1")
    deck.append("    ONOFF: [1]")
    deck.append("    VAL: [1.0]")
    deck.append("    FUNCT: [1]")
    
    deck.append("NODE COORDS:")
    for nid, x, y, z in nodes:
        deck.append(f'  - "NODE {nid} COORD {x:.16e} {y:.16e} {z:.16e}"')
        
    deck.append("THERMO ELEMENTS:")
    for elem in elements:
        deck.append(f'  - "{elem}"')
        
    deck.append("DLINE-NODE TOPOLOGY:")
    for line in dline_topology:
        deck.append(f'  - "{line}"')
        
    deck.append("DNODE-NODE TOPOLOGY:")
    for line in dnode_topology:
        deck.append(f'  - "{line}"')
        
    deck.append("DSURF-NODE TOPOLOGY:")
    for line in dsurf_topology:
        deck.append(f'  - "{line}"')
    
    # Write to file
    with open("slab_T.4C.yaml", "w") as f:
        f.write("\n".join(deck))

if __name__ == "__main__":
    main()
