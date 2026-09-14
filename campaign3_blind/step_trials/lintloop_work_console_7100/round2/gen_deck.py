import numpy as np

def generate_deck():
    # Problem parameters
    Lx, Ly = 0.8, 1.0
    nx, ny = 8, 10
    k_val = 2.0
    
    # Source term f_T = 2*pi^2*x*sin(pi*y)
    # 4C syntax uses ^ for power
    pi_str = "3.141592653589793"
    source_expr = f"2*{pi_str}^{2}*x*sin({pi_str}*y)"

    # Mesh generation
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    
    node_map = {}
    coord_lines = []
    
    current_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            node_map[(i, j)] = current_id
            coord_lines.append(f'    - "NODE {current_id} COORD {x:.16f} {y:.16f} 0.0"')
            current_id += 1
            
    num_nodes = len(coord_lines)
    
    # Elements: TRANSP QUAD4 for Scalar_Transport (Point 11)
    element_lines = []
    elem_id = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            element_lines.append(f'    - "{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
            elem_id += 1
            
    num_elements = len(element_lines)
    
    # Boundary Conditions
    # DLINE IDs: 1=Left(x=0), 2=Bottom(y=0), 3=Top(y=1), 4=Right(x=0.8, interface)
    dline_topology = []
    dnode_topology = []
    
    # DLINE 1: Left (x=0)
    for j in range(ny + 1):
        nid = node_map[(0, j)]
        dline_topology.append(f'    - "NODE {nid} DLINE 1"')
        
    # DLINE 2: Bottom (y=0)
    for i in range(nx + 1):
        nid = node_map[(i, 0)]
        dline_topology.append(f'    - "NODE {nid} DLINE 2"')
        
    # DLINE 3: Top (y=1)
    for i in range(nx + 1):
        nid = node_map[(i, ny)]
        dline_topology.append(f'    - "NODE {nid} DLINE 3"')
        
    # DLINE 4: Right interface (x=0.8) - for Flux Calc
    for j in range(ny + 1):
        nid = node_map[(nx, j)]
        dline_topology.append(f'    - "NODE {nid} DLINE 4"')
        
    # DNODEs: Right interface (x=0.8) - for Point Dirichlet values
    interface_vals = [
        0.0000000000, 0.2472135955, 0.4702282018, 0.6472135955, 0.7608452130,
        0.8000000000, 0.7608452130, 0.6472135955, 0.4702282018, 0.2472135955, 0.0000000000
    ]
    
    point_dirich_conditions = []
    for j, val in enumerate(interface_vals):
        nid = node_map[(nx, j)]
        dnode_id = j + 1
        dnode_topology.append(f'    - "NODE {nid} DNODE {dnode_id}"')
        point_dirich_conditions.append(f'''    - E: {dnode_id}
      NUMDOF: 1
      ONOFF: [1]
      VAL: [{val}]
      FUNCT: [0]''')
      
    # Source Term: DESIGN SURF TRANSPORT NEUMANN (Point 2, 11)
    dsurf_topology = []
    for eid in range(1, num_elements + 1):
        j_elem = (eid - 1) // nx
        i_elem = (eid - 1) % nx
        n1 = node_map[(i_elem, j_elem)]
        n2 = node_map[(i_elem+1, j_elem)]
        n3 = node_map[(i_elem+1, j_elem+1)]
        n4 = node_map[(i_elem, j_elem+1)]
        for n in [n1, n2, n3, n4]:
            dsurf_topology.append(f'    - "NODE {n} DSURFACE 1"')
            
    # Construct Deck YAML
    deck = []
    deck.append("TITLE:")
    deck.append('    - "Steady Heat Conduction Standalone"')
    deck.append("PROBLEM SIZE:")
    deck.append(f"    ELEMENTS: {num_elements}")
    deck.append(f"    NODES: {num_nodes}")
    deck.append("PROBLEM TYPE:")
    deck.append('    PROBLEMTYPE: "Scalar_Transport"')
    
    # Scalar Transport Dynamic with Flux Calc enabled (Point 8)
    deck.append("SCALAR TRANSPORT DYNAMIC:")
    deck.append('    TIMEINTEGR: "Stationary"')
    deck.append('    SOLVERTYPE: "linear_full"')
    deck.append("    NUMSTEP: 1")
    deck.append("    TIMESTEP: 1.0")
    deck.append("    MAXTIME: 1.0")
    deck.append("    LINEAR_SOLVER: 1")
    deck.append('    CALCFLUX_BOUNDARY: "diffusive"')
    
    deck.append("SOLVER 1:")
    deck.append('    SOLVER: "UMFPACK"')
    
    deck.append("MATERIALS:")
    deck.append("    - MAT: 1")
    deck.append("      MAT_scatra:")
    deck.append(f"        DIFFUSIVITY: {k_val}")
    
    deck.append("FUNCT1:")
    deck.append(f'    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_expr}"')
    
    # Conditions
    # 1. Line Dirichlet (Outer boundaries)
    deck.append("DESIGN LINE DIRICH CONDITIONS:")
    deck.append("    - E: 1")
    deck.append("      NUMDOF: 1")
    deck.append("      ONOFF: [1]")
    deck.append("      VAL: [0.0]")
    deck.append("      FUNCT: [0]")
    deck.append("    - E: 2")
    deck.append("      NUMDOF: 1")
    deck.append("      ONOFF: [1]")
    deck.append("      VAL: [0.0]")
    deck.append("      FUNCT: [0]")
    deck.append("    - E: 3")
    deck.append("      NUMDOF: 1")
    deck.append("      ONOFF: [1]")
    deck.append("      VAL: [0.0]")
    deck.append("      FUNCT: [0]")
    
    # 2. Point Dirichlet (Interface)
    deck.append("DESIGN POINT DIRICH CONDITIONS:")
    for cond in point_dirich_conditions:
        deck.append(cond)
        
    # 3. Surf Neumann (Source)
    deck.append("DESIGN SURF TRANSPORT NEUMANN CONDITIONS:")
    deck.append("    - E: 1")
    deck.append("      NUMDOF: 1")
    deck.append("      ONOFF: [1]")
    deck.append("      VAL: [1.0]")
    deck.append("      FUNCT: [1]")
    
    # 4. Flux Calc Line (Interface) - Point 8: SCATRA FLUX CALC LINE CONDITIONS
    deck.append("SCATRA FLUX CALC LINE CONDITIONS:")
    deck.append("    - E: 4")
    
    # Coordinates
    deck.append("NODE COORDS:")
    deck.extend(coord_lines)
    
    # Elements
    deck.append("TRANSPORT ELEMENTS:")
    deck.extend(element_lines)
    
    # Topologies
    deck.append("DLINE-NODE TOPOLOGY:")
    deck.extend(dline_topology)
    
    deck.append("DNODE-NODE TOPOLOGY:")
    deck.extend(dnode_topology)
    
    deck.append("DSURF-NODE TOPOLOGY:")
    deck.extend(dsurf_topology)
    
    with open('./slab_T.4C.yaml', 'w') as f:
        f.write('\n'.join(deck))

if __name__ == "__main__":
    generate_deck()
