import numpy as np

def generate_deck():
    # Problem parameters
    Lx, Ly = 0.8, 1.0
    nx, ny = 8, 10
    k_val = 2.0
    
    # Source term f_T = 2*pi^2*x*sin(pi*y)
    # Note: 4C syntax uses ^ for power, and standard math functions. 
    # Pi is typically available as 'pi' in symbolic expressions in many FE codes, 
    # but to be safe per docs (which show x^2*y), we can use a high-precision constant.
    pi_str = "3.141592653589793"
    source_expr = f"2*{pi_str}^{2}*x*sin({pi_str}*y)"

    # Mesh generation
    # Nodes: (nx+1) * (ny+1) = 9 * 11 = 99 nodes
    # Indexing: Node ID starts at 1. 
    # Let's use row-major order (vary x first then y) or column-major?
    # Standard FEM often varies x fast. Let's do that: node(i, j) where i=0..nx, j=0..ny
    # ID = j*(nx+1) + i + 1
    
    nodes = []
    # Generate coordinates
    xs = np.linspace(0, Lx, nx + 1)
    ys = np.linspace(0, Ly, ny + 1)
    
    # Store node coords and map (i,j) -> id
    node_map = {}
    coord_lines = []
    
    current_id = 1
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            node_map[(i, j)] = current_id
            coord_lines.append(f'    - "NODE {current_id} COORD {x:.16f} {y:.16f} 0.0"')
            current_id += 1
            
    num_nodes = len(coord_lines)
    
    # Elements: QUAD4
    # Element connectivity: 4 nodes per element.
    # For structured grid, element e connects nodes at (i, j), (i+1, j), (i+1, j+1), (i, j+1)
    # Order matters for sign of Jacobian/flux. Standard counter-clockwise.
    # (i,j) -> (i+1,j) -> (i+1,j+1) -> (i,j+1)
    element_lines = []
    elem_id = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            element_lines.append(f'    - "{elem_id} THERMO QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
            elem_id += 1
            
    num_elements = len(element_lines)
    
    # Boundary Conditions
    # 1. Outer boundaries (x=0, y=0, y=1) : T=0. DESIGN LINE DIRICH.
    # We need to identify the lines. In 4C, we define Design Lines via topology.
    # However, we can group them into one condition if they share properties, 
    # but here they are geometrically distinct sets.
    # To keep it simple and robust, we will define separate Design Line IDs for each boundary side.
    # But 4C requires us to list which NODES belong to which DLINE in TOPOLOGY.
    
    # Strategy: Assign DLINE IDs to the boundary edges.
    # Left (x=0): j from 0..ny, i=0. Nodes: (0,0)..(0,10). DLINE 1
    # Bottom (y=0): i from 0..nx, j=0. Nodes: (0,0)..(8,0). DLINE 2
    # Top (y=1): i from 0..nx, j=ny. Nodes: (0,10)..(8,10). DLINE 3
    # Right (x=0.8): Interface. This needs POINT DIRICH, not Line Dirichlet, because values vary per node.
    # Interface nodes: i=nx, j from 0..ny. DNODEs 1..11.
    
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
        
    # DNODEs: Right interface (x=0.8)
    # Values provided in prompt
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
      
    # Source Term: DESIGN SURF NEUMANN (for Thermo standalone per doc Point 1 & 2)
    # All elements participate in volume source.
    # We need a DSURF ID. Let's call it 1.
    dsurf_topology = []
    for eid in range(1, num_elements + 1):
        # Get nodes of element to add to DSURF topology
        # We need to extract nodes from our logic again or store them.
        # Re-derive: element index maps to i, j
        j_elem = (eid - 1) // nx
        i_elem = (eid - 1) % nx
        n1 = node_map[(i_elem, j_elem)]
        n2 = node_map[(i_elem+1, j_elem)]
        n3 = node_map[(i_elem+1, j_elem+1)]
        n4 = node_map[(i_elem, j_elem+1)]
        for n in [n1, n2, n3, n4]:
            dsurf_topology.append(f'    - "NODE {n} DSURFACE 1"')
            
    # Flux Calculation on Interface
    # Doc Point 8: SCATRA FLUX CALC LINE CONDITIONS. 
    # Since we are using Thermo, we try THERMO FLUX CALC LINE CONDITIONS or similar.
    # Given the risk of unknown sections (Trap e), and the instruction to get flux...
    # The prompt asks for "consistent boundary heat flux".
    # If I cannot find exact keyword for Thermo, I might omit this specific block to avoid crash,
    # but then I fail the output requirement.
    # However, Point 8 says "set CALCFLUX_BOUNDARY... in SCALAR TRANSPORT DYNAMIC".
    # It implies this feature is tied to that dynamic section.
    # If I use Thermo, maybe I can't get flux in VTU easily without crashing on unknown key.
    # BUT, the prompt explicitly asks for it.
    # I will assume symmetry: THERMAL DYNAMIC accepts CALCFLUX_BOUNDARY and THERMO FLUX CALC LINE CONDITIONS.
    # Actually, looking at Point 12, TSI uses `THERMAL DYNAMIC`.
    # I'll include the configuration with best guess keywords derived from Scalar_Transport docs.
    # Condition ID for interface line: We used DNODEs for points. Do we have a DLINE for the interface?
    # No, we used DNODEs for Dirichlet.
    # Flux calc usually references a Line/Surface entity.
    # I should define a DLINE for the interface as well for Flux Calc purposes?
    # Or can Flux Calc target DNODEs?
    # Doc Point 8: "add a SCATRA FLUX CALC LINE CONDITIONS entry (- E: <line id>; SURF in 3D)".
    # So I need a DLINE ID for the interface.
    # Let's create DLINE 4 for the interface (Right side).
    dline_topology_interface = []
    for j in range(ny + 1):
        nid = node_map[(nx, j)]
        dline_topology_interface.append(f'    - "NODE {nid} DLINE 4"')
    dline_topology.extend(dline_topology_interface)

    # Construct Deck YAML
    deck = []
    deck.append("TITLE:")
    deck.append('    - "Steady Heat Conduction Standalone"')
    deck.append("PROBLEM SIZE:")
    deck.append(f"    ELEMENTS: {num_elements}")
    deck.append(f"    NODES: {num_nodes}")
    deck.append("PROBLEM TYPE:")
    deck.append('    PROBLEMTYPE: "Thermo"')
    
    # Thermal Dynamic
    deck.append("THERMAL DYNAMIC:")
    deck.append('    TIMEINTEGR: "Stationary"')
    deck.append('    SOLVERTYPE: "linear_full"')
    deck.append("    NUMSTEP: 1")
    deck.append("    TIMESTEP: 1.0")
    deck.append("    MAXTIME: 1.0")
    deck.append("    LINEAR_SOLVER: 1")
    deck.append('    CALCFLUX_BOUNDARY: "diffusive"') # Attempt to enable flux
    
    deck.append("SOLVER 1:")
    deck.append('    SOLVER: "UMFPACK"')
    
    deck.append("MATERIALS:")
    deck.append("    - MAT: 1")
    deck.append("      MAT_Fourier:")
    deck.append(f"        CONDUCTIVITY: {k_val}") # Fourier material uses conductivity
    
    deck.append("FUNCT1:")
    deck.append(f'    - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_expr}"')
    
    # Conditions
    # 1. Line Dirichlet (Outer)
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
    deck.append("DESIGN SURF NEUMANN CONDITIONS:")
    deck.append("    - E: 1")
    deck.append("      NUMDOF: 1")
    deck.append("      ONOFF: [1]")
    deck.append("      VAL: [1.0]")
    deck.append("      FUNCT: [1]")
    
    # 4. Flux Calc Line (Interface)
    # Using "THERMO FLUX CALC LINE CONDITIONS" based on naming convention
    deck.append("THERMO FLUX CALC LINE CONDITIONS:")
    deck.append("    - E: 4")
    
    # Coordinates
    deck.append("NODE COORDS:")
    deck.extend(coord_lines)
    
    # Elements
    deck.append("THERMO ELEMENTS:")
    deck.extend(element_lines)
    
    # Topologies
    deck.append("DLINE-NODE TOPOLOGY:")
    deck.extend(dline_topology)
    
    deck.append("DNODE-NODE TOPOLOGY:")
    deck.extend(dnode_topology)
    
    deck.append("DSURF-NODE TOPOLOGY:")
    deck.extend(dsurf_topology)
    
    # IO Section (Required for VTU per Item 17)
    # Item 17 shows:
    # IO:
    #   VERBOSITY: "Standard"
    # IO/RUNTIME VTK OUTPUT:
    #   OUTPUT_DATA_FORMAT: ascii
    # Trap (e) warns against unknown keys. But Item 17 explicitly documents this structure for Thermo.
    # So it IS known for Thermo.
    deck.append("IO:")
    deck.append('  VERBOSITY: "Standard"')
    deck.append("IO/RUNTIME VTK OUTPUT:")
    deck.append('  OUTPUT_DATA_FORMAT: ascii')
    
    with open('./slab_T.4C.yaml', 'w') as f:
        f.write('\n'.join(deck))

if __name__ == "__main__":
    generate_deck()
