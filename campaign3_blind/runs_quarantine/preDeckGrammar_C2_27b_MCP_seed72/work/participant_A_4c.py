#!/usr/bin/env python3
"""
Participant A (4C): Subdomain (0, 0.625) x (0, 1), k = 1
Role: DIRICHLET side - receives temperature from B, applies as Dirichlet BC on right edge
      Exports outward normal flux (normal points +x direction)
"""

import os
import json
import numpy as np
import subprocess
import sys
from pathlib import Path

# Problem parameters
L_A = 0.625
H = 1.0
k_A = 1.0
x_interface = 0.625

# Source term for 4C (using ^ not **)
f_A_4c = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

def get_mesh_params():
    """Get mesh divisions from level.json or default"""
    level_file = Path("level.json")
    if level_file.exists():
        with open(level_file) as f:
            return json.load(f)["divisions"]
    return 8  # default h=1/8

def generate_mesh(divisions):
    """Generate uniform triangular mesh for subdomain A"""
    nx = int(L_A * divisions)
    ny = int(H * divisions)
    
    nodes = []
    node_map = {}
    nid = 1
    
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L_A / nx
            y = j * H / ny
            nodes.append((nid, x, y))
            node_map[(i, j)] = nid
            nid += 1
    
    elements = []
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            # Two triangles per quad
            elements.append((eid, n1, n2, n4))
            eid += 1
            elements.append((eid, n2, n3, n4))
            eid += 1
    
    return nodes, elements, nx, ny

def generate_4c_deck(nodes, elements, divisions, imports=None):
    """Generate 4C YAML deck for subdomain A"""
    
    nx = int(L_A * divisions)
    ny = int(H * divisions)
    
    # Build DLINE topology for boundaries
    # Line 1: left (x=0), Line 2: bottom (y=0), Line 3: top (y=1), Line 4: right (x=0.625)
    dline_nodes = {1: [], 2: [], 3: [], 4: []}
    
    for nid, x, y in nodes:
        tol = 1e-9
        if abs(x) < tol:
            dline_nodes[1].append(nid)  # left
        if abs(y) < tol:
            dline_nodes[2].append(nid)  # bottom
        if abs(y - H) < tol:
            dline_nodes[3].append(nid)  # top
        if abs(x - L_A) < tol:
            dline_nodes[4].append(nid)  # right (interface)
    
    # Generate NODE COORDS section
    node_coords = [f'NODE {nid} COORD {x:.10f} {y:.10f} 0.0' for nid, x, y in nodes]
    
    # Generate TRANSPORT ELEMENTS section
    transport_elements = [f'{eid} TRANSP TRI6 {n1} {n2} {n4} MAT 1 TYPE Std' 
                          for eid, n1, n2, n4 in elements]
    
    # Generate DLINE-NODE TOPOLOGY
    dline_topology = []
    for line_id, node_ids in dline_nodes.items():
        for nid in node_ids:
            dline_topology.append(f'NODE {nid} DLINE {line_id}')
    
    # Build deck
    deck = f'''TITLE:
  - "Coupled heat conduction - Subdomain A (4C)"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "direct_solver"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {k_A}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{f_A_4c}"
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''
    
    # Add interface Dirichlet condition if we have imports
    if imports is not None and "B" in imports:
        temp_data = imports["B"]
        iface_temps = temp_data.get("values", [])
        iface_coords = temp_data.get("coordinates", [])
        
        # Find interface nodes and their temperatures
        # We need to interpolate imported temps to our interface nodes
        interface_nodes = dline_nodes[4]
        
        # For simplicity, use nearest neighbor interpolation
        # In a real implementation, we'd do proper projection
        if len(iface_temps) > 0 and len(interface_nodes) > 0:
            # Create a mapping from y-coordinate to temperature
            y_to_temp = {coord[1]: val for coord, val in zip(iface_coords, iface_temps)}
            
            # Get temperatures for our interface nodes
            iface_temps_local = []
            for nid in interface_nodes:
                _, x, y = next((n for n in nodes if n[0] == nid), (0, 0, 0))
                # Find nearest imported temperature
                min_dist = float('inf')
                nearest_temp = 0.0
                for iy, itemp in y_to_temp.items():
                    dist = abs(y - iy)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_temp = itemp
                iface_temps_local.append(nearest_temp)
            
            # Write interface temperatures to a file for reading
            with open("interface_temps_A.txt", "w") as f:
                for nid, temp in zip(interface_nodes, iface_temps_local):
                    f.write(f"{nid} {temp}\n")
            
            # For now, use a simple approach: apply average temperature
            avg_temp = sum(iface_temps_local) / len(iface_temps_local) if iface_temps_local else 0.0
            deck += f'''  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{avg_temp}]
    FUNCT: [0]
'''
        else:
            # No imports yet, use zero
            deck += '''  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''
    else:
        # First iteration, no imports
        deck += '''  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''
    
    deck += "\nDLINE-NODE TOPOLOGY:\n"
    for entry in dline_topology:
        deck += f'  - "{entry}"\n'
    
    deck += "\nNODE COORDS:\n"
    for entry in node_coords:
        deck += f'  - "{entry}"\n'
    
    deck += "\nTRANSPORT ELEMENTS:\n"
    for entry in transport_elements:
        deck += f'  - "{entry}"\n'
    
    return deck

def run_4c(deck_content, output_prefix, work_dir):
    """Run 4C binary"""
    # Write deck
    deck_file = os.path.join(work_dir, f"{output_prefix}.4C.yaml")
    with open(deck_file, "w") as f:
        f.write(deck_content)
    
    # Run 4C
    cmd = ["LD_LIBRARY_PATH=/opt/4C-dependencies/lib", 
           "/home/alexander/4C/build/4C", 
           deck_file, 
           os.path.join(work_dir, output_prefix)]
    
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    
    return result.stdout + result.stderr, result.returncode

def extract_results(output_prefix, work_dir):
    """Extract solution from 4C VTU output"""
    import pyvista as pv
    
    # Find the last VTU file
    vtu_files = list(Path(work_dir).glob(f"{output_prefix}-vtk-files/scatra-*.vtu"))
    if not vtu_files:
        # Try pvtu
        vtu_files = list(Path(work_dir).glob(f"{output_prefix}-vtk-files/scatra-*.pvtu"))
    
    if not vtu_files:
        raise FileNotFoundError("No VTU files found")
    
    # Sort and take the last one
    vtu_files.sort()
    last_vtu = str(vtu_files[-1])
    
    mesh = pv.read(last_vtu)
    
    # Get nodal values
    coords = mesh.points[:, :2]
    values = mesh.point_data['phi_1']  # 4C uses phi_1 for scalar transport
    
    return coords, values, mesh

def compute_interface_flux(coords, values, mesh, interface_nodes, nodes_list):
    """Compute outward normal flux at interface (x = 0.625)"""
    # For P1 elements, flux = -k * grad(u) . n
    # At interface, n = (1, 0) (outward from A)
    # So flux = -k * du/dx
    
    # Find elements touching the interface
    interface_fluxes = []
    
    # Use pyvista to compute gradients
    mesh.compute_normals(cell_normals=False, point_normals=False)
    
    # Compute gradient of phi
    grad = mesh.point_data['Gradients'] if 'Gradients' in mesh.point_data else None
    
    if grad is None:
        # Manual computation using finite differences
        # For each interface node, find neighboring element and compute gradient
        pass
    
    # Simpler approach: use the fact that for linear elements,
    # we can compute exact gradients within each element
    interface_y_coords = []
    interface_qn_values = []
    
    # Get interface nodes sorted by y
    iface_node_info = [(nid, x, y) for nid, x, y in nodes_list if abs(x - L_A) < 1e-6]
    iface_node_info.sort(key=lambda t: t[2])
    
    for nid, x, y in iface_node_info:
        interface_y_coords.append(y)
        # Find the element to the left of this node
        # For now, approximate using central difference
        # This is a simplification - proper implementation needs element-wise gradient
        
        # Find node index in coords
        idx = np.argmin(np.linalg.norm(coords - np.array([x, y]), axis=1))
        
        # Approximate gradient using nearby nodes
        # Find nodes with similar y but smaller x
        nearby = coords[np.abs(coords[:, 1] - y) < 0.01]
        nearby_vals = values[np.abs(coords[:, 1] - y) < 0.01]
        nearby_x = coords[np.abs(coords[:, 1] - y) < 0.01, 0]
        
        if len(nearby) >= 2:
            # Linear fit to estimate du/dx
            coeffs = np.polyfit(nearby_x, nearby_vals, 1)
            du_dx = coeffs[0]
            qn = -k_A * du_dx  # Outward normal is +x
        else:
            qn = 0.0
        
        interface_qn_values.append(qn)
    
    return interface_y_coords, interface_qn_values

def main():
    work_dir = os.getcwd()
    
    # Read level info
    divisions = get_mesh_params()
    
    # Read imports if available
    imports = {}
    imports_file = os.path.join(work_dir, "imports.json")
    if os.path.exists(imports_file):
        with open(imports_file) as f:
            imports = json.load(f)
    
    # Generate mesh
    nodes, elements, nx, ny = generate_mesh(divisions)
    n_dof = len(nodes)
    
    # Generate and run 4C deck
    deck = generate_4c_deck(nodes, elements, divisions, imports)
    output_prefix = "subdomain_A"
    
    log_output, returncode = run_4c(deck, output_prefix, work_dir)
    
    # Write log
    with open(os.path.join(work_dir, "run.log"), "w") as f:
        f.write(log_output)
        f.write(f"\nNDOF = {n_dof}\n")
    
    if returncode != 0:
        print(f"4C failed with return code {returncode}")
        print(log_output)
        sys.exit(returncode)
    
    # Extract results
    coords, values, mesh = extract_results(output_prefix, work_dir)
    
    # Compute interface flux
    interface_nodes = [nid for nid, x, y in nodes if abs(x - L_A) < 1e-6]
    interface_y, interface_flux = compute_interface_flux(coords, values, mesh, interface_nodes, nodes)
    
    # Prepare exports
    exports = {
        "field_name": "temperature",
        "coordinates": [[x_interface, y] for y in interface_y],
        "values": interface_flux,  # Export flux (Dirichlet side exports flux)
        "normal_fluxes": interface_flux,
        "n_points": len(interface_y)
    }
    
    # Write exports
    with open(os.path.join(work_dir, "exports.json"), "w") as f:
        json.dump(exports, f, indent=2)
    
    # Save solution for post-processing
    np.savez(os.path.join(work_dir, "solution_A.npz"), 
             coords=coords, values=values, nodes=np.array(nodes))
    
    print(f"Participant A completed. NDOF={n_dof}, Interface points={len(interface_y)}")

if __name__ == "__main__":
    main()
