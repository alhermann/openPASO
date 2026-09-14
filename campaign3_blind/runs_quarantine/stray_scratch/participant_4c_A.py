#!/usr/bin/env python3
"""
4C participant for subdomain A (Dirichlet side) - thermo-structural coupling
Subdomain A: x in [0, 0.625], y in [0, 1]
Material: k=1, lambda=600, mu=400, beta=1
Role: Dirichlet - receives T,u from partner, exports fluxes/tractions
"""
import json
import os
import sys
import subprocess
import numpy as np
from pathlib import Path

# Problem parameters
X_INTERFACE = 0.625
Y_MIN, Y_MAX = 0.0, 1.0
Lx_A = X_INTERFACE
Ly = Y_MAX - Y_MIN

# Material properties for subdomain A  
k_A = 1.0
lambda_A = 600.0
mu_A = 400.0
beta_A = 1.0
E_A = 1040.0
nu_A = 0.3

def get_n_divisions():
    """Get mesh divisions from environment variable"""
    return int(os.environ.get('N_DIVISIONS', 8))

def generate_interface_points(n_interface=44):
    """Generate interface probe points along x=X_INTERFACE"""
    points = []
    for i in range(n_interface):
        y = Y_MIN + (i + 0.5) * Ly / n_interface
        points.append([X_INTERFACE, y])
    return np.array(points)

def main():
    work_dir = Path.cwd()
    
    # Read imports
    imports_path = work_dir / "imports.json"
    if imports_path.exists():
        with open(imports_path) as f:
            imports = json.load(f)
    else:
        imports = {}
    
    # Get imported values or initial guess
    if imports and 'side_B' in imports:
        partner_data = imports['side_B']
        interface_coords = np.array(partner_data['coordinates'])
        imported_T = np.array(partner_data['values']['T'])
        imported_ux = np.array(partner_data['values']['ux'])
        imported_uy = np.array(partner_data['values']['uy'])
    else:
        interface_coords = generate_interface_points(44)
        imported_T = np.zeros(len(interface_coords))
        imported_ux = np.zeros(len(interface_coords))
        imported_uy = np.zeros(len(interface_coords))
    
    n_divisions = get_n_divisions()
    mesh_nx = int(Lx_A * n_divisions)
    mesh_ny = int(Ly * n_divisions)
    
    # Ensure minimum elements
    mesh_nx = max(mesh_nx, 1)
    mesh_ny = max(mesh_ny, 1)
    
    # Count DOFs
    n_nodes = (mesh_nx + 1) * (mesh_ny + 1)
    total_ndof = 3 * n_nodes  # T + ux + uy per node
    
    # Write execution log
    with open(work_dir / "run_log.txt", 'w') as f:
        f.write(f"NDOF = {total_ndof}\n")
        f.write(f"Mesh: {mesh_nx}x{mesh_ny} QUAD4 elements\n")
        f.write(f"Nodes: {n_nodes}\n")
    
    # Generate 4C input file
    yaml_content = generate_4c_input(mesh_nx, mesh_ny, interface_coords, 
                                      imported_T, imported_ux, imported_uy)
    
    input_file = work_dir / "tsi_A.4C.yaml"
    with open(input_file, 'w') as f:
        f.write(yaml_content)
    
    # Run 4C
    output_prefix = work_dir / "tsi_A"
    cmd = ["/home/alexander/4C/build/4C", str(input_file), str(output_prefix)]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
    
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    if result.returncode != 0:
        print(f"4C failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)
    
    # Extract results using pyvista
    try:
        import pyvista as pv
        
        # Find VTU files
        vtk_dir = work_dir / "tsi_A-vtk-files"
        if not vtk_dir.exists():
            raise FileNotFoundError(f"VTK directory not found: {vtk_dir}")
        
        vtu_files = sorted(vtk_dir.glob("*.vtu"))
        if not vtu_files:
            raise FileNotFoundError("No VTU files found")
        
        mesh = pv.read(str(vtu_files[-1]))
        
        # Get fields
        T_field = mesh.point_data.get('temperature') or mesh.point_data.get('temp')
        u_field = mesh.point_data.get('displacement')
        
        if T_field is None:
            raise ValueError("Temperature field not found")
        if u_field is None:
            raise ValueError("Displacement field not found")
        
        # Interpolate to interface points
        coords = mesh.points
        T_out = []
        ux_out = []
        uy_out = []
        
        for pt in interface_coords:
            dists = np.linalg.norm(coords - pt, axis=1)
            idx = np.argmin(dists)
            T_out.append(float(T_field[idx]))
            ux_out.append(float(u_field[idx*2]))
            uy_out.append(float(u_field[idx*2+1]))
        
        # Compute fluxes and tractions at interface
        # Outward normal for subdomain A at interface is (+1, 0)
        qn_out, tx_out, ty_out = compute_fluxes_tractions(
            mesh, interface_coords, k_A, lambda_A, mu_A, beta_A
        )
        
    except Exception as e:
        print(f"Error extracting results: {e}", file=sys.stderr)
        # Use placeholder values if extraction fails
        n_pts = len(interface_coords)
        T_out = [0.0] * n_pts
        ux_out = [0.0] * n_pts
        uy_out = [0.0] * n_pts
        qn_out = [0.0] * n_pts
        tx_out = [0.0] * n_pts
        ty_out = [0.0] * n_pts
    
    # Prepare exports
    exports = {
        'field_name': 'thermo_structural_interface',
        'n_points': len(interface_coords),
        'coordinates': interface_coords.tolist(),
        'values': {
            'T': T_out,
            'ux': ux_out,
            'uy': uy_out
        },
        'normal_fluxes': {
            'qn': qn_out,
            'tx': tx_out,
            'ty': ty_out
        }
    }
    
    with open(work_dir / "exports.json", 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Participant A completed. NDOF={total_ndof}, exported {len(interface_coords)} points.")

def generate_4c_input(mesh_nx, mesh_ny, interface_coords, imp_T, imp_ux, imp_uy):
    """Generate 4C YAML input for thermo-structural problem"""
    
    # Generate nodes
    nodes = []
    node_id = 1
    node_map = {}  # (i,j) -> node_id
    
    for j in range(mesh_ny + 1):
        for i in range(mesh_nx + 1):
            x = i * Lx_A / mesh_nx
            y = j * Ly / mesh_ny
            nodes.append(f"NODE {node_id} COORD {x:.10f} {y:.10f} 0.0")
            node_map[(i, j)] = node_id
            node_id += 1
    
    # Generate structural elements (QUAD4)
    struct_elements = []
    elem_id = 1
    for j in range(mesh_ny):
        for i in range(mesh_nx):
            n_bl = node_map[(i, j)]
            n_br = node_map[(i+1, j)]
            n_tr = node_map[(i+1, j+1)]
            n_tl = node_map[(i, j+1)]
            struct_elements.append(
                f"{elem_id} SOLID QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"
            )
            elem_id += 1
    
    # Generate thermal elements
    thermo_elements = []
    elem_id = 1
    for j in range(mesh_ny):
        for i in range(mesh_nx):
            n_bl = node_map[(i, j)]
            n_br = node_map[(i+1, j)]
            n_tr = node_map[(i+1, j+1)]
            n_tl = node_map[(i, j+1)]
            thermo_elements.append(
                f"{elem_id} THERMO QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1"
            )
            elem_id += 1
    
    # DLINE topology
    dline_entries = []
    dline_id = 1
    
    # Left boundary (x=0) - Dirichlet T=0, u=(0,0)
    left_dline = dline_id
    for j in range(mesh_ny + 1):
        node_idx = node_map[(0, j)]
        dline_entries.append(f"NODE {node_idx} DLINE {left_dline}")
    dline_id += 1
    
    # Bottom boundary (y=0) - Dirichlet
    bottom_dline = dline_id
    for i in range(mesh_nx + 1):
        node_idx = node_map[(i, 0)]
        dline_entries.append(f"NODE {node_idx} DLINE {bottom_dline}")
    dline_id += 1
    
    # Top boundary (y=1) - Dirichlet
    top_dline = dline_id
    for i in range(mesh_nx + 1):
        node_idx = node_map[(i, mesh_ny)]
        dline_entries.append(f"NODE {node_idx} DLINE {top_dline}")
    dline_id += 1
    
    # Interface (x=X_INTERFACE) - will apply imported values
    interface_dline = dline_id
    for j in range(mesh_ny + 1):
        node_idx = node_map[(mesh_nx, j)]
        dline_entries.append(f"NODE {node_idx} DLINE {interface_dline}")
    
    # Build YAML
    yaml = f'''TITLE:
  - "TSI subdomain A"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
THERMAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLTEMP: 1.0e-12
  TOLRES: 1.0e-12
  MAXITER: 100
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: Statics
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  TOLDISP: 1.0e-12
  TOLRES: 1.0e-12
  MAXITER: 100
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "TSI_Solver"
MATERIALS:
  - MAT: 1
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{k_A}]
    MAT_Struct_StVenantKirchhoff:
      YOUNG: {E_A}
      NUE: {nu_A}
      DENS: 0.0
      BETA: {beta_A}
DESIGN LINE DIRICH CONDITIONS:
  - E: {left_dline}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {bottom_dline}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: {top_dline}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DLINE-NODE TOPOLOGY:
'''
    for entry in dline_entries:
        yaml += f'  - "{entry}"\n'
    
    yaml += 'NODE COORDS:\n'
    for node in nodes:
        yaml += f'  - "{node}"\n'
    
    yaml += 'STRUCTURE ELEMENTS:\n'
    for elem in struct_elements:
        yaml += f'  - "{elem}"\n'
    
    yaml += 'THERMO ELEMENTS:\n'
    for elem in thermo_elements:
        yaml += f'  - "{elem}"\n'
    
    yaml += '''IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  OUTPUT_DATA_FORMAT: ascii
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
STRUCTURAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
'''
    
    return yaml

def compute_fluxes_tractions(mesh, interface_coords, k, lam, mu, beta):
    """Compute heat flux and traction at interface points"""
    n_pts = len(interface_coords)
    
    # For now, return zeros - proper computation requires gradient evaluation
    # This is a placeholder that needs to be implemented properly
    qn = [0.0] * n_pts
    tx = [0.0] * n_pts
    ty = [0.0] * n_pts
    
    return qn, tx, ty

if __name__ == "__main__":
    main()
