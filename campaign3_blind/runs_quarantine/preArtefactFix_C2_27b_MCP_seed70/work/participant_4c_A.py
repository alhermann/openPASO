#!/usr/bin/env python3
"""
4C Participant for Subdomain A (Dirichlet side) - Coupled thermal diffusion
- Receives temperature from interface (from Kratos/B)
- Applies as Dirichlet BC at x = 0.625
- Returns outward normal flux at interface
- Outward normal for A at interface is +x direction
"""

import json
import os
import sys
import subprocess
import glob
import numpy as np
from pathlib import Path

# Problem parameters
L_A = 0.625  # Width of subdomain A
H = 1.0      # Height
k_A = 1.0    # Conductivity in A
x_interface = 0.625

# Get mesh level and work directory from environment
level = int(os.environ.get('LEVEL', '1'))
work_dir = os.environ.get('WORK_DIR', '.')

# Mesh parameters based on level
mesh_params = {
    1: {"nx_A": 5, "ny": 8},
    2: {"nx_A": 10, "ny": 16},
    3: {"nx_A": 20, "ny": 32}
}[level]

nx = mesh_params["nx_A"]
ny = mesh_params["ny"]

# Source term for subdomain A (polynomial) - converted to 4C syntax (^ instead of **)
source_A_str = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

def read_imports():
    """Read imports.json if it exists."""
    imports_file = os.path.join(work_dir, 'imports.json')
    if not os.path.exists(imports_file):
        return None
    try:
        with open(imports_file, 'r') as f:
            imports = json.load(f)
        return imports.get('B')
    except:
        return None

def generate_4c_input(interface_temp_values=None, interface_coords=None):
    """Generate 4C input YAML file."""
    
    # Generate nodes
    nodes = []
    node_coords = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L_A / nx
            y = j * H / ny
            nodes.append(f'NODE {nid} COORD {x:.10f} {y:.10f} 0.0')
            node_coords[(i, j)] = nid
            nid += 1
    
    n_nodes = nid - 1
    
    # Generate QUAD4 elements
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = node_coords[(i, j)]
            n2 = node_coords[(i+1, j)]
            n3 = node_coords[(i+1, j+1)]
            n4 = node_coords[(i, j+1)]
            elem_id = j * nx + i + 1
            elements.append(f'{elem_id} THERMO QUAD4 {n1} {n2} {n3} {n4} MAT 1')
    
    # Boundary definitions using DLINE-NODE TOPOLOGY
    dline_defs = []
    dline_id = 1
    
    # Left boundary (x=0) - Dirichlet u=0
    left_dline = dline_id
    for j in range(ny + 1):
        nid = node_coords[(0, j)]
        dline_defs.append(f'"NODE {nid} DLINE {left_dline}"')
    dline_id += 1
    
    # Bottom boundary (y=0) - Dirichlet u=0
    bottom_dline = dline_id
    for i in range(nx + 1):
        nid = node_coords[(i, 0)]
        dline_defs.append(f'"NODE {nid} DLINE {bottom_dline}"')
    dline_id += 1
    
    # Top boundary (y=H) - Dirichlet u=0
    top_dline = dline_id
    for i in range(nx + 1):
        nid = node_coords[(i, ny)]
        dline_defs.append(f'"NODE {nid} DLINE {top_dline}"')
    dline_id += 1
    
    # Right boundary (x=L_A) - Interface
    right_dline = dline_id
    for j in range(ny + 1):
        nid = node_coords[(nx, j)]
        dline_defs.append(f'"NODE {nid} DLINE {right_dline}"')
    dline_id += 1
    
    # Build YAML
    yaml_lines = [
        'PROBLEM TYPE:',
        '  PROBLEMTYPE: "Thermo"',
        '',
        'THERMAL DYNAMIC:',
        '  DYNAMICTYPE: Statics',
        '  TIMESTEP: 1.0',
        '  NUMSTEP: 1',
        '  MAXTIME: 1.0',
        '  LINEAR_SOLVER: 1',
        '',
        'SOLVER 1:',
        '  SOLVER: "UMFPACK"',
        '  NAME: "Thermal_Solver"',
        '',
        'MATERIALS:',
        f'  - MAT: 1',
        '    MAT_Fourier:',
        '      CAPA: 1.0',
        '      CONDUCT:',
        f'        constant: [{k_A}]',
        ''
    ]
    
    # Add source term as volume condition
    yaml_lines.extend([
        'DESIGN VOL THERMO NEUMANN CONDITIONS:',
        f'  - E: 1',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [1.0]',
        '    FUNCT: [1]',
        ''
    ])
    
    # Define the source function
    yaml_lines.extend([
        'FUNCT1:',
        f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_A_str}"',
        ''
    ])
    
    # Volume definition for source
    yaml_lines.extend([
        'DVOL-NODE TOPOLOGY:',
        f'  - "NODE {node_coords[(0,0)]} DVOL 1"',
        f'  - "NODE {node_coords[(nx,0)]} DVOL 1"',
        f'  - "NODE {node_coords[(nx,ny)]} DVOL 1"',
        f'  - "NODE {node_coords[(0,ny)]} DVOL 1"',
        ''
    ])
    
    # Dirichlet conditions for outer boundaries
    yaml_lines.extend([
        'DESIGN LINE DIRICH CONDITIONS:',
        f'  - E: {left_dline}',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
        f'  - E: {bottom_dline}',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
        f'  - E: {top_dline}',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
        ''
    ])
    
    # Interface condition - depends on whether we have imported values
    if interface_temp_values is not None and len(interface_temp_values) > 0:
        # Apply Dirichlet from imported values
        yaml_lines.extend([
            f'  - E: {right_dline}',
            '    NUMDOF: 1',
            '    ONOFF: [1]',
            '    VAL: [1.0]',
            '    FUNCT: [2]',
            ''
        ])
        
        # Create a polynomial fit for interface temperatures
        ys = np.array([c[1] for c in interface_coords])
        temps = np.array(interface_temp_values)
        
        # Fit a polynomial in y
        coeffs = np.polyfit(ys, temps, min(10, len(ys)-1))
        poly_str = ' '.join(f'{c:.10e}*y^{len(coeffs)-1-i}' for i, c in enumerate(coeffs))
        
        yaml_lines.extend([
            'FUNCT2:',
            f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{poly_str}"',
            ''
        ])
    else:
        # No interface values - use zero as default
        yaml_lines.extend([
            f'  - E: {right_dline}',
            '    NUMDOF: 1',
            '    ONOFF: [1]',
            '    VAL: [0.0]',
            '    FUNCT: [0]',
            ''
        ])
    
    # DLINE-NODE TOPOLOGY section
    yaml_lines.append('DLINE-NODE TOPOLOGY:')
    for d in dline_defs:
        yaml_lines.append(f'  - {d}')
    yaml_lines.append('')
    
    # NODE COORDS section
    yaml_lines.append('NODE COORDS:')
    for n in nodes:
        yaml_lines.append(f'  - "{n}"')
    yaml_lines.append('')
    
    # THERMO ELEMENTS section
    yaml_lines.append('THERMO ELEMENTS:')
    for e in elements:
        yaml_lines.append(f'  - "{e}"')
    
    return '\n'.join(yaml_lines), n_nodes

def run_4c(input_file, output_prefix):
    """Run 4C solver."""
    cmd = f'LD_LIBRARY_PATH=/opt/4C-dependencies/lib /home/alexander/4C/build/4C {input_file} {output_prefix}'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.returncode, result.stdout, result.stderr

def extract_results(output_prefix):
    """Extract solution and flux from 4C output VTU file."""
    # Find VTU files
    vtu_files = glob.glob(f'{output_prefix}-vtk-files/*.vtu')
    if not vtu_files:
        print("No VTU files found!")
        return None, None, None
    
    # Sort and get the last timestep
    vtu_files = sorted(vtu_files)
    vtu_file = vtu_files[-1]
    
    try:
        import pyvista as pv
        mesh = pv.read(vtu_file)
    except Exception as e:
        print(f"Error reading VTU: {e}")
        return None, None, None
    
    # Get temperature field
    temp_field = None
    for name in ['temperature', 'TEMPERATURE', 'temp']:
        if name in mesh.point_data:
            temp_field = mesh.point_data[name]
            break
    
    if temp_field is None:
        print("Temperature field not found in VTU!")
        return None, None, None
    
    pts = mesh.points
    temp = temp_field
    
    # Find interface nodes (x = L_A)
    tol = 1e-6
    interface_mask = np.abs(pts[:, 0] - L_A) < tol
    
    if not np.any(interface_mask):
        print("No interface nodes found!")
        return None, None, None
    
    interface_coords = pts[interface_mask]
    interface_temps = temp[interface_mask]
    
    # Remove duplicates by rounding y coordinates
    uy, inv = np.unique(np.round(interface_coords[:, 1], 10), return_inverse=True)
    
    # Average temperatures at duplicate locations
    unique_temps = np.zeros(len(uy))
    counts = np.zeros(len(uy))
    for i, idx in enumerate(inv):
        unique_temps[idx] += interface_temps[i]
        counts[idx] += 1
    unique_temps /= counts
    
    # Compute flux numerically (approximate gradient)
    hy = H / ny
    fluxes = np.zeros(len(unique_temps))
    
    for j in range(len(unique_temps)):
        y = uy[j]
        x_interior = L_A - L_A/nx
        mask_interior = (np.abs(pts[:, 0] - x_interior) < tol) & (np.abs(pts[:, 1] - y) < hy/2)
        if np.any(mask_interior):
            idx = np.argmax(mask_interior)
            du_dx = (unique_temps[j] - temp[idx]) / (L_A - x_interior)
            fluxes[j] = -k_A * du_dx
        else:
            fluxes[j] = 0.0
    
    coords_list = [[float(x_interface), float(y)] for y in uy]
    temps_list = [float(t) for t in unique_temps]
    fluxes_list = [float(q) for q in fluxes]
    
    return coords_list, temps_list, fluxes_list

def main():
    global_level = level
    global_work_dir = work_dir
    
    # Read imports if available
    imports = read_imports()
    interface_temp_values = None
    interface_coords = None
    
    if imports is not None and 'values' in imports:
        interface_coords = imports['coordinates']
        interface_temp_values = imports['values']
        print(f"Imported {len(interface_temp_values)} interface temperature values from B")
    
    # Generate and write input
    input_content, n_dof = generate_4c_input(interface_temp_values, interface_coords)
    input_file = os.path.join(work_dir, f'4c_A_level{level}.4C.yaml')
    with open(input_file, 'w') as f:
        f.write(input_content)
    
    # Run 4C
    output_prefix = os.path.join(work_dir, f'4c_A_level{level}')
    returncode, stdout, stderr = run_4c(input_file, output_prefix)
    
    # Print output for logging
    print(stdout)
    if stderr:
        print(stderr, file=sys.stderr)
    
    # Write log file
    log_file = os.path.join(work_dir, f'run_level{level}_A.log')
    with open(log_file, 'w') as f:
        f.write(stdout)
        if stderr:
            f.write(stderr)
        f.write(f"NDOF = {n_dof}\n")
    
    if returncode != 0:
        print(f"4C failed with return code {returncode}")
        sys.exit(returncode)
    
    # Extract results
    coords, temps, fluxes = extract_results(output_prefix)
    
    if coords is None:
        print("Failed to extract results")
        sys.exit(1)
    
    # Export data
    exports = {
        'field_name': 'temperature',
        'n_points': len(coords),
        'coordinates': coords,
        'values': temps,
        'normal_fluxes': fluxes
    }
    
    # Write exports
    exports_file = os.path.join(work_dir, 'exports.json')
    with open(exports_file, 'w') as f:
        json.dump({'A': exports}, f, indent=2)
    
    print(f"4C completed successfully. NDOF = {n_dof}, exported {len(coords)} interface points")

if __name__ == '__main__':
    main()
