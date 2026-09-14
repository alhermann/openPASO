#!/usr/bin/env python3
"""
Participant A (4C) - Subdomain A: (0, 0.625) x (0, 1), k=1
DIRICHLET side: receives field from B, imposes as Dirichlet BC on interface, returns flux
Interface at x = 0.625, outward normal points +x (right)
"""
import json
import os
import sys
import subprocess
import numpy as np
import glob

# Problem parameters
L_A = 0.625  # width of subdomain A
H = 1.0      # height
k_A = 1.0    # thermal conductivity in A
interface_x = 0.625

def get_mesh_level():
    """Determine mesh level from work_dir name"""
    work_dir = os.getcwd()
    if 'level1' in work_dir:
        return 1
    elif 'level2' in work_dir:
        return 2
    elif 'level3' in work_dir:
        return 3
    else:
        raise ValueError(f"Cannot determine mesh level from {work_dir}")

def generate_4c_yaml(nx, ny, interface_values=None):
    """Generate 4C YAML input file for subdomain A"""
    
    # Generate uniform mesh
    nodes = []
    node_coords = {}
    nid = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * L_A / nx
            y = j * H / ny
            nodes.append(f'"NODE {nid} COORD {x:.10f} {y:.10f} 0.0"')
            node_coords[(i, j)] = nid
            nid += 1
    
    n_nodes = nid - 1
    
    # Generate quadrilateral elements (QUAD4)
    elements = []
    for j in range(ny):
        for i in range(nx):
            n_bl = node_coords[(i, j)]
            n_br = node_coords[(i+1, j)]
            n_tr = node_coords[(i+1, j+1)]
            n_tl = node_coords[(i, j+1)]
            elements.append(f'"{len(elements)+1} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
    
    n_elements = len(elements)
    
    # Define boundary lines for Dirichlet BCs
    dline_node = []
    
    # Left boundary (x=0) - line 1
    for j in range(ny + 1):
        nid = node_coords[(0, j)]
        dline_node.append(f'"NODE {nid} DLINE 1"')
    
    # Bottom boundary (y=0) - line 2  
    for i in range(nx + 1):
        nid = node_coords[(i, 0)]
        dline_node.append(f'"NODE {nid} DLINE 2"')
    
    # Top boundary (y=1) - line 3
    for i in range(nx + 1):
        nid = node_coords[(i, ny)]
        dline_node.append(f'"NODE {nid} DLINE 3"')
    
    # Interface (x=0.625) - line 4 - exclude corners since they're on outer boundary
    for j in range(1, ny):
        nid = node_coords[(nx, j)]
        dline_node.append(f'"NODE {nid} DLINE 4"')
    
    # Create volume for source term
    dvol_node = [f'"NODE {nid} DVOL 1"' for nid in range(1, n_nodes + 1)]
    
    # Source term expression for subdomain A
    source_expr = "-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400"
    
    # Build all Dirichlet conditions first
    dirichlet_conditions = []
    
    # Left boundary (E=1)
    dirichlet_conditions.extend([
        '  - E: 1',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
    ])
    
    # Bottom boundary (E=2)
    dirichlet_conditions.extend([
        '  - E: 2',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
    ])
    
    # Top boundary (E=3)
    dirichlet_conditions.extend([
        '  - E: 3',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [0.0]',
        '    FUNCT: [0]',
    ])
    
    # Interface boundary (E=4)
    if interface_values is not None and len(interface_values) > 0:
        # Map imported values to our interface nodes
        interp_values = []
        for j in range(1, ny):
            y_node = j * H / ny
            val = 0.0
            min_y = min(iv[0][1] for iv in interface_values)
            max_y = max(iv[0][1] for iv in interface_values)
            if min_y <= y_node <= max_y:
                for k in range(len(interface_values)-1):
                    y1 = interface_values[k][0][1]
                    y2 = interface_values[k+1][0][1]
                    v1 = interface_values[k][1]
                    v2 = interface_values[k+1][1]
                    if y1 <= y_node <= y2:
                        t = (y_node - y1) / (y2 - y1) if y2 != y1 else 0
                        val = v1 + t * (v2 - v1)
                        break
                else:
                    min_dist = float('inf')
                    for iv, v in zip(interface_values, [iv[1] for iv in interface_values]):
                        dist = abs(iv[0][1] - y_node)
                        if dist < min_dist:
                            min_dist = dist
                            val = v
            else:
                val = 0.0
            interp_values.append(val)
        
        dirichlet_conditions.extend([
            '  - E: 4',
            '    NUMDOF: 1',
            '    ONOFF: [1]',
            '    VAL: [' + ', '.join([f'{v:.10e}' for v in interp_values]) + ']',
            '    FUNCT: [0]',
        ])
    else:
        # First iteration: use zero initial guess
        dirichlet_conditions.extend([
            '  - E: 4',
            '    NUMDOF: 1',
            '    ONOFF: [1]',
            '    VAL: [' + ', '.join(['0.0'] * (ny - 1)) + ']',
            '    FUNCT: [0]',
        ])
    
    # Build YAML
    yaml_lines = [
        'TITLE:',
        '  - "Subdomain A - Heat conduction with 4C"',
        'PROBLEM SIZE:',
        '  DIM: 2',
        'PROBLEM TYPE:',
        '  PROBLEMTYPE: "Scalar_Transport"',
        'SCALAR TRANSPORT DYNAMIC:',
        '  TIMEINTEGR: "Stationary"',
        '  SOLVERTYPE: "linear_full"',
        '  VELOCITYFIELD: "zero"',
        '  TIMESTEP: 1.0',
        '  NUMSTEP: 1',
        '  MAXTIME: 1.0',
        '  LINEAR_SOLVER: 1',
        'SOLVER 1:',
        '  SOLVER: "UMFPACK"',
        '  NAME: "direct"',
        'MATERIALS:',
        f'  - MAT: 1',
        '    MAT_scatra:',
        f'      DIFFUSIVITY: {k_A:.10f}',
        'DESIGN LINE DIRICH CONDITIONS:',
    ] + dirichlet_conditions + [
        'DESIGN VOL TRANSPORT NEUMANN CONDITIONS:',
        '  - E: 1',
        '    NUMDOF: 1',
        '    ONOFF: [1]',
        '    VAL: [1.0]',
        '    FUNCT: [1]',
        'FUNCT1:',
        f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_expr}"',
        'DLINE-NODE TOPOLOGY:',
    ] + dline_node + [
        'DVOL-NODE TOPOLOGY:',
    ] + dvol_node + [
        'NODE COORDS:',
    ] + nodes + [
        'TRANSPORT ELEMENTS:',
    ] + elements
    
    yaml_content = '\n'.join(yaml_lines)
    
    return yaml_content, node_coords, n_nodes, n_elements

def main():
    work_dir = os.getcwd()
    print(f"Working directory: {work_dir}")
    sys.stdout.flush()
    
    level = get_mesh_level()
    divisions = [8, 16, 32][level - 1]
    nx = int(0.625 * divisions)
    ny = int(1.0 * divisions)
    
    print(f"Level {level}: nx={nx}, ny={ny}")
    sys.stdout.flush()
    
    # Read imports
    imports_path = os.path.join(work_dir, 'imports.json')
    interface_values = None
    if os.path.exists(imports_path):
        with open(imports_path, 'r') as f:
            imports_data = json.load(f)
        if 'B' in imports_data:
            partner_data = imports_data['B']
            interface_values = list(zip(partner_data['coordinates'], partner_data['values']))
        print(f"Read imports from {imports_path}")
    else:
        print("No imports found - first iteration")
    sys.stdout.flush()
    
    # Generate and write YAML
    yaml_content, node_coords, n_nodes, n_elements = generate_4c_yaml(nx, ny, interface_values)
    yaml_file = os.path.join(work_dir, 'subdomain_A.4C.yaml')
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    # Run 4C - need to unset DISPLAY and use mpirun
    output_prefix = os.path.join(work_dir, 'result_A')
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'
    env.pop('DISPLAY', None)  # Remove DISPLAY to avoid X11 issues
    
    cmd = ['mpirun', '-np', '1', '/home/alexander/4C/build/4C', yaml_file, output_prefix]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    sys.stdout.flush()
    
    if result.returncode != 0:
        print(f"4C failed with return code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)
    
    # Extract solution from VTU
    vtu_files = glob.glob(os.path.join(work_dir, 'result_A-vtk-files', 'scatra-*.vtu'))
    if not vtu_files:
        print("No VTU files found!", file=sys.stderr)
        sys.exit(1)
    
    vtu_file = sorted(vtu_files)[-1]
    print(f"Reading solution from {vtu_file}")
    sys.stdout.flush()
    
    import pyvista as pv
    mesh = pv.read(vtu_file)
    phi = mesh.point_data['phi_1']
    pts = mesh.points
    
    # Export interface data
    interface_coords = []
    interface_values_out = []
    interface_fluxes = []
    
    tol = 1e-6
    dx = L_A / nx
    
    for j in range(1, ny):
        y_j = j * H / ny
        x_interface = L_A
        
        # Find node at interface
        idx_i = None
        idx_l = None
        for idx, pt in enumerate(pts):
            if abs(pt[0] - x_interface) < tol and abs(pt[1] - y_j) < tol:
                idx_i = idx
            if abs(pt[0] - (x_interface - dx)) < tol and abs(pt[1] - y_j) < tol:
                idx_l = idx
        
        if idx_i is not None:
            interface_coords.append([pts[idx_i][0], pts[idx_i][1]])
            interface_values_out.append(phi[idx_i])
            
            if idx_l is not None:
                du_dx = (phi[idx_i] - phi[idx_l]) / dx
                qn = -k_A * du_dx  # outward normal is +x
                interface_fluxes.append(qn)
            else:
                interface_fluxes.append(0.0)
    
    # Write exports
    exports = {
        'field_name': 'temperature',
        'coordinates': interface_coords,
        'values': interface_values_out,
        'normal_fluxes': interface_fluxes,
        'n_points': len(interface_coords)
    }
    
    exports_path = os.path.join(work_dir, 'exports.json')
    with open(exports_path, 'w') as f:
        json.dump(exports, f, indent=2)
    
    print(f"Wrote exports to {exports_path}")
    print(f"NDOF = {n_nodes}")
    sys.stdout.flush()
    
    # Save solution VTU
    import shutil
    shutil.copy(vtu_file, os.path.join(work_dir, 'solution_A.vtu'))

if __name__ == '__main__':
    main()
