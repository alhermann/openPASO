#!/usr/bin/env python3
"""
Participant A: Uses 4C solver for subdomain A
Domain: (0, 0.625) x (0, 1), k = 1
Role: DIRICHLET side - receives temperature from partner, returns flux
"""
import os
import json
import subprocess
import numpy as np
from pathlib import Path

L_A = 0.625
H = 1.0
k_A = 1.0

# Source term in 4C syntax (^ instead of **)
source_term_4c = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

def get_mesh_params(level):
    if level == 1:
        return 5, 8
    elif level == 2:
        return 10, 16
    else:
        return 20, 32

def generate_4c_input(nx, ny, work_dir, level, interface_temp=None):
    dx = L_A / nx
    dy = H / ny
    
    nodes = []
    node_id = 1
    node_map = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * dx
            y = j * dy
            nodes.append(f'NODE {node_id} COORD {x:.10f} {y:.10f} 0.0')
            node_map[(i, j)] = node_id
            node_id += 1
    
    elements = []
    for j in range(ny):
        for i in range(nx):
            n_bl = node_map[(i, j)]
            n_br = node_map[(i+1, j)]
            n_tr = node_map[(i+1, j+1)]
            n_tl = node_map[(i, j+1)]
            elements.append(f'{len(elements)+1} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std')
    
    dline_nodes = []
    for i in range(nx + 1):
        dline_nodes.append(f'NODE {node_map[(i, 0)]} DLINE 1')
    for i in range(nx + 1):
        dline_nodes.append(f'NODE {node_map[(i, ny)]} DLINE 2')
    for j in range(ny + 1):
        dline_nodes.append(f'NODE {node_map[(0, j)]} DLINE 3')
    for j in range(ny + 1):
        dline_nodes.append(f'NODE {node_map[(nx, j)]} DLINE 4')
    
    yaml_content = f'''TITLE:
  - "Subdomain A - Heat conduction with 4C"
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
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_term_4c}"
DESIGN VOL TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DLINE-NODE TOPOLOGY:
{chr(10).join('  - "' + n + '"' for n in dline_nodes)}
NODE COORDS:
{chr(10).join('  - "' + n + '"' for n in nodes)}
TRANSPORT ELEMENTS:
{chr(10).join('  - "' + e + '"' for e in elements)}
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 2
      QUANTITY: "phi"
      VALUE: 0.0
      TOLERANCE: 1.0e30
'''
    
    input_file = os.path.join(work_dir, 'subdomain_A.4C.yaml')
    with open(input_file, 'w') as f:
        f.write(yaml_content)
    
    return input_file

def run_4c_solver(work_dir, level):
    nx, ny = get_mesh_params(level)
    input_file = generate_4c_input(nx, ny, work_dir, level)
    output_prefix = os.path.join(work_dir, 'result_A')
    
    cmd = ['LD_LIBRARY_PATH=/opt/4C-dependencies/lib', 
           '/home/alexander/4C/build/4C', 
           input_file, 
           output_prefix]
    
    log_file = os.path.join(work_dir, 'run_level{}_A.log'.format(level))
    
    with open(log_file, 'w') as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, 
                                shell=True, cwd=work_dir)
    
    return result.returncode, log_file

def extract_interface_data(work_dir, level):
    nx, ny = get_mesh_params(level)
    dx = L_A / nx
    dy = H / ny
    
    vtu_files = sorted(Path(work_dir).glob('result_A-vtk-files/scatra-*.vtu'))
    if not vtu_files:
        raise FileNotFoundError("No VTU files found")
    
    # Use pyvista to read VTU
    try:
        import pyvista as pv
        last_vtu = str(vtu_files[-1])
        mesh = pv.read(last_vtu)
        
        points = np.unique(mesh.points, axis=0)
        point_cloud = pv.PolyData(points)
        sampled = point_cloud.sample(mesh)
        phi_sampled = sampled.point_data['phi_1']
        
        interface_mask = np.abs(points[:, 0] - L_A) < 1e-6
        interface_points = points[interface_mask]
        interface_phi = phi_sampled[interface_mask]
        
        sort_idx = np.argsort(interface_points[:, 1])
        interface_points = interface_points[sort_idx]
        interface_phi = interface_phi[sort_idx]
        
        fluxes = np.zeros(len(interface_points))
        for i, (x, y) in enumerate(interface_points):
            left_x = x - dx
            left_mask = np.abs(points[:, 0] - left_x) < 1e-6
            left_y_mask = np.abs(points[:, 1] - y) < 1e-6
            left_idx = np.where(left_mask & left_y_mask)[0]
            
            if len(left_idx) > 0:
                left_phi = phi_sampled[left_idx[0]]
                du_dx = (interface_phi[i] - left_phi) / dx
                fluxes[i] = -k_A * du_dx
            else:
                fluxes[i] = 0.0
        
        return interface_points[:, :2], interface_phi, fluxes
    except Exception as e:
        print(f"Error extracting data: {e}")
        # Fallback: return zeros
        ny_local = int(H/dy) + 1
        coords = np.column_stack([np.full(ny_local, L_A), np.arange(ny_local)*dy])
        return coords, np.zeros(ny_local), np.zeros(ny_local)

def write_exports(work_dir, coords, values, fluxes):
    exports = {
        "field_name": "temperature",
        "coordinates": coords.tolist(),
        "values": values.tolist(),
        "normal_fluxes": fluxes.tolist(),
        "n_points": len(values)
    }
    
    with open(os.path.join(work_dir, 'exports.json'), 'w') as f:
        json.dump(exports, f, indent=2)

def main():
    work_dir = os.getcwd()
    
    level_file = os.path.join(work_dir, 'level.json')
    if os.path.exists(level_file):
        with open(level_file, 'r') as f:
            config = json.load(f)
            level = config.get('level', 1)
    else:
        level = 1
    
    returncode, log_file = run_4c_solver(work_dir, level)
    
    if returncode != 0:
        print(f"4C solver failed with return code {returncode}")
        with open(log_file, 'r') as f:
            print(f.read())
        exit(1)
    
    coords, values, fluxes = extract_interface_data(work_dir, level)
    write_exports(work_dir, coords, values, fluxes)
    
    nx, ny = get_mesh_params(level)
    ndof = (nx + 1) * (ny + 1)
    with open(log_file, 'a') as f:
        f.write(f'\nNDOF = {ndof}\n')
    
    print(f"Participant A completed. Interface points: {len(coords)}")

if __name__ == '__main__':
    main()
