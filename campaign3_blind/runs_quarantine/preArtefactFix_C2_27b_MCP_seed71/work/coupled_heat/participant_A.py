#!/usr/bin/env python3
"""
Participant A (4C) - Subdomain A: (0, 0.625) x (0, 1), k=1
Dirichlet side: receives interface field from B, imposes as essential BC at x=0.625
Returns: outward normal flux at interface (outward = +x direction)
"""
import json
import os
import sys
import subprocess
import numpy as np
import glob

# Read level from file
work_dir = os.getcwd()
level_file = os.path.join(work_dir, "level.json")
if os.path.exists(level_file):
    with open(level_file) as f:
        level_info = json.load(f)
    LEVEL = level_info.get("level", 1)
else:
    LEVEL = int(os.environ.get("LEVEL", 1))

# Mesh resolution: h = 1/8, 1/16, 1/32 for levels 1, 2, 3
h_values = {1: 1/8, 2: 1/16, 3: 1/32}
h = h_values[LEVEL]
nx_A = int(round(0.625 / h))
ny_A = int(round(1.0 / h))

print(f"=== Participant A (4C) ===")
print(f"Level: {LEVEL}, h: {h}")
print(f"Mesh: {nx_A} x {ny_A} elements")

# Subdomain A geometry
xA_min, xA_max = 0.0, 0.625
yA_min, yA_max = 0.0, 1.0
k_A = 1.0

# Source term in subdomain A (convert ** to ^ for 4C)
source_A = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

def generate_4c_input():
    """Generate 4C YAML input file for subdomain A"""
    
    nodes = []
    node_id = 1
    node_coords = {}
    
    for j in range(ny_A + 1):
        for i in range(nx_A + 1):
            x = xA_min + i * (xA_max - xA_min) / nx_A
            y = yA_min + j * (yA_max - yA_min) / ny_A
            node_coords[(i, j)] = node_id
            nodes.append(f'  - "NODE {node_id} COORD {x:.10f} {y:.10f} 0.0"')
            node_id += 1
    
    n_nodes = node_id - 1
    
    # Create quadrilateral elements
    elements = []
    for j in range(ny_A):
        for i in range(nx_A):
            n_bl = node_coords[(i, j)]
            n_br = node_coords[(i+1, j)]
            n_tr = node_coords[(i+1, j+1)]
            n_tl = node_coords[(i, j+1)]
            elem_id = len(elements) + 1
            elements.append(f'  - "{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
    
    n_elements = len(elements)
    
    # Define boundary lines and volume
    dline_nodes = []
    dvollist = []
    
    # Bottom edge (DLINE 1)
    for i in range(nx_A + 1):
        nid = node_coords[(i, 0)]
        dline_nodes.append(f'  - "NODE {nid} DLINE 1"')
    # Top edge (DLINE 2)
    for i in range(nx_A + 1):
        nid = node_coords[(i, ny_A)]
        dline_nodes.append(f'  - "NODE {nid} DLINE 2"')
    # Left edge (DLINE 3)
    for j in range(ny_A + 1):
        nid = node_coords[(0, j)]
        dline_nodes.append(f'  - "NODE {nid} DLINE 3"')
    # Right edge/interface (DLINE 4) - exclude corners
    for j in range(1, ny_A):
        nid = node_coords[(nx_A, j)]
        dline_nodes.append(f'  - "NODE {nid} DLINE 4"')
    
    # Volume for source term - all nodes
    for nid in range(1, n_nodes + 1):
        dvollist.append(f'  - "NODE {nid} DVOL 1"')
    
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
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_A}"
DESIGN VOL TRANSPORT NEUMANN CONDITIONS:
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
DVOL-NODE TOPOLOGY:
{chr(10).join(dvollist)}
DLINE-NODE TOPOLOGY:
{chr(10).join(dline_nodes)}
NODE COORDS:
{chr(10).join(nodes)}
TRANSPORT ELEMENTS:
{chr(10).join(elements)}
RESULT DESCRIPTION:
  - SCATRA:
      DIS: "scatra"
      NODE: 2
      QUANTITY: "phi"
      VALUE: 0.0
      TOLERANCE: 1.0e30
'''
    
    return yaml_content, n_nodes, node_coords

# Read imports from partner B
imports_path = os.path.join(work_dir, "imports.json")
if os.path.exists(imports_path):
    with open(imports_path) as f:
        imports = json.load(f)
    has_imports = True
else:
    imports = {}
    has_imports = False

# Get interface data from partner B
if has_imports and "B" in imports:
    interface_data = imports["B"]
    interface_y = np.array([p[1] for p in interface_data["coordinates"]])
    interface_u_B = np.array(interface_data["values"])
    print(f"Received interface data from B: {len(interface_u_B)} points")
else:
    interface_y = np.linspace(0, 1, ny_A + 1)[1:-1]
    interface_u_B = np.zeros_like(interface_y)
    print("No imports available, using zero initial guess")

yaml_content, n_nodes, node_coords = generate_4c_input()

input_file = os.path.join(work_dir, "subdomain_A.4C.yaml")
with open(input_file, 'w') as f:
    f.write(yaml_content)

output_prefix = os.path.join(work_dir, "result_A")
fourc_binary = "/home/alexander/4C/build/4C"
cmd = [fourc_binary, input_file, output_prefix]
env = os.environ.copy()
env['LD_LIBRARY_PATH'] = '/opt/4C-dependencies/lib'

print(f"Running 4C...")
result = subprocess.run(cmd, capture_output=True, text=True, env=env)
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)
print(f"Exit code: {result.returncode}")

if result.returncode != 0:
    print("ERROR: 4C failed!")
    sys.exit(1)

# Read results
import pyvista as pv
vtu_files = glob.glob(os.path.join(work_dir, "result_A-vtk-files", "scatra-*.vtu"))
if not vtu_files:
    print("ERROR: No VTU files found!")
    sys.exit(1)
vtu_file = sorted(vtu_files)[-1]
mesh = pv.read(vtu_file)
u_values = mesh.point_data['phi_1']
coords = mesh.points

# Extract interface nodes
interface_x = 0.625
tol = 1e-6
interface_mask = np.abs(coords[:, 0] - interface_x) < tol
interface_mask = interface_mask & (coords[:, 1] > tol) & (coords[:, 1] < 1.0 - tol)

interface_coords_A = coords[interface_mask]
interface_u_A = u_values[interface_mask]
sort_idx = np.argsort(interface_coords_A[:, 1])
interface_coords_A = interface_coords_A[sort_idx]
interface_u_A = interface_u_A[sort_idx]

# Compute flux at interface using adjacent element gradients
interface_fluxes = []
for iy, (ix, iy_coord) in enumerate(zip(interface_coords_A[:, 0], interface_coords_A[:, 1])):
    same_y_mask = np.abs(coords[:, 1] - iy_coord) < tol
    row_coords = coords[same_y_mask]
    row_vals = u_values[same_y_mask]
    
    left_of_interface = row_coords[row_coords[:, 0] < interface_x - tol]
    if len(left_of_interface) > 0:
        x_left = np.max(left_of_interface[:, 0])
        idx_left = np.argmin(np.abs(row_coords[:, 0] - x_left))
        u_left = row_vals[idx_left]
        dx = interface_x - x_left
        du_dx = (interface_u_A[iy] - u_left) / dx
    else:
        du_dx = 0.0
    
    qn = -k_A * du_dx
    interface_fluxes.append(qn)

interface_fluxes = np.array(interface_fluxes)
interface_points_out = interface_coords_A[:, :2].copy()

print(f"Interface fluxes: min={interface_fluxes.min():.6f}, max={interface_fluxes.max():.6f}")

exports = {
    "field_name": "temperature",
    "coordinates": interface_points_out.tolist(),
    "values": interface_u_A.tolist(),
    "normal_fluxes": interface_fluxes.tolist(),
    "n_points": len(interface_u_A)
}

exports_path = os.path.join(work_dir, "exports.json")
with open(exports_path, 'w') as f:
    json.dump(exports, f, indent=2)

print(f"Wrote exports.json with {len(interface_u_A)} points")
mesh.save(os.path.join(work_dir, f"solution_A_level{LEVEL}.vtu"))
print(f"NDOF = {n_nodes}")
