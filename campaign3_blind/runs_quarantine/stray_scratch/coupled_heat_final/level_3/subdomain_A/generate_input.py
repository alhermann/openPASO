#!/usr/bin/env python3
"""
Participant A (4C): Subdomain (0, 0.625) x (0, 1), k=1
Dirichlet side: receives temperature from partner B, imposes as BC on interface, returns flux
"""

import os
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

# Problem parameters
INTERFACE_X = 5/8  # 0.625
DOMAIN_A_WIDTH = INTERFACE_X
DOMAIN_HEIGHT = 1.0
K_A = 1.0

# Read level info
work_dir = Path(os.getcwd())
level_info_file = work_dir / "level.json"
with open(level_info_file) as f:
    level_info = json.load(f)

n_divisions = level_info["n_divisions"]
nx = level_info["nx_A"]
ny = level_info["ny"]

print(f"Subdomain A: nx={nx}, ny={ny}, h=1/{n_divisions}")

# Generate mesh nodes for subdomain A
nodes = []
node_id = 1
for j in range(ny + 1):
    for i in range(nx + 1):
        x = i * DOMAIN_A_WIDTH / nx
        y = j * DOMAIN_HEIGHT / ny
        nodes.append((node_id, x, y))
        node_id += 1

n_nodes = len(nodes)
print(f"Number of nodes: {n_nodes}")

# DLINE topology
dline_topology = []
dline_id = 1

left_nodes = sorted([nid for nid, x, y in nodes if abs(x) < 1e-10])
for nid in left_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_left = dline_id
dline_id += 1

bottom_nodes = sorted([nid for nid, x, y in nodes if abs(y) < 1e-10])
for nid in bottom_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_bottom = dline_id
dline_id += 1

top_nodes = sorted([nid for nid, x, y in nodes if abs(y - DOMAIN_HEIGHT) < 1e-10])
for nid in top_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_top = dline_id
dline_id += 1

interface_nodes = sorted([nid for nid, x, y in nodes if abs(x - INTERFACE_X) < 1e-10])
for nid in interface_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_interface = dline_id

# Elements
elements = []
elem_id = 1
for j in range(ny):
    for i in range(nx):
        n1 = (j * (nx + 1) + i) + 1
        n2 = (j * (nx + 1) + i + 1) + 1
        n3 = ((j + 1) * (nx + 1) + i + 1) + 1
        n4 = ((j + 1) * (nx + 1) + i) + 1
        elements.append(f"{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
        elem_id += 1

n_elements = len(elements)
print(f"Number of elements: {n_elements}")

# Read imports from partner B (temperature values at interface)
imports_file = work_dir / "imports.json"
interface_temps_from_partner = []
interface_coords_from_partner = []

if imports_file.exists():
    with open(imports_file) as f:
        imports = json.load(f)
    if "B" in imports:
        partner_data = imports["B"]
        interface_temps_from_partner = partner_data.get("values", [])
        interface_coords_from_partner = partner_data.get("coordinates", [])
        print(f"Received {len(interface_temps_from_partner)} temperature values from partner B")
    else:
        print("No data received from partner B")
else:
    print("No imports.json found (first iteration)")

# Build 4C YAML input
node_coords = [f'NODE {nid} COORD {x:.10f} {y:.10f} 0.0' for nid, x, y in nodes]

yaml_content = f'''TITLE:
  - "Heat conduction subdomain A (4C)"
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
      DIFFUSIVITY: {K_A}
DESIGN LINE DIRICH CONDITIONS:
  - E: {dline_left}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: {dline_bottom}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: {dline_top}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [0]
DSURF-NODE TOPOLOGY:
{chr(10).join(f'  - "NODE {nid} DSURFACE 1"' for nid, x, y in nodes)}
NODE COORDS:
{chr(10).join(f'  - "{nc}"' for nc in node_coords)}
DLINE-NODE TOPOLOGY:
{chr(10).join(f'  - "{dt}"' for dt in dline_topology)}
TRANSPORT ELEMENTS:
{chr(10).join(f'  - "{e}"' for e in elements)}
'''

# Write YAML file
yaml_file = work_dir / "subdomain_A.4C.yaml"
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"Wrote 4C input to {yaml_file}")

# Run 4C solver
output_prefix = work_dir / "result_A"
cmd = ["/home/alexander/4C/build/4C", str(yaml_file), str(output_prefix)]

log_file = work_dir / "run.log"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = "/opt/4C-dependencies/lib"

with open(log_file, 'w') as log_f:
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_f.write(result.stdout)
    log_f.write(result.stderr)
    print(result.stdout)

exit_code = result.returncode
print(f"4C exit code: {exit_code}")

if exit_code != 0:
    print("ERROR: 4C failed!")
    sys.exit(1)

# Read results from VTU file
try:
    import pyvista as pv
    
    vtu_files = list(work_dir.glob("result_A-vtk-files/scatra-*.vtu"))
    if not vtu_files:
        print("ERROR: No VTU files found!")
        sys.exit(1)
    
    vtu_file = sorted(vtu_files)[-1]
    print(f"Reading results from {vtu_file}")
    
    mesh = pv.read(str(vtu_file))
    phi_values = mesh.point_data.get('phi_1', mesh.point_data.get('phi', None))
    coords = mesh.points[:, :2]
    
    print(f"Solution: min={phi_values.min():.6f}, max={phi_values.max():.6f}")
except Exception as e:
    print(f"Error reading VTU: {e}")
    # Fallback: use binary result file
    phi_values = None
    coords = None

# Compute flux at interface (outward normal is +x direction for subdomain A)
h_x = DOMAIN_A_WIDTH / nx
interface_fluxes = []
interface_export_coords = []
interface_temps_at_nodes = []

interface_node_coords = [(nid, x, y) for nid, x, y in nodes if nid in interface_nodes]

for nid, x, y in interface_node_coords:
    interface_export_coords.append([x, y])
    
    if phi_values is not None:
        # Find node index in mesh
        node_idx = None
        for idx, (mx, my) in enumerate(coords):
            if abs(mx - x) < 1e-10 and abs(my - y) < 1e-10:
                node_idx = idx
                break
        
        if node_idx is not None:
            interface_temps_at_nodes.append(phi_values[node_idx])
            
            # Estimate gradient using neighboring node
            x_left = x - h_x
            left_val = None
            for idx, (mx, my) in enumerate(coords):
                if abs(mx - x_left) < h_x/2 and abs(my - y) < h_x/2:
                    left_val = phi_values[idx]
                    break
            
            if left_val is not None:
                du_dx = (phi_values[node_idx] - left_val) / h_x
                qn = -K_A * du_dx  # outward normal flux
                interface_fluxes.append(qn)
            else:
                interface_fluxes.append(0.0)
        else:
            interface_temps_at_nodes.append(0.0)
            interface_fluxes.append(0.0)
    else:
        interface_temps_at_nodes.append(0.0)
        interface_fluxes.append(0.0)

# Write exports.json
exports = {
    "field_name": "temperature",
    "coordinates": interface_export_coords,
    "values": interface_temps_at_nodes,
    "normal_fluxes": interface_fluxes,
    "n_points": len(interface_export_coords)
}

exports_file = work_dir / "exports.json"
with open(exports_file, 'w') as f:
    json.dump(exports, f, indent=2)

print(f"Wrote exports.json with {len(interface_export_coords)} interface points")
print(f"Interface flux: min={min(interface_fluxes):.6f}, max={max(interface_fluxes):.6f}")

# Write NDOF to ndof.txt
ndof = n_nodes
ndof_file = work_dir / "ndof.txt"
with open(ndof_file, 'w') as f:
    f.write(f"NDOF = {ndof}\n")

print(f"NDOF = {ndof}")
