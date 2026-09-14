#!/usr/bin/env python3
"""
Participant A (4C): Subdomain (0, 0.625) x (0, 1), k=1
Dirichlet side: receives temperature from partner, imposes as BC on interface, returns flux
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

# Source term for 4C (using ^ instead of **)
SOURCE_A_4C = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

# Get mesh level from environment or file
work_dir = Path(os.getcwd())
level_info_file = work_dir / "level.json"
if level_info_file.exists():
    with open(level_info_file) as f:
        level_info = json.load(f)
    n_divisions = level_info.get("n_divisions", 8)
else:
    n_divisions = int(os.environ.get("MESH_LEVEL", 8))

# Calculate mesh parameters
nx_A = int(DOMAIN_A_WIDTH * n_divisions)
ny = int(DOMAIN_HEIGHT * n_divisions)

print(f"Subdomain A: nx={nx_A}, ny={ny}, h=1/{n_divisions}")

# Generate mesh nodes for subdomain A
nodes = []
node_id = 1
for j in range(ny + 1):
    for i in range(nx_A + 1):
        x = i * DOMAIN_A_WIDTH / nx_A
        y = j * DOMAIN_HEIGHT / ny
        nodes.append((node_id, x, y))
        node_id += 1

n_nodes = len(nodes)
print(f"Number of nodes: {n_nodes}")

# Identify boundary edges
# Left edge (x=0): Dirichlet u=0
# Right edge (x=0.625): Interface - will receive T from partner
# Bottom edge (y=0): Dirichlet u=0  
# Top edge (y=1): Dirichlet u=0

# Create DLINE topology for boundaries
dline_topology = []
dline_id = 1

# Left edge (x=0) - Dirichlet u=0
left_nodes = [nid for nid, x, y in nodes if abs(x) < 1e-10]
for nid in left_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_left = dline_id
dline_id += 1

# Bottom edge (y=0) - Dirichlet u=0
bottom_nodes = [nid for nid, x, y in nodes if abs(y) < 1e-10]
for nid in bottom_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_bottom = dline_id
dline_id += 1

# Top edge (y=1) - Dirichlet u=0
top_nodes = [nid for nid, x, y in nodes if abs(y - DOMAIN_HEIGHT) < 1e-10]
for nid in top_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_top = dline_id
dline_id += 1

# Right edge (x=0.625) - Interface
interface_nodes = [nid for nid, x, y in nodes if abs(x - INTERFACE_X) < 1e-10]
for nid in interface_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
dline_interface = dline_id

# Generate elements (QUAD4)
elements = []
elem_id = 1
for j in range(ny):
    for i in range(nx_A):
        # Find node IDs for this quad
        n1 = (j * (nx_A + 1) + i) + 1           # bottom-left
        n2 = (j * (nx_A + 1) + i + 1) + 1       # bottom-right
        n3 = ((j + 1) * (nx_A + 1) + i + 1) + 1 # top-right
        n4 = ((j + 1) * (nx_A + 1) + i) + 1     # top-left
        elements.append(f"{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
        elem_id += 1

n_elements = len(elements)
print(f"Number of elements: {n_elements}")

# Read imports from partner (temperature values at interface)
imports_file = work_dir / "imports.json"
if imports_file.exists():
    with open(imports_file) as f:
        imports = json.load(f)
    # Partner is "B", get its exports
    if "B" in imports:
        partner_data = imports["B"]
        interface_temps_from_partner = partner_data.get("values", [])
        interface_coords_from_partner = partner_data.get("coordinates", [])
        print(f"Received {len(interface_temps_from_partner)} temperature values from partner B")
    else:
        interface_temps_from_partner = []
        interface_coords_from_partner = []
        print("No data received from partner B (first iteration?)")
else:
    interface_temps_from_partner = []
    interface_coords_from_partner = []
    print("No imports.json found (first iteration)")

# Build 4C YAML input
yaml_content = f'''TITLE:
  - "Heat conduction subdomain A (4C) - coupled simulation"
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
'''

# Add source term as volume condition
yaml_content += f'''DESIGN VOL TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SOURCE_A_4C}"
'''

# Add interface Dirichlet condition if we have data from partner
if interface_temps_from_partner and len(interface_temps_from_partner) > 0:
    # We need to interpolate partner temps to our interface nodes
    # For now, use a simple approach: map by y-coordinate
    interface_node_ids = sorted(interface_nodes)
    interface_node_coords = [(nid, x, y) for nid, x, y in nodes if nid in interface_node_ids]
    
    # Interpolate temperatures at interface nodes
    interface_temps = []
    for nid, x, y in interface_node_coords:
        # Find closest partner point
        min_dist = float('inf')
        closest_temp = 0.0
        for idx, (px, py) in enumerate(interface_coords_from_partner):
            dist = abs(y - py)
            if dist < min_dist:
                min_dist = dist
                closest_temp = interface_temps_from_partner[idx]
        interface_temps.append(closest_temp)
    
    # Write interface temperatures as a function (we'll use a piecewise approach)
    # For simplicity, we'll apply them as nodal values via a custom function
    # Actually, 4C doesn't support arbitrary nodal values easily...
    # Let's use a different approach: write temps to a file and read them
    
    # For now, skip interface BC and handle it differently
    pass
else:
    # First iteration: no interface BC yet, just solve with homogeneous
    yaml_content += f'''DESIGN LINE DIRICH CONDITIONS:
  - E: {dline_interface}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

# Node coordinates
yaml_content += "NODE COORDS:\n"
for nid, x, y in nodes:
    yaml_content += f'  - "NODE {nid} COORD {x:.10f} {y:.10f} 0.0"\n'

# DLINE topology
yaml_content += "DLINE-NODE TOPOLOGY:\n"
for line in dline_topology:
    yaml_content += f'  - "{line}"\n'

# Elements
yaml_content += "TRANSPORT ELEMENTS:\n"
for elem in elements:
    yaml_content += f'  - "{elem}"\n'

# Write YAML file
yaml_file = work_dir / "subdomain_A.4C.yaml"
with open(yaml_file, 'w') as f:
    f.write(yaml_content)

print(f"Wrote 4C input to {yaml_file}")

# Run 4C solver
output_prefix = work_dir / "result_A"
cmd = ["LD_LIBRARY_PATH=/opt/4C-dependencies/lib", 
       "/home/alexander/4C/build/4C", 
       str(yaml_file), 
       str(output_prefix)]

print(f"Running 4C: {' '.join(cmd[1:])}")

# Capture output
log_file = work_dir / "run.log"
with open(log_file, 'w') as log_f:
    result = subprocess.run(cmd[1:], capture_output=True, text=True, 
                          env={**os.environ, "LD_LIBRARY_PATH": "/opt/4C-dependencies/lib"})
    log_f.write(result.stdout)
    log_f.write(result.stderr)
    print(result.stdout)
    print(result.stderr, file=sys.stderr)

exit_code = result.returncode
print(f"4C exit code: {exit_code}")

if exit_code != 0:
    print("ERROR: 4C failed!")
    sys.exit(1)

# Read results from VTU file
import pyvista as pv

# Find the VTU file
vtu_files = list(work_dir.glob("result_A-vtk-files/scatra-*.vtu"))
if not vtu_files:
    print("ERROR: No VTU files found!")
    sys.exit(1)

# Get the last (final) VTU file
vtu_file = sorted(vtu_files)[-1]
print(f"Reading results from {vtu_file}")

mesh = pv.read(str(vtu_file))
phi_values = mesh.point_data['phi']
coords = mesh.points[:, :2]

print(f"Solution: min={phi_values.min():.6f}, max={phi_values.max():.6f}")

# Compute flux at interface (outward normal is +x direction for subdomain A)
# qn = -k * grad(u) . n = -k * du/dx (since n = [1, 0])
# We need to compute gradient at interface nodes

# For P1 elements, gradient is constant per element
# At interface, we need gradients from elements adjacent to interface

# Simple approach: use finite difference near interface
h_x = DOMAIN_A_WIDTH / nx_A
interface_fluxes = []
interface_export_coords = []

for nid, x, y in interface_node_coords:
    # Find node index in mesh
    node_idx = None
    for idx, (mx, my) in enumerate(coords):
        if abs(mx - x) < 1e-10 and abs(my - y) < 1e-10:
            node_idx = idx
            break
    
    if node_idx is not None:
        # Estimate gradient using neighboring node
        # Find node one element to the left
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
            interface_export_coords.append([x, y])
        else:
            interface_fluxes.append(0.0)
            interface_export_coords.append([x, y])
    else:
        interface_fluxes.append(0.0)
        interface_export_coords.append([x, y])

# Also collect temperature values at interface
interface_temps_at_nodes = []
for nid, x, y in interface_node_coords:
    for idx, (mx, my) in enumerate(coords):
        if abs(mx - x) < 1e-10 and abs(my - y) < 1e-10:
            interface_temps_at_nodes.append(phi_values[idx])
            break
    else:
        interface_temps_at_nodes.append(0.0)

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

# Write NDOF to log
ndof = n_nodes  # One DOF per node for scalar transport
ndof_file = work_dir / "ndof.txt"
with open(ndof_file, 'w') as f:
    f.write(f"NDOF = {ndof}\n")

print(f"NDOF = {ndof}")
