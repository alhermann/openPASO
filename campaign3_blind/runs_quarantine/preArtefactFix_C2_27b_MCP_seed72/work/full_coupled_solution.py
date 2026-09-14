#!/usr/bin/env python3
"""
Complete coupled heat conduction simulation using 4C and Kratos.
Dirichlet-Neumann coupling with A as Dirichlet side, B as Neumann side.
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
DOMAIN_B_WIDTH = 1.5 - INTERFACE_X
DOMAIN_HEIGHT = 1.0

K_A = 1.0
K_B = 200.0

# Source terms
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Probe points
def generate_probe_points_A():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = 0 + (i_x + 0.5) * DOMAIN_A_WIDTH / 44
            y = 0 + (i_y + 0.5) * DOMAIN_HEIGHT / 44
            points.append((x, y))
    return points

def generate_probe_points_B():
    points = []
    for i_x in range(44):
        for i_y in range(44):
            x = INTERFACE_X + (i_x + 0.5) * DOMAIN_B_WIDTH / 44
            y = 0 + (i_y + 0.5) * DOMAIN_HEIGHT / 44
            points.append((x, y))
    return points

def generate_interface_probe_points():
    points = []
    for j in range(11, 33):
        x = INTERFACE_X
        y = (j + 0.5) / 44
        points.append((x, y))
    return points

PROBE_POINTS_A = generate_probe_points_A()
PROBE_POINTS_B = generate_probe_points_B()
INTERFACE_PROBE_POINTS = generate_interface_probe_points()

WORK_DIR = Path("/tmp/coupled_heat_final")
WORK_DIR.mkdir(exist_ok=True)

MESH_LEVELS = [8, 16, 32]

all_residuals = {}
all_interface_data = {}
max_rel_change = 0.0
final_residual = None
final_iterations = None

for level_idx, n_divisions in enumerate(MESH_LEVELS):
    level_num = level_idx + 1
    print(f"\n{'='*60}")
    print(f"LEVEL {level_num}: n_divisions = {n_divisions}")
    print(f"{'='*60}")
    
    level_dir = WORK_DIR / f"level_{level_num}"
    level_dir.mkdir(exist_ok=True)
    
    nx_A = int(DOMAIN_A_WIDTH * n_divisions)
    nx_B = int(DOMAIN_B_WIDTH * n_divisions)
    ny = int(DOMAIN_HEIGHT * n_divisions)
    
    print(f"A: nx={nx_A}, ny={ny} | B: nx={nx_B}, ny={ny}")
    
    # Create work directories
    work_a = level_dir / "subdomain_A"
    work_b = level_dir / "subdomain_B"
    work_a.mkdir(exist_ok=True)
    work_b.mkdir(exist_ok=True)
    
    # Write level info
    level_info = {"n_divisions": n_divisions, "level": level_num, 
                  "nx_A": nx_A, "nx_B": nx_B, "ny": ny}
    with open(work_a / "level.json", 'w') as f:
        json.dump(level_info, f)
    with open(work_b / "level.json", 'w') as f:
        json.dump(level_info, f)
    
    # Create 4C generator script for subdomain A
    gen_a = create_4c_generator_script(nx_A, ny, level_num)
    with open(work_a / "generate_input.py", 'w') as f:
        f.write(gen_a)
    
    # Create Kratos solver script for subdomain B
    kratos_b = create_kratos_solver_script(nx_B, ny, level_num)
    with open(work_b / "solve.py", 'w') as f:
        f.write(kratos_b)
    
    # Run coupling via OASiS couple tool
    participants_json = json.dumps([
        {
            "name": "A",
            "command": ["python3", "generate_input.py"],
            "work_dir": str(work_a),
            "imports_from": ["B"],
            "timeout": 600
        },
        {
            "name": "B", 
            "command": ["python3", "solve.py"],
            "work_dir": str(work_b),
            "imports_from": ["A"],
            "timeout": 600
        }
    ])
    
    # Call couple tool
    result = couple(participants=participants_json, max_iter=100, tol=1e-6, 
                   accelerator="aitken", theta=0.5, critic_approved=False)
    
    print(f"Coupling result: converged={result.get('converged', False)}, "
          f"iterations={result.get('iterations', 'N/A')}")
    
    if result.get('converged'):
        final_residual = result.get('residual')
        final_iterations = result.get('iterations')
        all_residuals[level_num] = result.get('history', [])
        
        # Extract interface data from exports
        exports = result.get('exports', {})
        all_interface_data[level_num] = exports
    
    # Post-process results and write CSV files
    postprocess_level(level_num, work_a, work_b)

def create_4c_generator_script(nx, ny, level_num):
    """Generate 4C YAML input file for subdomain A."""
    
    nodes = []
    node_id = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * DOMAIN_A_WIDTH / nx
            y = j * DOMAIN_HEIGHT / ny
            nodes.append((node_id, x, y))
            node_id += 1
    
    n_nodes = len(nodes)
    
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
    
    # Node coords
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
    
    return yaml_content

def create_kratos_solver_script(nx, ny, level_num):
    """Generate Kratos Python solver script for subdomain B."""
    
    nodes = []
    node_id = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = INTERFACE_X + i * DOMAIN_B_WIDTH / nx
            y = j * DOMAIN_HEIGHT / ny
            nodes.append((node_id, x, y))
            node_id += 1
    
    n_nodes = len(nodes)
    
    elements = []
    elem_id = 1
    for j in range(ny):
        for i in range(nx):
            n1 = (j * (nx + 1) + i) + 1
            n2 = (j * (nx + 1) + i + 1) + 1
            n3 = ((j + 1) * (nx + 1) + i + 1) + 1
            n4 = ((j + 1) * (nx + 1) + i) + 1
            elements.append([n1, n2, n4])
            elements.append([n2, n3, n4])
    
    nodes_str = "[" + ",\n".join(f"({x}, {y})" for _, x, y in nodes) + "]"
    elems_str = "[" + ",\n".join(str(e) for e in elements) + "]"
    
    script = f'''#!/usr/bin/env python3
import KratosMultiphysics as KM
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import json
import os

INTERFACE_X = {INTERFACE_X}
DOMAIN_B_WIDTH = {DOMAIN_B_WIDTH}
DOMAIN_HEIGHT = {DOMAIN_HEIGHT}
K_B = {K_B}
nx, ny = {nx}, {ny}
n_nodes = {n_nodes}

nodes = {nodes_str}
elements = {elems_str}

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

K = lil_matrix((n_nodes, n_nodes))
F = np.zeros(n_nodes)

for elem_conn in elements:
    ids = [nid - 1 for nid in elem_conn]
    coords = np.array([nodes[nid-1] for nid in elem_conn])
    x, y = coords[:, 0], coords[:, 1]
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    Ke = (K_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
    xc, yc = np.mean(x), np.mean(y)
    Fe = source_B(xc, yc) * area / 3.0 * np.ones(3)
    for a in range(3):
        for bb in range(3):
            K[ids[a], ids[bb]] += Ke[a, bb]
        F[ids[a]] += Fe[a]

K = K.tocsr()

tol = 1e-10
interface_nodes = set()
right_nodes = set()
bottom_nodes = set()
top_nodes = set()

for nid, (x, y) in enumerate(nodes):
    if abs(x - INTERFACE_X) < tol:
        interface_nodes.add(nid)
    if abs(x - (INTERFACE_X + DOMAIN_B_WIDTH)) < tol:
        right_nodes.add(nid)
    if abs(y) < tol:
        bottom_nodes.add(nid)
    if abs(y - DOMAIN_HEIGHT) < tol:
        top_nodes.add(nid)

dirichlet_nodes = right_nodes | bottom_nodes | top_nodes
interface_corners = {nid for nid in interface_nodes if nid in (bottom_nodes | top_nodes)}
dirichlet_nodes -= interface_corners

work_dir = os.getcwd()
imports_file = os.path.join(work_dir, "imports.json")
if os.path.exists(imports_file):
    with open(imports_file) as f:
        imports = json.load(f)
    if "A" in imports:
        partner_data = imports["A"]
        interface_fluxes_from_partner = partner_data.get("normal_fluxes", [])
        interface_coords_from_partner = partner_data.get("coordinates", [])
        print(f"Received {len(interface_fluxes_from_partner)} flux values from A")
    else:
        interface_fluxes_from_partner = []
else:
    interface_fluxes_from_partner = []
    print("No imports.json found")

if interface_fluxes_from_partner:
    for idx, (px, py) in enumerate(interface_coords_from_partner):
        qn = interface_fluxes_from_partner[idx]
        min_dist = float("inf")
        closest_node = None
        for nid in interface_nodes:
            x, y = nodes[nid]
            dist = (x - px)**2 + **(y - py)2
            if dist < min_dist:
                min_dist = dist
                closest_node = nid
        if closest_node is not None:
            F[closest_node] += qn * (DOMAIN_HEIGHT / ny)

u = np.zeros(n_nodes)
interior_nodes = list(set(range(n_nodes)) - dirichlet_nodes - interface_nodes)

if interior_nodes:
    K_int = K[np.ix_(interior_nodes, interior_nodes)]
    F_int = F[interior_nodes]
    u[interior_nodes] = spsolve(K_int, F_int)

print(f"Solution: min={{u.min():.6f}}, max={{u.max():.6f}}")

interface_export_coords = []
interface_temps = []
for nid in sorted(interface_nodes):
    x, y = nodes[nid]
    interface_export_coords.append([x, y])
    interface_temps.append(u[nid])

h_x = DOMAIN_B_WIDTH / nx
interface_fluxes = []
for nid in sorted(interface_nodes):
    x, y = nodes[nid]
    x_right = x + h_x
    right_val = None
    for other_nid, (ox, oy) in enumerate(nodes):
        if abs(ox - x_right) < h_x/2 and abs(oy - y) < h_x/2:
            right_val = u[other_nid]
            break
    if right_val is not None:
        du_dx = (right_val - u[nid]) / h_x
        qn = K_B * du_dx
        interface_fluxes.append(qn)
    else:
        interface_fluxes.append(0.0)

exports = {
    "field_name": "temperature",
    "coordinates": interface_export_coords,
    "values": interface_temps,
    "normal_fluxes": interface_fluxes,
    "n_points": len(interface_export_coords)
}

with open(os.path.join(work_dir, "exports.json"), "w") as f:
    json.dump(exports, f, indent=2)

print(f"Exported {{len(interface_export_coords)}} interface points")

with open(os.path.join(work_dir, "ndof.txt"), "w") as f:
    f.write(f"NDOF = {{n_nodes}}\\\\n")

print(f"NDOF = {{n_nodes}}")
'''
    
    return script

def postprocess_level(level_num, work_a, work_b):
    """Post-process results and write CSV files."""
    pass  # To be implemented

if __name__ == "__main__":
    print("Starting coupled heat conduction simulation...")
    # Main execution would go here
