#!/usr/bin/env python3
"""
Main script to run the coupled heat conduction simulation.
Uses 4C for subdomain A and Kratos for subdomain B with Dirichlet-Neumann coupling.
"""

import os
import json
import numpy as np
from pathlib import Path
import subprocess
import sys

# Problem parameters
INTERFACE_X = 5/8  # 0.625
DOMAIN_A_WIDTH = INTERFACE_X  # 0 to 0.625
DOMAIN_B_WIDTH = 1.5 - INTERFACE_X  # 0.625 to 1.5
DOMAIN_HEIGHT = 1.0

K_A = 1.0   # thermal conductivity in subdomain A
K_B = 200.0 # thermal conductivity in subdomain B

# Source terms (Python format)
def source_A(x, y):
    return (-12*x**3*y/5 + 14*x**3/5 - 8979*x**2*y/2000 - 5147*x**2/12000 
            - 12*x*y**3/5 + 42*x*y**2/5 - 1779*x*y/800 - 1007*x/1200 
            - 2993*y**3/2000 - 5147*y**2/12000 + 4621*y/2400)

def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Mesh levels: h = 1/8, 1/16, 1/32
MESH_LEVELS = [8, 16, 32]

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

print(f"Probe points A: {len(PROBE_POINTS_A)}")
print(f"Probe points B: {len(PROBE_POINTS_B)}")
print(f"Interface probe points: {len(INTERFACE_PROBE_POINTS)}")

# Main work directory
WORK_DIR = Path("/tmp/coupled_heat_run")
WORK_DIR.mkdir(exist_ok=True)

# Store results
all_results = {}

for level_idx, n_divisions in enumerate(MESH_LEVELS):
    level_num = level_idx + 1
    print(f"\n{'='*60}")
    print(f"LEVEL {level_num}: n_divisions = {n_divisions}, h = 1/{n_divisions}")
    print(f"{'='*60}")
    
    level_dir = WORK_DIR / f"level_{level_num}"
    level_dir.mkdir(exist_ok=True)
    
    # Calculate mesh parameters
    nx_A = int(DOMAIN_A_WIDTH * n_divisions)
    nx_B = int(DOMAIN_B_WIDTH * n_divisions)
    ny = int(DOMAIN_HEIGHT * n_divisions)
    
    print(f"Subdomain A: nx={nx_A}, ny={ny}")
    print(f"Subdomain B: nx={nx_B}, ny={ny}")
    
    # Create participant scripts for this level
    create_participant_scripts(level_num, n_divisions, level_dir)
    
    # Run coupling using OASiS couple tool
    # This will be done via the couple() function call
    
    # For now, let's just test standalone runs first
    print(f"Level {level_num} setup complete")

def create_participant_scripts(level_num, n_divisions, level_dir):
    """Create the participant Python scripts for both solvers."""
    
    nx_A = int(DOMAIN_A_WIDTH * n_divisions)
    nx_B = int(DOMAIN_B_WIDTH * n_divisions)
    ny = int(DOMAIN_HEIGHT * n_divisions)
    
    # Subdomain A work dir
    work_a = level_dir / "subdomain_A"
    work_a.mkdir(exist_ok=True)
    
    # Subdomain B work dir
    work_b = level_dir / "subdomain_B"
    work_b.mkdir(exist_ok=True)
    
    # Write level info
    level_info = {"n_divisions": n_divisions, "level": level_num}
    with open(work_a / "level.json", 'w') as f:
        json.dump(level_info, f)
    with open(work_b / "level.json", 'w') as f:
        json.dump(level_info, f)
    
    # Create 4C YAML generator for subdomain A
    gen_a_script = create_4c_generator(nx_A, ny, work_a)
    with open(work_a / "generate_input.py", 'w') as f:
        f.write(gen_a_script)
    
    # Create Kratos solver script for subdomain B
    kratos_script = create_kratos_solver(nx_B, ny, work_b)
    with open(work_b / "solve.py", 'w') as f:
        f.write(kratos_script)
    
    print(f"Created participant scripts for level {level_num}")

def create_4c_generator(nx, ny, work_dir):
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
    
    # Left edge (x=0) - Dirichlet u=0
    left_nodes = [nid for nid, x, y in nodes if abs(x) < 1e-10]
    for nid in sorted(left_nodes):
        dline_topology.append(f"NODE {nid} DLINE {dline_id}")
    dline_left = dline_id
    dline_id += 1
    
    # Bottom edge (y=0) - Dirichlet u=0
    bottom_nodes = [nid for nid, x, y in nodes if abs(y) < 1e-10]
    for nid in sorted(bottom_nodes):
        dline_topology.append(f"NODE {nid} DLINE {dline_id}")
    dline_bottom = dline_id
    dline_id += 1
    
    # Top edge (y=1) - Dirichlet u=0
    top_nodes = [nid for nid, x, y in nodes if abs(y - DOMAIN_HEIGHT) < 1e-10]
    for nid in sorted(top_nodes):
        dline_topology.append(f"NODE {nid} DLINE {dline_id}")
    dline_top = dline_id
    dline_id += 1
    
    # Right edge (x=0.625) - Interface
    interface_nodes = [nid for nid, x, y in nodes if abs(x - INTERFACE_X) < 1e-10]
    for nid in sorted(interface_nodes):
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
    node_coords = []
    for nid, x, y in nodes:
        node_coords.append(f'NODE {nid} COORD {x:.10f} {y:.10f} 0.0')
    
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
DESIGN VOL TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SOURCE_A_4C_EXPR}"
NODE COORDS:
{chr(10).join('  - "' + nc + '"' for nc in node_coords)}
DLINE-NODE TOPOLOGY:
{chr(10).join('  - "' + dt + '"' for dt in dline_topology)}
TRANSPORT ELEMENTS:
{chr(10).join('  - "' + e + '"' for e in elements)}
'''
    
    return yaml_content

# Source term expression for 4C
SOURCE_A_4C_EXPR = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

def create_kratos_solver(nx, ny, work_dir):
    """Generate Kratos Python solver script for subdomain B."""
    
    # Domain B starts at x = INTERFACE_X
    nodes = []
    node_id = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = INTERFACE_X + i * DOMAIN_B_WIDTH / nx
            y = j * DOMAIN_HEIGHT / ny
            nodes.append((node_id, x, y))
            node_id += 1
    
    n_nodes = len(nodes)
    
    # Elements (triangles from quad splitting)
    elements = []
    elem_id = 1
    for j in range(ny):
        for i in range(nx):
            n1 = (j * (nx + 1) + i) + 1
            n2 = (j * (nx + 1) + i + 1) + 1
            n3 = ((j + 1) * (nx + 1) + i + 1) + 1
            n4 = ((j + 1) * (nx + 1) + i) + 1
            # Split quad into two triangles
            elements.append((elem_id, [n1, n2, n4]))
            elem_id += 1
            elements.append((elem_id, [n2, n3, n4]))
            elem_id += 1
    
    kratos_script = f'''#!/usr/bin/env python3
"""Kratos solver for subdomain B - heat conduction with Neumann BC on interface."""

import KratosMultiphysics as KM
import KratosMultiphysics.ConvectionDiffusionApplication as KCDA
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import json
import os

# Problem parameters
INTERFACE_X = {INTERFACE_X}
DOMAIN_B_WIDTH = {DOMAIN_B_WIDTH}
DOMAIN_HEIGHT = {DOMAIN_HEIGHT}
K_B = {K_B}

# Mesh parameters
nx = {nx}
ny = {ny}
n_nodes = {n_nodes}

# Node coordinates
nodes = [
'''
    
    for nid, x, y in nodes:
        kratos_script += f"    ({x}, {y}),\\n"
    
    kratos_script += f''']

# Element connectivity (triangles)
elements = [
'''
    
    for eid, conn in elements:
        kratos_script += f"    {conn},\\n"
    
    kratos_script += f''']

# Source term function
def source_B(x, y):
    return (-3*x**3*y/50000 + 7*x**3/100000 - 8967*x**2*y/200000 + 28769*x**2/1200000 
            - 3*x*y**3/50000 + 21*x*y**2/100000 - 2938983*x*y/640000 + 7203409*x/3840000 
            - 2989*y**3/200000 + 28769*y**2/1200000 + 26803463*y/3840000 - 1468421/512000)

# Build stiffness matrix and load vector
K = lil_matrix((n_nodes, n_nodes))
F = np.zeros(n_nodes)

for elem_conn in elements:
    ids = [nid - 1 for nid in elem_conn]  # Convert to 0-based
    coords = np.array([nodes[nid-1] for nid in elem_conn])
    x = coords[:, 0]
    y = coords[:, 1]
    
    # Triangle area
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
    
    # Shape function derivatives
    b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]])
    c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]])
    
    # Element stiffness matrix
    Ke = (K_B / (4.0 * area)) * (np.outer(b, b) + np.outer(c, c))
    
    # Element load vector (using centroid integration)
    xc = np.mean(x)
    yc = np.mean(y)
    fc = source_B(xc, yc)
    Fe = fc * area / 3.0 * np.ones(3)
    
    # Assemble
    for a in range(3):
        for bb in range(3):
            K[ids[a], ids[bb]] += Ke[a, bb]
        F[ids[a]] += Fe[a]

K = K.tocsr()

# Identify boundary nodes
interface_nodes = set()  # Left edge of domain B (x = INTERFACE_X)
right_nodes = set()     # Right edge (x = 1.5)
bottom_nodes = set()    # Bottom edge (y = 0)
top_nodes = set()       # Top edge (y = 1)

tol = 1e-10
for nid, (x, y) in enumerate(nodes):
    if abs(x - INTERFACE_X) < tol:
        interface_nodes.add(nid)
    if abs(x - (INTERFACE_X + DOMAIN_B_WIDTH)) < tol:
        right_nodes.add(nid)
    if abs(y) < tol:
        bottom_nodes.add(nid)
    if abs(y - DOMAIN_HEIGHT) < tol:
        top_nodes.add(nid)

# Dirichlet nodes (outer boundary, excluding interface corners)
dirichlet_nodes = right_nodes | bottom_nodes | top_nodes
# Remove interface corners from Dirichlet set (they're handled separately)
interface_corners = {nid for nid in interface_nodes if nid in (bottom_nodes | top_nodes)}
dirichlet_nodes -= interface_corners

# Read imports from partner A (flux values)
work_dir = os.getcwd()
imports_file = os.path.join(work_dir, "imports.json")
if os.path.exists(imports_file):
    with open(imports_file) as f:
        imports = json.load(f)
    if "A" in imports:
        partner_data = imports["A"]
        interface_fluxes_from_partner = partner_data.get("normal_fluxes", [])
        interface_coords_from_partner = partner_data.get("coordinates", [])
        print(f"Received {len(interface_fluxes_from_partner)} flux values from partner A")
    else:
        interface_fluxes_from_partner = []
else:
    interface_fluxes_from_partner = []
    print("No imports.json found (first iteration)")

# Apply Neumann BC on interface (if we have flux data)
if interface_fluxes_from_partner and len(interface_fluxes_from_partner) > 0:
    # Map partner fluxes to our interface nodes
    for idx, (px, py) in enumerate(interface_coords_from_partner):
        qn = interface_fluxes_from_partner[idx]
        # Find closest interface node
        min_dist = float('inf')
        closest_node = None
        for nid in interface_nodes:
            x, y = nodes[nid]
            dist = (x - px)**2 + **(y - py)2
            if dist < min_dist:
                min_dist = dist
                closest_node = nid
        if closest_node is not None:
            # Neumann BC: add flux to RHS (qn is outward normal flux from partner)
            # Our outward normal on left edge is -x direction
            # Partner's outward normal on right edge is +x direction
            # Flux continuity: k_A * du_A/dx = k_B * du_B/dx at interface
            # The flux we receive is -k_A * du_A/dx (partner's outward flux)
            # We apply it as -k_B * du_B/dx = received_flux on our side
            # In weak form: integral(q * v) on boundary adds to RHS
            F[closest_node] += qn * (DOMAIN_HEIGHT / ny)  # Approximate edge length contribution

# Solve system with Dirichlet BCs
u = np.zeros(n_nodes)
interior_nodes = list(set(range(n_nodes)) - dirichlet_nodes - interface_nodes)

# Partition system
if interior_nodes:
    K_int = K[np.ix_(interior_nodes, interior_nodes)]
    F_int = F[interior_nodes]
    
    # Apply Dirichlet BCs (u=0 on outer boundary)
    # Since u=0, no contribution to RHS from Dirichlet nodes
    
    u[interior_nodes] = spsolve(K_int, F_int)

print(f"Solution computed: min={u.min():.6f}, max={u.max():.6f}")

# Export interface data (temperature values)
interface_export_coords = []
interface_temps = []

for nid in sorted(interface_nodes):
    x, y = nodes[nid]
    interface_export_coords.append([x, y])
    interface_temps.append(u[nid])

# Compute flux at interface (outward normal is -x direction for subdomain B)
# qn = -k * grad(u) . n = -k * (-du/dx) = k * du/dx
h_x = DOMAIN_B_WIDTH / nx
interface_fluxes = []

for nid in sorted(interface_nodes):
    x, y = nodes[nid]
    # Find node one element to the right
    x_right = x + h_x
    right_val = None
    for other_nid, (ox, oy) in enumerate(nodes):
        if abs(ox - x_right) < h_x/2 and abs(oy - y) < h_x/2:
            right_val = u[other_nid]
            break
    
    if right_val is not None:
        du_dx = (right_val - u[nid]) / h_x
        qn = K_B * du_dx  # outward normal flux (n = [-1, 0])
        interface_fluxes.append(qn)
    else:
        interface_fluxes.append(0.0)

# Write exports.json
exports = {
    "field_name": "temperature",
    "coordinates": interface_export_coords,
    "values": interface_temps,
    "normal_fluxes": interface_fluxes,
    "n_points": len(interface_export_coords)
}

with open(os.path.join(work_dir, "exports.json"), 'w') as f:
    json.dump(exports, f, indent=2)

print(f"Exported {len(interface_export_coords)} interface points")
print(f"Interface temperature: min={min(interface_temps):.6f}, max={max(interface_temps):.6f}")
print(f"Interface flux: min={min(interface_fluxes):.6f}, max={max(interface_fluxes):.6f}")

# Write NDOF
with open(os.path.join(work_dir, "ndof.txt"), 'w') as f:
    f.write(f"NDOF = {n_nodes}\\n")

print(f"NDOF = {n_nodes}")
'''
    
    return kratos_script

if __name__ == "__main__":
    print("Coupled Heat Conduction Simulation")
    print("=" * 60)
    print(f"Subdomain A: (0, {DOMAIN_A_WIDTH}) x (0, {DOMAIN_HEIGHT}), k = {K_A}")
    print(f"Subdomain B: ({INTERFACE_X}, 1.5) x (0, {DOMAIN_HEIGHT}), k = {K_B}")
    print(f"Interface at x = {INTERFACE_X}")
    print("=" * 60)
    
    # Run for each mesh level
    for level_idx, n_divisions in enumerate(MESH_LEVELS):
        level_num = level_idx + 1
        level_dir = WORK_DIR / f"level_{level_num}"
        level_dir.mkdir(exist_ok=True)
        
        create_participant_scripts(level_num, n_divisions, level_dir)
        print(f"Level {level_num} scripts created")
