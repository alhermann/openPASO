#!/usr/bin/env python3
"""
4C participant for Subdomain A (Dirichlet side) in coupled thermal problem.
Domain: (0, 0.625) x (0, 1), k = 1
Receives temperature at interface from partner B, exports flux.
"""
import os, sys, json, numpy as np, subprocess, glob

work_dir = os.getcwd()
level = int(os.environ.get('LEVEL', 1))
resolution = int(os.environ.get('RESOLUTION', 8))

# Subdomain A parameters
x_A_max = 0.625
y_max = 1.0
k_A = 1.0

nx_A = int(x_A_max * resolution)
ny = int(y_max * resolution)

# Source term f(x,y) in subdomain A (4C uses ^ for power)
source_A = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

# Generate mesh nodes
nodes = []
node_map = {}
nid = 1
for j in range(ny + 1):
    for i in range(nx_A + 1):
        x = i / resolution
        y = j / resolution
        nodes.append((nid, x, y))
        node_map[(i, j)] = nid
        nid += 1

n_nodes = len(nodes)

# Generate QUAD4 elements
elements = []
eid = 1
for j in range(ny):
    for i in range(nx_A):
        n1 = node_map[(i, j)]
        n2 = node_map[(i+1, j)]
        n3 = node_map[(i+1, j+1)]
        n4 = node_map[(i, j+1)]
        elements.append(f"{eid} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
        eid += 1

# Define boundary lines and volume
dline_assignments = {}
dvol_elements = []
for (i, j), nid in node_map.items():
    if i == 0:
        dline_assignments.setdefault(1, []).append(nid)  # left: u=0
    if j == 0:
        dline_assignments.setdefault(2, []).append(nid)  # bottom: u=0
    if j == ny:
        dline_assignments.setdefault(3, []).append(nid)  # top: u=0
    if i == nx_A:
        dline_assignments.setdefault(4, []).append(nid)  # right: interface

# All elements belong to DVOL 1
dvol_elements = list(range(1, len(elements) + 1))

# Read imports from partner B
imports_file = os.path.join(work_dir, 'imports.json')
if os.path.exists(imports_file):
    with open(imports_file, 'r') as f:
        imports = json.load(f)
    if 'B' in imports:
        imp_data = imports['B']
        iface_coords = np.array(imp_data['coordinates'])
        iface_temps = np.array(imp_data['values'])
        sidx = np.argsort(iface_coords[:, 1])
        y_iface_sorted = iface_coords[sidx, 1]
        t_iface_sorted = iface_temps[sidx]
    else:
        y_iface_sorted = np.linspace(0, 1, ny + 1)
        t_iface_sorted = np.zeros(len(y_iface_sorted))
else:
    y_iface_sorted = np.linspace(0, 1, ny + 1)
    t_iface_sorted = np.zeros(len(y_iface_sorted))

# Build 4C YAML input
yaml_lines = [
    "TITLE:",
    '  - "Subdomain A - 4C thermal participant"',
    "PROBLEM SIZE:",
    "  DIM: 2",
    "PROBLEM TYPE:",
    '  PROBLEMTYPE: "Scalar_Transport"',
    "SCALAR TRANSPORT DYNAMIC:",
    '  TIMEINTEGR: "Stationary"',
    '  SOLVERTYPE: "linear_full"',
    '  VELOCITYFIELD: "zero"',
    "  TIMESTEP: 1.0",
    "  NUMSTEP: 1",
    "  MAXTIME: 1.0",
    "  LINEAR_SOLVER: 1",
    "SOLVER 1:",
    '  SOLVER: "UMFPACK"',
    "MATERIALS:",
    "  - MAT: 1",
    "    MAT_scatra:",
    f"      DIFFUSIVITY: {k_A}",
    "",
    "DESIGN VOL NEUMANN CONDITIONS:",
    "  - E: 1",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [1.0]",
    "    FUNCT: [1]",
    "",
    "FUNCT1:",
    "  - COMPONENT: 0",
    f'    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{source_A}"',
    "",
    "DESIGN LINE DIRICH CONDITIONS:",
    "  - E: 1",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    "  - E: 2",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    "  - E: 3",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    "  - E: 4",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    f"    VAL: [{np.mean(t_iface_sorted):.15e}]",
    "    FUNCT: [0]",
    "",
    "DLINE-NODE TOPOLOGY:"
]

for lid in sorted(dline_assignments.keys()):
    for nid in dline_assignments[lid]:
        yaml_lines.append(f'  - "NODE {nid} DLINE {lid}"')

yaml_lines.extend(["", "DVOL-ELEMENT TOPOLOGY:"])
for eid in dvol_elements:
    yaml_lines.append(f'  - "ELEMENT {eid} DVOL 1"')

yaml_lines.extend(["", "NODE COORDS:"])
for nid, x, y in nodes:
    yaml_lines.append(f'  - "NODE {nid} COORD {x:.15e} {y:.15e} 0.0"')

yaml_lines.extend(["", "TRANSPORT ELEMENTS:"])
for el in elements:
    yaml_lines.append(f'  - "{el}"')

input_file = os.path.join(work_dir, f'subA_L{level}.4C.yaml')
with open(input_file, 'w') as f:
    f.write('\n'.join(yaml_lines))

# Run 4C
output_prefix = os.path.join(work_dir, f'subA_L{level}')
cmd = f"LD_LIBRARY_PATH=/opt/4C-dependencies/lib mpirun --allow-run-as-root -np 1 /home/alexander/4C/build/4C {input_file} {output_prefix}"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

# Read results
vtu_files = glob.glob(os.path.join(work_dir, f'{output_prefix}-vtk-files/scatra-*.vtu'))
if not vtu_files:
    print("ERROR: No VTU files found")
    sys.exit(1)

vtu_file = sorted(vtu_files)[-1]
import pyvista as pv
mesh = pv.read(vtu_file)
temps = mesh.point_data.get('phi_1')
coords = mesh.points

if temps is None:
    print("ERROR: phi_1 field not found")
    sys.exit(1)

# Find interface nodes (at x = 0.625)
iface_x = 0.625
tol = 1e-6
iface_indices = [idx for idx, (x, y, z) in enumerate(coords) if abs(x - iface_x) < tol]

# Compute outward normal flux at interface
interface_fluxes = []
dx = 1.0 / resolution
for idx in iface_indices:
    x, y, z = coords[idx]
    best_dist = float('inf')
    best_temp = None
    for jdx, (jx, jy, jz) in enumerate(coords):
        if jx < x - dx/3:
            dist = (jx - (x - dx))**2 + (jy - y)**2
            if dist < best_dist:
                best_dist = dist
                best_temp = temps[jdx]
    
    if best_temp is not None:
        grad_u_x = (temps[idx] - best_temp) / dx
        flux = -k_A * grad_u_x
    else:
        flux = 0.0
    interface_fluxes.append(flux)

# Export data
export_data = {
    'field_name': 'temperature',
    'n_points': len(iface_indices),
    'coordinates': [[iface_x, coords[idx, 1], 0.0] for idx in iface_indices],
    'values': [temps[idx] for idx in iface_indices],
    'normal_fluxes': interface_fluxes
}

with open(os.path.join(work_dir, 'exports.json'), 'w') as f:
    json.dump({'A': export_data}, f)

print(f"NDOF = {len(temps)}")
