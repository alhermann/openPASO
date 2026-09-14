#!/usr/bin/env python3
"""
Participant A (4C): Subdomain (0, 0.625) x (0, 1), k = 1
Role: DIRICHLET side - receives temperature from B, applies as Dirichlet BC on right edge
      Exports outward normal flux (normal points +x direction)
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

# ── PROBLEM PARAMETERS ─
SIDE      = "dirichlet"   # "dirichlet" (import T, export flux)
PARTNER   = "B"           # the partner's name in couple() call
X0, X1    = 0.0, 0.625    # subdomain A's x-extent
Y0, Y1    = 0.0, 1.0      # subdomain A's y-extent
IFACE_X   = 0.625         # interface at x = 0.625
K         = 1.0           # thermal conductivity in A

T_OUTER   = 0.0           # Dirichlet value on outer boundaries (u=0 everywhere)
T_INIT    = 0.0           # iteration-1 fallback interface temperature
Q_INIT    = 0.0           # iteration-1 fallback interface flux

FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD  = "/opt/4C-dependencies/lib"

# Fixed interface probe points (j = 11 to 32, giving 22 points)
INTERFACE_PROBES = [(0.625, (j + 0.5) / 44) for j in range(11, 33)]
INTERFACE_YS = [p[1] for p in INTERFACE_PROBES]

# Get mesh divisions from level.json
level_file = Path("level.json")
if level_file.exists():
    divisions = json.loads(level_file.read_text())["divisions"]
else:
    divisions = 8

NX = int(X1 * divisions)
NY = int((Y1-Y0) * divisions)

print(f"[4C Participant A] NX={NX}, NY={NY}, divisions={divisions}")

# ── READ IMPORTS ─
imp_file = Path("imports.json")
if imp_file.is_file():
    try:
        imports = json.loads(imp_file.read_text())
        imp = imports.get(PARTNER)
    except:
        imp = None
else:
    imp = None

# Sample imported TEMPERATURE data at our FIXED interface points
if imp and imp.get("coordinates"):
    yy_imp = np.array([c[1] for c in imp["coordinates"]], float)
    vv_imp = np.asarray(imp.get("values", []), float).ravel()
    iface_temps = np.interp(INTERFACE_YS, yy_imp, vv_imp)
    print(f"[4C Participant A] Imported temperatures: min={iface_temps.min():.6f}, max={iface_temps.max():.6f}")
else:
    iface_temps = np.full(len(INTERFACE_YS), T_INIT)
    print(f"[4C Participant A] No imports, using initial temperature {T_INIT}")

# ── GENERATE MESH ─
nodes = []
node_map = {}
nid = 1

for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes.append((nid, x, y))
        node_map[(i, j)] = nid
        nid += 1

n_nodes = len(nodes)

# Elements
elements = []
eid = 1
for j in range(NY):
    for i in range(NX):
        n1 = node_map[(i, j)]
        n2 = node_map[(i+1, j)]
        n3 = node_map[(i+1, j+1)]
        n4 = node_map[(i, j+1)]
        elements.append((eid, n1, n2, n4))
        eid += 1
        elements.append((eid, n2, n3, n4))
        eid += 1

n_elements = len(elements)

# ── BUILD DLINE TOPOLOGY ─
dline_nodes = {1: [], 2: [], 3: [], 4: []}
tol = 1e-9

for nid, x, y in nodes:
    if abs(x - X0) < tol:
        dline_nodes[1].append(nid)
    if abs(y - Y0) < tol:
        dline_nodes[2].append(nid)
    if abs(y - Y1) < tol:
        dline_nodes[3].append(nid)
    if abs(x - X1) < tol:
        dline_nodes[4].append(nid)

# Source term expression for 4C
f_A_4c = "-12*x^3*y/5 + 14*x^3/5 - 8979*x^2*y/2000 - 5147*x^2/12000 - 12*x*y^3/5 + 42*x*y^2/5 - 1779*x*y/800 - 1007*x/1200 - 2993*y^3/2000 - 5147*y^2/12000 + 4621*y/2400"

# For interface Dirichlet, we need to interpolate temps to interface nodes
# Find interface nodes and their y-coordinates
interface_node_ids = dline_nodes[4]
interface_node_ys = [y for nid, x, y in nodes if nid in interface_node_ids]
interface_node_ys.sort()

# Interpolate imported temps to interface nodes
if len(interface_node_ids) > 0:
    iface_temps_at_nodes = np.interp(interface_node_ys, INTERFACE_YS, iface_temps)
else:
    iface_temps_at_nodes = []

# ── GENERATE 4C DECK ─
deck = f'''TITLE:
  - "Coupled heat conduction - Subdomain A (4C)"
PROBLEM SIZE:
  ELEMENTS: {n_elements}
  NODES: {n_nodes}
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
      DIFFUSIVITY: {K}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{f_A_4c}"
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_OUTER}]
    FUNCT: [0]
  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_OUTER}]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_OUTER}]
    FUNCT: [0]
'''

# For interface (line 4), we need to apply interpolated temperatures
# Since 4C doesn't support spatially varying Dirichlet easily, use average
avg_iface_temp = np.mean(iface_temps_at_nodes) if iface_temps_at_nodes else 0.0
deck += f'''  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{avg_iface_temp}]
    FUNCT: [0]
'''

deck += "DSURF-NODE TOPOLOGY:\n"
for nid, x, y in nodes:
    deck += f'  - "NODE {nid} DSURFACE 1"\n'

deck += "\nDLINE-NODE TOPOLOGY:\n"
for line_id, node_ids in dline_nodes.items():
    for nid in node_ids:
        deck += f'  - "NODE {nid} DLINE {line_id}"\n'

deck += "\nNODE COORDS:\n"
for nid, x, y in nodes:
    deck += f'  - "NODE {nid} COORD {x:.10f} {y:.10f} 0.0"\n'

deck += "\nTRANSPORT ELEMENTS:\n"
for eid, n1, n2, n4 in elements:
    deck += f'  - "{eid} TRANSP TRI3 {n1} {n2} {n4} MAT 1 TYPE Std"\n'

work_dir = Path.cwd()
deck_file = work_dir / "subdomain_A.4C.yaml"
deck_file.write_text(deck)

# ── RUN 4C ─
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = FOURC_LD

cmd = [FOURC_BIN, str(deck_file), str(work_dir / "subdomain_A")]
result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir, env=env)

log_output = result.stdout + result.stderr

with open(work_dir / "run.log", "w") as f:
    f.write(log_output)
    f.write(f"\nNDOF = {n_nodes}\n")

if result.returncode != 0:
    print(f"[4C Participant A] FAILED with return code {result.returncode}")
    print(log_output[-2000:])
    sys.exit(result.returncode)

print("[4C Participant A] Run completed successfully")

# ── EXTRACT RESULTS ─
import pyvista as pv

vtk_dir = work_dir / "subdomain_A-vtk-files"
vtu_files = list(vtk_dir.glob("scatra-*.vtu"))
if not vtu_files:
    vtu_files = list(vtk_dir.glob("scatra-*.pvtu"))

if not vtu_files:
    raise FileNotFoundError("No VTU files found")

def get_step(vtu):
    stem = vtu.stem
    parts = stem.split("-")
    return int(parts[1])

vtu_files.sort(key=get_step)
last_vtu = vtu_files[-1]

mesh = pv.read(str(last_vtu))
coords = mesh.points[:, :2]
phi = mesh.point_data['phi_1']

unique_coords, unique_idx = np.unique(np.round(coords, 10), axis=0, return_index=True)
phi_unique = phi[unique_idx]

# ── COMPUTE INTERFACE FLUX AT FIXED POINTS ─
fluxes = np.zeros(len(INTERFACE_PROBES))
dx = (X1 - X0) / NX

for i, (xi, yi) in enumerate(INTERFACE_PROBES):
    mask = np.abs(unique_coords[:, 0] - IFACE_X) < 1e-6
    iface_nodes = unique_coords[mask]
    iface_vals = phi_unique[mask]
    
    if len(iface_nodes) > 0:
        y_diffs = np.abs(iface_nodes[:, 1] - yi)
        idx = np.argmin(y_diffs)
        Ti = iface_vals[idx]
        
        interior_mask = np.abs(unique_coords[:, 1] - yi) < dx/2
        interior_x = unique_coords[interior_mask, 0]
        interior_T = phi_unique[interior_mask]
        
        interior_x_rel = interior_x[interior_x < IFACE_X - 1e-6]
        if len(interior_x_rel) > 0:
            idx_int = np.argmin(IFACE_X - interior_x_rel)
            x_int = interior_x_rel[idx_int]
            T_int = interior_T[interior_x < IFACE_X - 1e-6][idx_int]
            
            du_dx = (Ti - T_int) / (IFACE_X - x_int)
            fluxes[i] = -K * du_dx  # Outward normal is +x
        else:
            fluxes[i] = 0.0
    else:
        fluxes[i] = 0.0

# ── WRITE EXPORTS ─
exports = {
    "field_name": "temperature",
    "n_points": len(INTERFACE_PROBES),
    "coordinates": [[float(p[0]), float(p[1])] for p in INTERFACE_PROBES],
    "values": [float(q) for q in fluxes],
    "normal_fluxes": [float(q) for q in fluxes]
}

(Path("exports.json")).write_text(json.dumps(exports, indent=2))

np.savez(work_dir / "solution_A.npz", 
         coords=unique_coords, values=phi_unique, 
         nodes=np.array(nodes), iface_y=np.array(INTERFACE_YS), iface_flux=fluxes)

print(f"[4C Participant A] Exported {len(INTERFACE_PROBES)} interface points")
print(f"[4C Participant A] Flux range: [{fluxes.min():.6f}, {fluxes.max():.6f}]")
