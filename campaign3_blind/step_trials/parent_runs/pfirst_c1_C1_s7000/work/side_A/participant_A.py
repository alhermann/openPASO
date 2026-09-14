#!/usr/bin/env python3
"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
The traction uses the heat flux's sign rule (minus the flux through n_out): the two sides' exports
cancel and the Neumann partner applies them UNCHANGED. The task's OUTWARD traction is MINUS (qx, qy).
TWO 4C RUNS PER ITERATION: run T (Scalar_Transport with CALCFLUX_BOUNDARY), run U (TSI slab).
"""
import atexit
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import meshio

# Problem parameters for subdomain A
X0, X1 = 0.0, 0.625  # subdomain A x-extent
Y0, Y1 = 0.0, 1.0    # subdomain A y-extent
IFACE_X = 0.625      # interface at right edge
KV = 1.0             # conductivity
LAM = 600.0          # lambda
MU = 400.0           # mu
BETA = 1.0           # thermal stress coefficient

# Derived material properties
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)  # E = 1040
NU = LAM / (2.0 * (LAM + MU))                     # nu = 3/10
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)             # alpha = 1/2600

# 4C binary path
FOURC_BIN = "/home/alexander/4C/build/4C"

# Source terms in 4C format (^ instead of **)
SRC_T = "-6*x^3*y + 8*x^3/5 - x^2*y/2 - 2*x^2/3 - 6*x*y^3 + 24*x*y^2/5 + 67*x*y/10 - 11*x/15 - y^3/6 - 2*y^2/3 + 5*y/6"
SRC_UX = "-33*x^2*y^3/5 - 312*x^2*y^2/25 + 453*x^2*y/25 - 18*x^2/5 + 599*x*y^3/10 + 4754*x*y^2/75 - 17597*x*y/150 + 112*x/5 - 84*y^5/25 - 147*y^4/25 + 18277*y^3/300 + 893*y^2/30 - 1549*y/20 + 15"
SRC_UY = "3*x^3*y^2 - 8*x^3*y/5 - x^3/5 + 2693*x^2*y^2/20 + 7106*x^2*y/75 - 26333*x^2/300 - 12*x*y^4 - 84*x*y^3/5 + 4241*x*y^2/20 + 301*x*y/3 - 2173*x/20 + 56*y^4/15 + 392*y^3/75 - 364*y^2/25 + 28*y/5"

# Read config for level and mesh size
CFG = {"level": 1, "nx": 8, "ny": 8}
if Path("config.json").is_file():
    CFG.update(json.loads(Path("config.json").read_text()))
env_cfg = os.environ.get("OASIS_CONFIG_JSON", "{}")
if env_cfg:
    CFG.update(json.loads(env_cfg))

NX = int(CFG["nx"])
NY = int(CFG["ny"])
LEVEL = int(CFG.get("level", 1))

# Build 2-D node layout (NX x NY cells on [X0,X1] x [Y0,Y1])
dx = (X1 - X0) / NX
dy = (Y1 - Y0) / NY
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * dx
        y = Y0 + j * dy
        nodes.append((x, y))

N_NODES = len(nodes)
NODES_PER_ROW = NX + 1

# Interface nodes (at x = IFACE_X, which is the right edge)
# Node id = i + 1 + (NX + 1) * j, so right edge has i = NX
interface_node_ids = [(NX + 1) + (NX + 1) * j for j in range(NY + 1)]  # 1-based ids
# Interior interface nodes (exclude corners at y=Y0 and y=Y1)
interior = [nid for nid in interface_node_ids if nid not in [interface_node_ids[0], interface_node_ids[-1]]]

# Outer boundary nodes (left, top, bottom - NOT including interface)
outer_nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        nid = i + 1 + (NX + 1) * j
        x, y = nodes[i + j * NODES_PER_ROW]
        # Left edge (i=0), bottom (j=0), top (j=NY) - but exclude interface corners
        if i == 0 or j == 0 or j == NY:
            if not (i == NX and (j == 0 or j == NY)):  # exclude interface corners
                outer_nodes.append(nid)

# Build quad elements (counter-clockwise)
elements = []
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n00 = i + 1 + (NX + 1) * j
        n10 = i + 2 + (NX + 1) * j
        n11 = i + 2 + (NX + 1) * (j + 1)
        n01 = i + 1 + (NX + 1) * (j + 1)
        elements.append((elem_id, n00, n10, n11, n01))
        elem_id += 1

N_ELEMS = len(elements)

# Slab thickness for TSI (one element thick in z)
TZ = (X1 - X0) / NX  # same as dx for well-shaped hex

# Partner's interface data
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")

def partner_values(y_coord):
    """Get partner's (T, ux, uy) at given y coordinate via interpolation."""
    for partner_name, data in imp.items():
        coords = data.get("coordinates", [])
        values = data.get("values", [])
        if not coords or not values:
            continue
        values = np.asarray(values, float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.shape[1] < 3:
            continue
        ys = np.array([c[1] for c in coords], float)
        idx = np.argsort(ys)
        t = min(max(float(y_coord), ys[idx][0]), ys[idx][-1])
        return tuple(float(np.interp(t, ys[idx], values[idx, c])) for c in range(3))
    return (0.0, 0.0, 0.0)

# Output prefixes
OUT_T = f"run_T_level{LEVEL}"
OUT_U = f"run_U_level{LEVEL}"

# Write deck T: Scalar_Transport with CALCFLUX_BOUNDARY
deck_t = f'''TITLE:
  - "Subdomain A Heat Transfer Level {LEVEL}"
PROBLEM SIZE:
  ELEMENTS: {N_ELEMS}
  NODES: {N_NODES}
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_BOUNDARY: "diffusive"
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {KV}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DSURF-NODE TOPOLOGY:
'''
# Add all nodes to DSURFACE 1
for nid in range(1, N_NODES + 1):
    deck_t += f'  - "NODE {nid} DSURFACE 1"\n'

# Outer Dirichlet conditions (left, top, bottom edges, excluding interface corners)
deck_t += '''DESIGN POINT DIRICH CONDITIONS:
'''
dirich_id = 1
for nid in outer_nodes:
    deck_t += f'  - E: {dirich_id}\n'
    deck_t += f'    NUMDOF: 1\n'
    deck_t += f'    ONOFF: [1]\n'
    deck_t += f'    VAL: [0.0]\n'
    deck_t += f'    FUNCT: [0]\n'
    dirich_id += 1

deck_t += '''DNODE-NODE TOPOLOGY:
'''
for nid in outer_nodes:
    deck_t += f'  - "NODE {nid} DNODE {nid}"\n'

# Interface Dirichlet conditions (from partner, excluding corners)
deck_t += '''DESIGN POINT DIRICH CONDITIONS:
'''
for k, nid in enumerate(interior):
    _, y = nodes[nid - 1]
    T_val, _, _ = partner_values(y)
    deck_t += f'  - E: {dirich_id}\n'
    deck_t += f'    NUMDOF: 1\n'
    deck_t += f'    ONOFF: [1]\n'
    deck_t += f'    VAL: [{T_val:.15e}]\n'
    deck_t += f'    FUNCT: [0]\n'
    dirich_id += 1

deck_t += '''DNODE-NODE TOPOLOGY:
'''
for nid in interior:
    deck_t += f'  - "NODE {nid} DNODE {nid}"\n'

# Flux calc line condition on interface
deck_t += '''SCATRA FLUX CALC LINE CONDITIONS:
  - E: 2
DLINE-NODE TOPOLOGY:
'''
for nid in interface_node_ids:
    deck_t += f'  - "NODE {nid} DLINE 2"\n'

# Node coordinates
deck_t += '''NODE COORDS:
'''
for nid, (x, y) in enumerate(nodes, 1):
    deck_t += f'  - "NODE {nid} COORD {x:.15e} {y:.15e} 0.0"\n'

# Transport elements (QUAD4)
deck_t += '''TRANSPORT ELEMENTS:
'''
for eid, n00, n10, n11, n01 in elements:
    deck_t += f'  - "{eid} TRANSP QUAD4 {n00} {n10} {n11} {n01} MAT 1 TYPE Std"\n'

# IO for VTU output
deck_t += '''IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  FILE_TYPE: vtu
  OUTPUT_DATA_FORMAT: ascii
'''

Path(f"{OUT_T}.4C.yaml").write_text(deck_t)

# Write deck U: Thermo_Structure_Interaction with SOLIDSCATRA HEX8 slab
# Slab has two layers in z: z=0 and z=TZ
slab_nodes = []
slab_node_map = {}  # (x,y,z_layer) -> node_id
nid = 1
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * dx
        y = Y0 + j * dy
        # Layer 0 (z=0)
        slab_nodes.append((x, y, 0.0))
        slab_node_map[(x, y, 0)] = nid
        nid += 1
        # Layer 1 (z=TZ)
        slab_nodes.append((x, y, TZ))
        slab_node_map[(x, y, 1)] = nid
        nid += 1

SLAB_N_NODES = len(slab_nodes)

# Hex elements (two layers of quads make one layer of hexes)
slab_elements = []
elem_id = 1
for j in range(NY):
    for i in range(NX):
        # Bottom face nodes (z=0)
        n00_0 = slab_node_map[(X0 + i*dx, Y0 + j*dy, 0)]
        n10_0 = slab_node_map[(X0 + (i+1)*dx, Y0 + j*dy, 0)]
        n11_0 = slab_node_map[(X0 + (i+1)*dx, Y0 + (j+1)*dy, 0)]
        n01_0 = slab_node_map[(X0 + i*dx, Y0 + (j+1)*dy, 0)]
        # Top face nodes (z=TZ)
        n00_1 = slab_node_map[(X0 + i*dx, Y0 + j*dy, 1)]
        n10_1 = slab_node_map[(X0 + (i+1)*dx, Y0 + j*dy, 1)]
        n11_1 = slab_node_map[(X0 + (i+1)*dx, Y0 + (j+1)*dy, 1)]
        n01_1 = slab_node_map[(X0 + i*dx, Y0 + (j+1)*dy, 1)]
        # HEX8: bottom face (counter-clockwise) then top face (counter-clockwise)
        slab_elements.append((elem_id, n00_0, n10_0, n11_0, n01_0, n00_1, n10_1, n11_1, n01_1))
        elem_id += 1

SLAB_N_ELEMS = len(slab_elements)

# Interface nodes on both layers (excluding corners)
iface_nids_layer0 = [slab_node_map[(IFACE_X, Y0 + j*dy, 0)] for j in range(1, NY)]  # exclude corners
iface_nids_layer1 = [slab_node_map[(IFACE_X, Y0 + j*dy, 1)] for j in range(1, NY)]  # exclude corners

# Outer boundary nodes on both layers (left, top, bottom, excluding interface corners)
outer_slab_nids = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * dx
        y = Y0 + j * dy
        if i == 0 or j == 0 or j == NY:  # left, bottom, top
            if not (i == NX and (j == 0 or j == NY)):  # exclude interface corners
                outer_slab_nids.append(slab_node_map[(x, y, 0)])
                outer_slab_nids.append(slab_node_map[(x, y, 1)])

# u_z pinned on ALL nodes (both layers)
all_slab_nids = list(range(1, SLAB_N_NODES + 1))

deck_u = f'''TITLE:
  - "Subdomain A Thermo-Elastic Level {LEVEL}"
PROBLEM SIZE:
  ELEMENTS: {SLAB_N_ELEMS}
  NODES: {SLAB_N_NODES}
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  TIMEINTEGR: "Statics"
  SOLVERTYPE: "linear_full"
  LINEAR_SOLVER: 1
THERMAL DYNAMIC:
  TIMEINTEGR: "Statics"
  SOLVERTYPE: "linear_full"
  LINEAR_SOLVER: 1
COUPALGO: "tsi_oneway"
COUPVARIABLE: "Temperature"
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNGNUM: 1
      YOUNG: [{E_MOD:.15e}]
      NUE: {NU:.15e}
      DENS: 1.0
      THEXPANS: {ALPHA:.15e}
      INITTEMP: 0.0
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CONDUCTIVITY: {KV:.15e}
CLONING MATERIAL MAP:
  - STRUCTMAT: 1
    THERMMAT: 2
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DVOL-NODE TOPOLOGY:
'''
for nid in range(1, SLAB_N_NODES + 1):
    deck_u += f'  - "NODE {nid} DVOL 1"\n'

# Body force (structural Neumann) - use DESIGN VOL STRUCTURE NEUMANN
deck_u += '''DESIGN VOL STRUCTURE NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [1.0, 1.0, 1.0]
    FUNCT: [2, 3, 0]
DVOL-NODE TOPOLOGY:
'''
for nid in range(1, SLAB_N_NODES + 1):
    deck_u += f'  - "NODE {nid} DVOL 2"\n'

# Outer Dirichlet for displacement (u_x=u_y=0 on left, top, bottom)
deck_u += '''DESIGN POINT DIRICH CONDITIONS:
'''
dirich_id = 1
for nid in outer_slab_nids:
    deck_u += f'  - E: {dirich_id}\n'
    deck_u += f'    NUMDOF: 3\n'
    deck_u += f'    ONOFF: [1, 1, 1]\n'
    deck_u += f'    VAL: [0.0, 0.0, 0.0]\n'
    deck_u += f'    FUNCT: [0, 0, 0]\n'
    dirich_id += 1

deck_u += '''DNODE-NODE TOPOLOGY:
'''
for nid in outer_slab_nids:
    deck_u += f'  - "NODE {nid} DNODE {nid}"\n'

# u_z = 0 on ALL nodes (plane strain constraint)
deck_u += '''DESIGN POINT DIRICH CONDITIONS:
'''
for nid in all_slab_nids:
    deck_u += f'  - E: {dirich_id}\n'
    deck_u += f'    NUMDOF: 3\n'
    deck_u += f'    ONOFF: [0, 0, 1]\n'
    deck_u += f'    VAL: [0.0, 0.0, 0.0]\n'
    deck_u += f'    FUNCT: [0, 0, 0]\n'
    dirich_id += 1

deck_u += '''DNODE-NODE TOPOLOGY:
'''
for nid in all_slab_nids:
    deck_u += f'  - "NODE {nid} DNODE {nid}"\n'

# Interface Dirichlet for displacement (from partner, on both layers, excluding corners)
deck_u += '''DESIGN POINT DIRICH CONDITIONS:
'''
for k, (nid0, nid1) in enumerate(zip(iface_nids_layer0, iface_nids_layer1)):
    _, y, _ = slab_nodes[nid0 - 1]
    _, ux_val, uy_val = partner_values(y)
    # Layer 0
    deck_u += f'  - E: {dirich_id}\n'
    deck_u += f'    NUMDOF: 3\n'
    deck_u += f'    ONOFF: [1, 1, 0]\n'
    deck_u += f'    VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]\n'
    deck_u += f'    FUNCT: [0, 0, 0]\n'
    deck_u += f'    TAG: monitor_reaction\n'
    dirich_id += 1
    # Layer 1
    deck_u += f'  - E: {dirich_id}\n'
    deck_u += f'    NUMDOF: 3\n'
    deck_u += f'    ONOFF: [1, 1, 0]\n'
    deck_u += f'    VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]\n'
    deck_u += f'    FUNCT: [0, 0, 0]\n'
    deck_u += f'    TAG: monitor_reaction\n'
    dirich_id += 1

deck_u += '''DNODE-NODE TOPOLOGY:
'''
for nid in iface_nids_layer0 + iface_nids_layer1:
    deck_u += f'  - "NODE {nid} DNODE {nid}"\n'

# Temperature Dirichlet on outer boundary (T=0)
deck_u += '''DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DSURF-NODE TOPOLOGY:
'''
# Outer surface nodes (left, top, bottom faces of slab)
outer_surf_nids = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * dx
        y = Y0 + j * dy
        if i == 0 or j == 0 or j == NY:
            if not (i == NX and (j == 0 or j == NY)):
                outer_surf_nids.append(slab_node_map[(x, y, 0)])
                outer_surf_nids.append(slab_node_map[(x, y, 1)])

for nid in outer_surf_nids:
    deck_u += f'  - "NODE {nid} DSURFACE 3"\n'

# Temperature Dirichlet on interface (from partner)
deck_u += '''DESIGN POINT THERMO DIRICH CONDITIONS:
'''
for k, (nid0, nid1) in enumerate(zip(iface_nids_layer0, iface_nids_layer1)):
    _, y, _ = slab_nodes[nid0 - 1]
    T_val, _, _ = partner_values(y)
    deck_u += f'  - E: {dirich_id}\n'
    deck_u += f'    NUMDOF: 1\n'
    deck_u += f'    ONOFF: [1]\n'
    deck_u += f'    VAL: [{T_val:.15e}]\n'
    deck_u += f'    FUNCT: [0]\n'
    dirich_id += 1
    deck_u += f'  - E: {dirich_id}\n'
    deck_u += f'    NUMDOF: 1\n'
    deck_u += f'    ONOFF: [1]\n'
    deck_u += f'    VAL: [{T_val:.15e}]\n'
    deck_u += f'    FUNCT: [0]\n'
    dirich_id += 1

deck_u += '''DNODE-NODE TOPOLOGY:
'''
for nid in iface_nids_layer0 + iface_nids_layer1:
    deck_u += f'  - "NODE {nid} DNODE {nid}"\n'

# Node coordinates (slab)
deck_u += '''NODE COORDS:
'''
for nid, (x, y, z) in enumerate(slab_nodes, 1):
    deck_u += f'  - "NODE {nid} COORD {x:.15e} {y:.15e} {z:.15e}"\n'

# Structure elements (SOLIDSCATRA HEX8)
deck_u += '''STRUCTURE ELEMENTS:
'''
for eid, *node_ids in slab_elements:
    deck_u += f'  - "{eid} SOLIDSCATRA HEX8 {" ".join(map(str, node_ids))} MAT 1 KINEM linear EAS none THICK 1.0 STRESS_STRAIN plane_strain GP 2 2"\n'

# Thermal elements (same topology)
deck_u += '''THERMAL ELEMENTS:
'''
for eid, *node_ids in slab_elements:
    deck_u += f'  - "{eid} SOLIDSCATRA HEX8 {" ".join(map(str, node_ids))} MAT 2"\n'

# IO sections
deck_u += '''IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  FILE_TYPE: vtu
  OUTPUT_DATA_FORMAT: ascii
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
'''

DECK_U = f"{OUT_U}.4C.yaml"
Path(DECK_U).write_text(deck_u)

# Run deck T
print(f"Running 4C heat transfer deck (level {LEVEL})...", flush=True)
cmd_t = ["stdbuf", "-oL", "-eL", FOURC_BIN, f"{OUT_T}.4C.yaml", OUT_T]
result_t = subprocess.run(cmd_t, capture_output=True, text=True, timeout=300)
log_t = result_t.stdout + result_t.stderr
Path(f"run_T_level{LEVEL}.log").write_text(log_t)
if result_t.returncode != 0:
    print(f"4C run T failed with code {result_t.returncode}", flush=True)
    print(log_t, flush=True)
    raise SystemExit(f"4C run T failed: {log_t[:500]}")

# Run deck U
print(f"Running 4C thermo-elastic deck (level {LEVEL})...", flush=True)
cmd_u = ["stdbuf", "-oL", "-eL", FOURC_BIN, DECK_U, OUT_U]
result_u = subprocess.run(cmd_u, capture_output=True, text=True, timeout=300)
log_u = result_u.stdout + result_u.stderr
Path(f"run_U_level{LEVEL}.log").write_text(log_u)
if result_u.returncode != 0:
    print(f"4C run U failed with code {result_u.returncode}", flush=True)
    print(log_u, flush=True)
    raise SystemExit(f"4C run U failed: {log_u[:500]}")

# Combine logs for the participant log file
combined_log = log_t + "\n" + log_u
ndof = 3 * N_NODES  # T + ux + uy per 2D node
Path(f"run_level{LEVEL}_A.log").write_text(f"NDOF = {ndof}\n\n" + combined_log)

# Read results from VTU files
def latest_vtu(pattern):
    vs = sorted(glob.glob(pattern))
    if not vs:
        return None
    def step(p):
        m = re.match(r".*-(\d+)-\d+\.vtu$", p)
        return int(m.group(1)) if m else -1
    return max(vs, key=step)

# Read temperature from scatra VTU
vtu_T = latest_vtu(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None:
    raise SystemExit(f"No scatra VTU found for run T")
mT = meshio.read(vtu_T)
pts_T = np.asarray(mT.points)
phi_1 = np.asarray(mT.point_data["phi_1"])

# Collapse by coordinate (VTU has one point per element corner)
key_T = {}
for r in range(len(pts_T)):
    key = (round(float(pts_T[r, 0]), 9), round(float(pts_T[r, 1]), 9))
    if key not in key_T:
        key_T[key] = []
    key_T[key].append(phi_1[r])

T2d = np.zeros(N_NODES)
for i, (x, y) in enumerate(nodes):
    key = (round(float(x), 9), round(float(y), 9))
    if key in key_T:
        T2d[i] = np.mean(key_T[key])

# Read flux from scatra VTU
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("No flux_boundary field in scatra VTU")
FB = np.asarray(mT.point_data[fbname])

key_FB = {}
for r in range(len(pts_T)):
    key = (round(float(pts_T[r, 0]), 9), round(float(pts_T[r, 1]), 9))
    if key not in key_FB:
        key_FB[key] = []
    key_FB[key].append(FB[r])

FB2d = np.zeros((N_NODES, 2))
for i, (x, y) in enumerate(nodes):
    key = (round(float(x), 9), round(float(y), 9))
    if key in key_FB:
        FB2d[i] = np.mean(key_FB[key], axis=0)

# Outward normal at interface (right edge, pointing +x)
nrm = (1.0, 0.0)
q_n = [float(FB2d[n - 1, 0] * nrm[0] + FB2d[n - 1, 1] * nrm[1]) for n in interior]

# Read displacement from structure VTU
vtu_U = latest_vtu(f"{OUT_U}-vtk-files/structure-*.vtu")
if vtu_U is None:
    raise SystemExit(f"No structure VTU found for run U")
mU = meshio.read(vtu_U)
pts_U = np.asarray(mU.points)
disp = np.asarray(mU.point_data["displacement"])

# Get only layer 0 (z=0)
layer0_mask = np.abs(pts_U[:, 2]) < 1e-9
pts_U0 = pts_U[layer0_mask]
disp_U0 = disp[layer0_mask]

key_U = {}
for r in range(len(pts_U0)):
    key = (round(float(pts_U0[r, 0]), 9), round(float(pts_U0[r, 1]), 9))
    if key not in key_U:
        key_U[key] = []
    key_U[key].append(disp_U0[r])

U2d = np.zeros((N_NODES, 2))
for i, (x, y) in enumerate(nodes):
    key = (round(float(x), 9), round(float(y), 9))
    if key in key_U:
        U2d[i] = np.mean(key_U[key], axis=0)

# Read reactions from monitor yaml files
_gid_xy = {}
for ln in deck_u.splitlines():
    m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', ln)
    if m:
        _gid_xy[int(m.group(1))] = (round(float(m.group(2)), 9), round(float(m.group(3)), 9), round(float(m.group(4)), 9))

_F = {}
for yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    txt = Path(yf).read_text(errors="ignore")
    gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", txt.split("dbc monitor condition data", 1)[0], re.M)]
    fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt, re.M)
    if not (gids and fs):
        continue
    fx, fy = float(fs[-1][0]), float(fs[-1][1])
    xy = _gid_xy.get(gids[0] + 1)  # yaml gids are 0-based, deck nodes are 1-based
    if xy is None:
        continue
    xy2d = (xy[0], xy[1])
    _acc = _F.setdefault(xy2d, [0.0, 0.0])
    _acc[0] += fx
    _acc[1] += fy

_h = dy  # tributary length along interface
_share = _h * TZ

q_t = []
for n in interior:
    xy = (round(float(nodes[n - 1][0]), 9), round(float(nodes[n - 1][1]), 9))
    if xy not in _F:
        raise SystemExit(f"No reaction file for interface node at {xy}")
    q_t.append([_F[xy][0] / _share, _F[xy][1] / _share])

# Build export data
co = [[float(nodes[n - 1][0]), float(nodes[n - 1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# Export self-check
_chk = np.asarray(Q, float)
if not np.isfinite(_chk).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface fluxes")

# Write exports.json LAST
json.dump({
    "field_name": "thermoelastic",
    "coordinates": co,
    "values": [],  # Dirichlet side doesn't own values
    "normal_fluxes": Q,
    "n_points": len(co)
}, open("exports.json", "w"), indent=2)

# Write per-level field file
with open(f"field_level{LEVEL}_A.csv", "w") as f:
    f.write("x,y,T,ux,uy\n")
    for (px, py), t, (ux, uy) in zip(nodes, T2d, U2d):
        f.write(f"{px:.11e},{py:.11e},{t:.11e},{ux:.11e},{uy:.11e}\n")

# Write per-level interface file (OUTWARD traction = -exported traction)
with open(f"interface_level{LEVEL}_A.csv", "w") as f:
    f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for (px, py), n, (qn, qx, qy) in zip(co, interior, Q):
        f.write(f"{px:.11e},{py:.11e},{T2d[n-1]:.11e},{U2d[n-1,0]:.11e},{U2d[n-1,1]:.11e},{qn:.11e},{-qx:.11e},{-qy:.11e}\n")

print(f"NDOF = {ndof}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {ndof} T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]")
