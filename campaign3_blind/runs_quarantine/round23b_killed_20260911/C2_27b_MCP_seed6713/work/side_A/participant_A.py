#!/usr/bin/env python3
"""4C participant for subdomain A - DIRICHLET side of coupling."""
import json
import subprocess
import glob
import sys
import os
from pathlib import Path
import numpy as np
import meshio

# Configuration
FOURC_BIN = "/home/alexander/4C/build/4C"
os.environ["LD_LIBRARY_PATH"] = "/opt/4C-dependencies/lib"

# Read config
cfg_path = Path("config.json")
if cfg_path.is_file():
    CFG = json.loads(cfg_path.read_text())
else:
    CFG = {"level": 1, "nx": 5, "ny": 8}

LEVEL = CFG.get("level", 1)
NX = CFG.get("nx", 5)
NY = CFG.get("ny", 8)

# Subdomain A geometry
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
K = 1.0

# Source term in 4C format
SRC_EXPR = "-6*x^3*y + 16*x^3/5 - 3379*x^2*y/800 + 18979*x^2/1500 - 6*x*y^3 + 48*x*y^2/5 + 287*x*y/200 - 11077*x/1200 - 3379*y^3/2400 + 18979*y^2/1500 - 44979*y/4000"

# Read imports
imp_path = Path("imports.json")
imports = json.loads(imp_path.read_text() or "{}") if imp_path.is_file() else {}
PARTNER = "B"

def get_partner_values(y_coords):
    if PARTNER not in imports or "values" not in imports[PARTNER]:
        return np.zeros(len(y_coords))
    d = imports[PARTNER]
    coords = np.array(d["coordinates"])
    vals = np.array(d["values"]).ravel()
    if len(coords) == 0:
        return np.zeros(len(y_coords))
    src_y = coords[:, 1]
    idx = np.argsort(src_y)
    return np.interp(y_coords, src_y[idx], vals[idx])

# Build mesh
hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY

nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * hx
        y = Y0 + j * hy
        nodes.append((len(nodes)+1, x, y))

node_coords = [f"NODE {nid} COORD {x:.10f} {y:.10f} 0.0" for nid, x, y in nodes]

# Elements
elements = []
for j in range(NY):
    for i in range(NX):
        n_bl = i + j * (NX + 1) + 1
        n_br = i + 1 + j * (NX + 1) + 1
        n_tr = i + 1 + (j + 1) * (NX + 1) + 1
        n_tl = i + (j + 1) * (NX + 1) + 1
        elements.append(f"{len(elements)+1} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std")

# Boundary identification using strict inequalities for corners
left_nodes = [nid for nid, x, y in nodes if abs(x - X0) < 1e-9]
right_nodes = [nid for nid, x, y in nodes if abs(x - X1) < 1e-9]
bottom_nodes = [nid for nid, x, y in nodes if abs(y - Y0) < 1e-9]
top_nodes = [nid for nid, x, y in nodes if abs(y - Y1) < 1e-9]

# Interface interior nodes (exclude corners at y=0 and y=1)
interface_nodes = [nid for nid, x, y in nodes if abs(x - X1) < 1e-9 and y > 1e-6 and y < 1.0 - 1e-6]
interface_ys = [y for nid, x, y in nodes if abs(x - X1) < 1e-9 and y > 1e-6 and y < 1.0 - 1e-6]

partner_T = get_partner_values(interface_ys)

# Build deck
dirichlet_lines = []
dline_id = 1

# Left edge
if left_nodes:
    dirichlet_lines.append(f"  - E: {dline_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]")
    dline_id += 1

# Bottom inner (exclude corners)
bottom_inner = [nid for nid in bottom_nodes if nid not in left_nodes and nid not in right_nodes]
if bottom_inner:
    dirichlet_lines.append(f"  - E: {dline_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]")
    dline_id += 1

# Top inner (exclude corners)
top_inner = [nid for nid in top_nodes if nid not in left_nodes and nid not in right_nodes]
if top_inner:
    dirichlet_lines.append(f"  - E: {dline_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]")
    dline_id += 1

# Point Dirichlet on interface
point_dirichlet = []
dnode_id = dline_id
for idx, (nid, T_val) in enumerate(zip(interface_nodes, partner_T)):
    point_dirichlet.append(f"  - E: {dnode_id + idx}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{T_val:.15e}]\n    FUNCT: [0]")

# Topology
dline_topo = [f"NODE {nid} DLINE 1" for nid in left_nodes]
if bottom_inner:
    dline_topo.extend([f"NODE {nid} DLINE 2" for nid in bottom_inner])
if top_inner:
    dline_topo.extend([f"NODE {nid} DLINE 3" for nid in top_inner])

dnode_topo = [f"NODE {nid} DNODE {dnode_id + idx}" for idx, nid in enumerate(interface_nodes)]
surf_topo = [f"NODE {nid} DSURFACE 1" for nid, x, y in nodes]

deck = f'''TITLE:
  - "Subdomain A - Dirichlet side"
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
  CALCFLUX_BOUNDARY: "diffusive"
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {K}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"
DESIGN LINE DIRICH CONDITIONS:
{chr(10).join(dirichlet_lines)}
DESIGN POINT DIRICH CONDITIONS:
{chr(10).join(point_dirichlet)}
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DLINE-NODE TOPOLOGY:
{chr(10).join(["  - " + t for t in dline_topo])}
DNODE-NODE TOPOLOGY:
{chr(10).join(["  - " + t for t in dnode_topo])}
DSURF-NODE TOPOLOGY:
{chr(10).join(["  - " + t for t in surf_topo])}
NODE COORDS:
{chr(10).join(["  - \"" + nc + "\"" for nc in node_coords])}
TRANSPORT ELEMENTS:
{chr(10).join(["  - \"" + e + "\"" for e in elements])}
SCATRA FLUX CALC LINE CONDITIONS:
  - E: 1
'''

Path("subdomain_A.4C.yaml").write_text(deck)

# Run 4C
env = dict(os.environ)
result = subprocess.run(["stdbuf", "-oL", "-eL", FOURC_BIN, "subdomain_A.4C.yaml", "out"], 
                       capture_output=True, text=True, env=env)

print(result.stdout)
if result.stderr:
    print(result.stderr)

if result.returncode != 0:
    print(f"4C failed: {result.returncode}", file=sys.stderr)
    sys.exit(1)

# Read VTU
vtu_files = sorted(glob.glob("out-vtk-files/*.vtu"))
if not vtu_files:
    print("No VTU files!", file=sys.stderr)
    sys.exit(1)

mesh = meshio.read(vtu_files[-1])
pts = mesh.points
phi = mesh.point_data["phi_1"]

# pts is (N, 3) for 3D coordinates, use first two
val_map = {}
for pt, u in zip(pts, phi):
    x, y = pt[0], pt[1]
    val_map[(round(x, 10), round(y, 10))] = u

node_vals = {nid: val_map[(round(x, 10), round(y, 10))] for nid, x, y in nodes}

# Flux
fbname = next((da for da in mesh.point_data if "flux_boundary" in da), None)
if fbname is None:
    print("No flux_boundary!", file=sys.stderr)
    sys.exit(1)

fb = mesh.point_data[fbname]
fb_map = {}
for pt, vec in zip(pts, fb):
    x, y = pt[0], pt[1]
    fb_map[(round(x, 10), round(y, 10))] = float(vec[0])

# Export
export_coords = []
export_values = []
export_fluxes = []

for nid in interface_nodes:
    x, y = nodes[nid-1][1], nodes[nid-1][2]
    export_coords.append([x, y])
    export_values.append(node_vals[nid])
    qn = fb_map.get((round(x, 10), round(y, 10)), 0.0)
    export_fluxes.append(qn)

json.dump({"field_name": "temperature", "n_points": len(export_coords),
           "coordinates": export_coords, "values": export_values, "normal_fluxes": export_fluxes},
          open("exports.json", "w"), indent=2)

with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u\n")
    for nid, x, y in nodes:
        f.write(f"{x:.11e},{y:.11e},{node_vals[nid]:.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for i, nid in enumerate(interface_nodes):
        x, y = nodes[nid-1][1], nodes[nid-1][2]
        f.write(f"{x:.11e},{y:.11e},{node_vals[nid]:.11e},{export_fluxes[i]:.11e}\n")

print(f"NDOF = {len(nodes)}")
