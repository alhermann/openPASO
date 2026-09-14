#!/usr/bin/env python3
"""4C participant script for Side A (Dirichlet side) of coupled heat problem."""
import json
import numpy as np
from pathlib import Path
import glob
import meshio
import subprocess
import os
import sys

# Read configuration
CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
LEVEL = CFG.get("level", 1)

# Subdomain A geometry
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
K = 1.0

IFACE = "right"

SRC_EXPR = "-6*x^3*y + 16*x^3/5 - 3379*x^2*y/800 + 18979*x^2/1500 - 6*x*y^3 + 48*x*y^2/5 + 287*x*y/200 - 11077*x/1200 - 3379*y^3/2400 + 18979*y^2/1500 - 44979*y/4000"

FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"

Hx = (X1 - X0) / NX
Hy = (Y1 - Y0) / NY

nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * Hx
        y = Y0 + j * Hy
        nodes.append((x, y))

N_NODES = len(nodes)

elements = []
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n_bl = i + 1 + j * (NX + 1)
        n_br = i + 2 + j * (NX + 1)
        n_tr = i + 2 + (j + 1) * (NX + 1)
        n_tl = i + 1 + (j + 1) * (NX + 1)
        elements.append(f"{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std")
        elem_id += 1

N_ELEMS = len(elements)

interface_nodes = []
for j in range(NY + 1):
    node_id = NX + 1 + j * (NX + 1)
    interface_nodes.append(node_id)

interior_interface_nodes = [node_id for node_id in interface_nodes if node_id not in [NX + 1, N_NODES]]

dline_topology = []
dline_id = 1

left_nodes = [1 + j * (NX + 1) for j in range(NY + 1)]
for nid in left_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
left_dline_id = dline_id
dline_id += 1

bottom_nodes = [i + 1 for i in range(1, NX)]
for nid in bottom_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
bottom_dline_id = dline_id
dline_id += 1

top_nodes = [i + 1 + NY * (NX + 1) for i in range(1, NX)]
for nid in top_nodes:
    dline_topology.append(f"NODE {nid} DLINE {dline_id}")
top_dline_id = dline_id
dline_id += 1

dsurf_topology = [f"NODE {nid} DSURFACE 1" for nid in range(1, N_NODES + 1)]

imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())

def get_partner_value(y_coord):
    for partner_name, data in imp.items():
        coords = data.get("coordinates", [])
        values = data.get("values", [])
        if coords and values:
            pts = sorted(zip([c[1] for c in coords], values))
            ys = [p[0] for p in pts]
            vs = [float(p[1]) for p in pts]
            t = max(min(y_coord, ys[-1]), ys[0])
            for a, b, va, vb in zip(ys, ys[1:], vs, vs[1:]):
                if a <= t <= b:
                    w = 0.0 if b == a else (t - a) / (b - a)
                    return va + w * (vb - va)
            return vs[-1]
    return 0.0

point_dirich_conditions = []
dnode_topology = []
dnode_id = 1

for node_id in interior_interface_nodes:
    x, y = nodes[node_id - 1]
    val = get_partner_value(y)
    point_dirich_conditions.append(f"- E: {dnode_id}\n      NUMDOF: 1\n      ONOFF: [1]\n      VAL: [{val}]\n      FUNCT: [0]")
    dnode_topology.append(f"NODE {node_id} DNODE {dnode_id}")
    dnode_id += 1

interface_line_id = dline_id

deck_lines = [
    "TITLE:",
    '  - "Side A - Dirichlet side"',
    "",
    "PROBLEM SIZE:",
    f"  ELEMENTS: {N_ELEMS}",
    f"  NODES: {N_NODES}",
    "",
    "PROBLEM TYPE:",
    '  PROBLEMTYPE: "Scalar_Transport"',
    "",
    "SCALAR TRANSPORT DYNAMIC:",
    '  TIMEINTEGR: "Stationary"',
    '  SOLVERTYPE: "linear_full"',
    "  NUMSTEP: 1",
    "  TIMESTEP: 1.0",
    "  MAXTIME: 1.0",
    "  LINEAR_SOLVER: 1",
    '  CALCFLUX_BOUNDARY: "diffusive"',
    "",
    "SOLVER 1:",
    '  SOLVER: "UMFPACK"',
    "",
    "MATERIALS:",
    "  - MAT: 1",
    "    MAT_scatra:",
    f"      DIFFUSIVITY: {K}",
    "",
    "FUNCT1:",
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"',
    "",
    "DESIGN LINE DIRICH CONDITIONS:",
    f"  - E: {left_dline_id}",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    f"  - E: {bottom_dline_id}",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    f"  - E: {top_dline_id}",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    "",
    "DESIGN POINT DIRICH CONDITIONS:",
] + point_dirich_conditions + [
    "",
    "DESIGN SURF TRANSPORT NEUMANN CONDITIONS:",
    "  - E: 1",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [1.0]",
    "    FUNCT: [1]",
    "",
    "NODE COORDS:",
] + [f'  - "NODE {nid} COORD {x:.10e} {y:.10e} 0.0"' for nid, (x, y) in enumerate(nodes, 1)] + [
    "",
    "TRANSPORT ELEMENTS:",
] + [f'  - "{e}"' for e in elements] + [
    "",
    "DLINE-NODE TOPOLOGY:",
] + [f'  - "{t}"' for t in dline_topology] + [f'  - "NODE {nid} DLINE {interface_line_id}"' for nid in interface_nodes] + [
    "",
    "DNODE-NODE TOPOLOGY:",
] + [f'  - "{t}"' for t in dnode_topology] + [
    "",
    "DSURF-NODE TOPOLOGY:",
] + [f'  - "{t}"' for t in dsurf_topology] + [
    "",
    "SCATRA FLUX CALC LINE CONDITIONS:",
    f"  - E: {interface_line_id}",
]

DECK = "side_A.4C.yaml"
with open(DECK, "w") as f:
    f.write("\n".join(deck_lines))

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{FOURC_LD}:{env.get('LD_LIBRARY_PATH', '')}"

cmd = ["stdbuf", "-oL", "-eL", FOURC_BIN, DECK, "out"]
result = subprocess.run(cmd, capture_output=True, text=True, env=env)

vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not vtus:
    print("ERROR: No VTU files found", file=sys.stderr)
    print("STDOUT:", result.stdout, file=sys.stderr)
    print("STDERR:", result.stderr, file=sys.stderr)
    sys.exit(1)

vtu = vtus[-1]
mesh = meshio.read(vtu)

vpts = [(float(p[0]), float(p[1])) for p in mesh.points]
phi_vals = [float(x) for x in mesh.point_data["phi_1"].ravel()]

val_map = {}
for (x, y), u in zip(vpts, phi_vals):
    key = (round(x, 12), round(y, 12))
    val_map[key] = u

u_all = [val_map[(round(nodes[n][0], 12), round(nodes[n][1], 12))] for n in range(N_NODES)]

fbname = next((da for da in mesh.point_data if "flux_boundary" in da), None)
if fbname is None:
    print("ERROR: No flux_boundary field in VTU", file=sys.stderr)
    sys.exit(1)

fb = mesh.point_data[fbname]
nrm = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1), "top": (0, 1)}[IFACE]

fb_map = {}
for (x, y), vec in zip(vpts, fb):
    key = (round(x, 10), round(y, 10))
    flux_normal = float(vec[0] * nrm[0] + vec[1] * nrm[1])
    fb_map[key] = flux_normal

export_coords = [list(nodes[nid - 1]) for nid in interior_interface_nodes]
export_values = [u_all[nid - 1] for nid in interior_interface_nodes]
export_fluxes = [fb_map[(round(nodes[nid - 1][0], 10), round(nodes[nid - 1][1], 10))] for nid in interior_interface_nodes]

exports = {
    "field_name": "u",
    "n_points": len(interior_interface_nodes),
    "coordinates": export_coords,
    "values": [],
    "normal_fluxes": export_fluxes
}
with open("exports.json", "w") as f:
    json.dump(exports, f)

with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(nodes, u_all):
        f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for coord, u_val, q_val in zip(export_coords, export_values, export_fluxes):
        f.write(f"{coord[0]:.11e},{coord[1]:.11e},{u_val:.11e},{q_val:.11e}\n")

print(f"NDOF = {N_NODES}")
print(f"Side A (Dirichlet): max|u| = {max(abs(u) for u in u_all):.6e}")
