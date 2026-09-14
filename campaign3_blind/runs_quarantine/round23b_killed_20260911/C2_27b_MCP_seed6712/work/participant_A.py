#!/usr/bin/env python3
"""4C as DIRICHLET side of a partitioned coupling (Scalar_Transport).

Subdomain A geometry: (0, 0.625) x (0, 1), conductivity k=1
Interface is at x=0.625 (right edge of subdomain A)
Outer boundary u=0 at x=0, y=0, y=1

Reads ./config.json {"level":k,"nx":..,"ny":..}
Reads ./imports.json for partner's interface temperature values
Generates a 4C YAML deck for Scalar_Transport problem
Applies imported temperature as Dirichlet BC on interior interface nodes (x=0.625)
Applies u=0 on outer boundary (x=0, y=0, y=1)
Includes the source term f(x,y) as a volume source
Runs the 4C binary
Reads solution from VTU file
Recovers consistent outward normal flux
Writes exports.json with field_name, coordinates, values=[], normal_fluxes
Writes field_level{k}_A.csv and interface_level{k}_A.csv
Prints "NDOF = <count>"
"""
import json
import numpy as np
from pathlib import Path
import glob
import meshio
import subprocess
import os
import sys
import atexit

# ============================================================================
# CONFIGURATION
# ============================================================================
CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
LEVEL = CFG.get("level", 1)

# Subdomain A geometry
X0, X1 = 0.0, 0.625
Y0, Y1 = 0.0, 1.0
KV = 1.0  # conductivity

# Interface is at right edge (x = 0.625)
IFACE_X = 0.625
IFACE = "right"

# 4C binary path
FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"

# Source term in 4C syntax (^ for powers, lowercase pi)
SRC_EXPR = "-6*x^3*y + 16*x^3/5 - 3379*x^2*y/800 + 18979*x^2/1500 - 6*x*y^3 + 48*x*y^2/5 + 287*x*y/200 - 11077*x/1200 - 3379*y^3/2400 + 18979*y^2/1500 - 44979*y/4000"

# ============================================================================
# PARTNER INTERFACE DATA (handshake)
# ============================================================================
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())

def partner_value(y):
    """DIRICHLET side: the partner's field value at one of your interface points."""
    for _n, d in imp.items():
        co = d.get("coordinates") or []
        vals = d.get("values") or []
        if co and vals and len(vals) == len(co):
            pts = sorted(zip([c[0] for c in co], [c[1] for c in co], vals))
            ys = [p[1] for p in pts]
            vs = [float(p[2]) for p in pts]
            t = min(max(y, ys[0]), ys[-1])
            for a, b, va, vb in zip(ys, ys[1:], vs, vs[1:]):
                if a <= t <= b:
                    w = 0.0 if b == a else (t - a) / (b - a)
                    return va + w * (vb - va)
            return vs[-1]
    return 0.0

# ============================================================================
# MESH GENERATION
# ============================================================================
hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY

# Generate node coordinates (1-based indexing for 4C)
nodes = []  # nodes[gid-1] = (x, y)
node_id = 1
for j in range(NY + 1):
    y = Y0 + j * hy
    for i in range(NX + 1):
        x = X0 + i * hx
        nodes.append((x, y))
        node_id += 1

N_NODES = len(nodes)

# Identify interface nodes (at x = IFACE_X, excluding corners which are on outer boundary)
interface_nodes = []  # list of 4C node ids (1-based)
for gid, (x, y) in enumerate(nodes, start=1):
    if abs(x - IFACE_X) < 1e-10:
        # Exclude corners (y=0 and y=1 are outer boundary)
        if abs(y - Y0) > 1e-10 and abs(y - Y1) > 1e-10:
            interface_nodes.append(gid)

# Interior interface nodes (exclude endpoints that lie on outer boundary)
interior = interface_nodes  # Already excludes corners

# ============================================================================
# BUILD 4C DECK
# ============================================================================
deck_lines = []

# TITLE
deck_lines.append('TITLE:')
deck_lines.append(f'  - "Side A (4C) - Dirichlet side - Level {LEVEL}"')

# PROBLEM SIZE
deck_lines.append('PROBLEM SIZE:')
deck_lines.append(f'  DIM: 2')
deck_lines.append(f'  ELEMENTS: {NX * NY}')
deck_lines.append(f'  NODES: {N_NODES}')

# PROBLEM TYPE
deck_lines.append('PROBLEM TYPE:')
deck_lines.append('  PROBLEMTYPE: "Scalar_Transport"')

# SCALAR TRANSPORT DYNAMIC
deck_lines.append('SCALAR TRANSPORT DYNAMIC:')
deck_lines.append('  TIMEINTEGR: "Stationary"')
deck_lines.append('  SOLVERTYPE: "linear_full"')
deck_lines.append('  VELOCITYFIELD: "zero"')
deck_lines.append('  TIMESTEP: 1.0')
deck_lines.append('  NUMSTEP: 1')
deck_lines.append('  MAXTIME: 1.0')
deck_lines.append('  LINEAR_SOLVER: 1')
deck_lines.append('  CALCFLUX_BOUNDARY: "diffusive"')

# SOLVER 1
deck_lines.append('SOLVER 1:')
deck_lines.append('  SOLVER: "UMFPACK"')
deck_lines.append('  NAME: "direct"')

# MATERIALS
deck_lines.append('MATERIALS:')
deck_lines.append(f'  - MAT: 1')
deck_lines.append(f'    MAT_scatra:')
deck_lines.append(f'      DIFFUSIVITY: {KV}')

# FUNCT1 for source term
deck_lines.append('FUNCT1:')
deck_lines.append(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"')

# DESIGN LINE DIRICH CONDITIONS - outer boundary (u=0)
# Left edge (x=0), bottom edge (y=0), top edge (y=1)
deck_lines.append('DESIGN LINE DIRICH CONDITIONS:')

# Left edge (x=0) - DLINE 1
deck_lines.append('  - E: 1')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')

# Bottom edge (y=0) - DLINE 2
deck_lines.append('  - E: 2')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')

# Top edge (y=1) - DLINE 3
deck_lines.append('  - E: 3')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')

# DESIGN POINT DIRICH CONDITIONS - interface (imported values)
# One per interior interface node
deck_lines.append('DESIGN POINT DIRICH CONDITIONS:')
dnode_id = 1
for nid in interior:
    x, y = nodes[nid - 1]
    val = partner_value(y)
    deck_lines.append(f'  - E: {dnode_id}')
    deck_lines.append('    NUMDOF: 1')
    deck_lines.append('    ONOFF: [1]')
    deck_lines.append(f'    VAL: [{val}]')
    deck_lines.append('    FUNCT: [0]')
    dnode_id += 1

# NODE COORDS
deck_lines.append('NODE COORDS:')
for gid, (x, y) in enumerate(nodes, start=1):
    deck_lines.append(f'  - "NODE {gid} COORD {x:.10f} {y:.10f} 0.0"')

# TRANSPORT ELEMENTS
deck_lines.append('TRANSPORT ELEMENTS:')
elem_id = 1
for j in range(NY):
    for i in range(NX):
        # Quad element nodes (counter-clockwise from bottom-left)
        n_bl = i + 1 + j * (NX + 1)
        n_br = i + 2 + j * (NX + 1)
        n_tr = i + 2 + (j + 1) * (NX + 1)
        n_tl = i + 1 + (j + 1) * (NX + 1)
        deck_lines.append(f'  - "{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
        elem_id += 1

# DLINE-NODE TOPOLOGY
deck_lines.append('DLINE-NODE TOPOLOGY:')

# DLINE 1: left edge (x=0)
for j in range(NY + 1):
    nid = 1 + j * (NX + 1)
    deck_lines.append(f'  - "NODE {nid} DLINE 1"')

# DLINE 2: bottom edge (y=0)
for i in range(NX + 1):
    nid = i + 1
    deck_lines.append(f'  - "NODE {nid} DLINE 2"')

# DLINE 3: top edge (y=1)
for i in range(NX + 1):
    nid = i + 1 + NY * (NX + 1)
    deck_lines.append(f'  - "NODE {nid} DLINE 3"')

# DLINE 4: interface (x=0.625) - for flux calculation
for j in range(NY + 1):
    nid = NX + 1 + j * (NX + 1)
    deck_lines.append(f'  - "NODE {nid} DLINE 4"')

# DNODE-NODE TOPOLOGY (interface nodes)
deck_lines.append('DNODE-NODE TOPOLOGY:')
for idx, nid in enumerate(interior, start=1):
    deck_lines.append(f'  - "NODE {nid} DNODE {idx}"')

# DSURF-NODE TOPOLOGY (all nodes for volume source)
deck_lines.append('DSURF-NODE TOPOLOGY:')
for gid in range(1, N_NODES + 1):
    deck_lines.append(f'  - "NODE {gid} DSURFACE 1"')

# SCATRA FLUX CALC LINE CONDITIONS (for flux recovery on interface)
deck_lines.append('SCATRA FLUX CALC LINE CONDITIONS:')
deck_lines.append('  - E: 4')

# DESIGN SURF TRANSPORT NEUMANN CONDITIONS (volume source)
deck_lines.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
deck_lines.append('  - E: 1')
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [1.0]')
deck_lines.append('    FUNCT: [1]')


DECK = "\n".join(deck_lines)

# Write deck to file
with open("deck.4C.yaml", "w") as f:
    f.write(DECK)

# ============================================================================
# RUN 4C BINARY
# ============================================================================
env = os.environ.copy()
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = f"{FOURC_LD}:{env.get('LD_LIBRARY_PATH', '')}"

cmd = ["stdbuf", "-oL", "-eL", FOURC_BIN, "deck.4C.yaml", "out"]
result = subprocess.run(cmd, capture_output=True, text=True, env=env)

# Write log
with open("run.log", "w") as f:
    f.write(result.stdout)
    f.write(result.stderr)

# Check for errors
if result.returncode != 0:
    print(f"4C failed with return code {result.returncode}", file=sys.stderr)
    print("STDOUT:", result.stdout[:2000], file=sys.stderr)
    print("STDERR:", result.stderr[:2000], file=sys.stderr)
    raise SystemExit("4C execution failed - check run.log")

# ============================================================================
# READ SOLUTION FROM VTU
# ============================================================================
vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not vtus:
    raise SystemExit("No VTU files found - 4C did not produce output")

# Take the last VTU (final step)
vtu = vtus[-1]
mesh = meshio.read(vtu)

vpts = [(float(p[0]), float(p[1])) for p in mesh.points]
phi = [float(x) for x in mesh.point_data["phi_1"].ravel()]

# Collapse by coordinate (QUAD4 VTU repeats each node once per element)
val_map = {}
for (x, y), u_val in zip(vpts, phi):
    key = (round(x, 12), round(y, 12))
    val_map[key] = u_val

# Get solution at all nodes
u = [val_map[(round(nodes[i][0], 12), round(nodes[i][1], 12))] for i in range(N_NODES)]

# ============================================================================
# RECOVER OUTWARD NORMAL FLUX
# ============================================================================
fbname = next((da for da in mesh.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("No flux_boundary field in VTU - check CALCFLUX_BOUNDARY setting")

fb = mesh.point_data[fbname]

# Outward normal for right interface: (1, 0)
nrm = (1, 0)

# Map flux vectors by coordinate
fbmap = {}
for (x, y), vec in zip(vpts, fb):
    key = (round(x, 10), round(y, 10))
    flux_normal = float(vec[0] * nrm[0] + vec[1] * nrm[1])
    fbmap[key] = flux_normal

# Get flux at interior interface nodes
q_own = [fbmap[(round(nodes[nid-1][0], 10), round(nodes[nid-1][1], 10))] for nid in interior]
coords = [[nodes[nid-1][0], nodes[nid-1][1]] for nid in interior]

# ============================================================================
# WRITE EXPORTS.JSON
# ============================================================================
# DIRICHLET side: values = [], export normal_fluxes
exports = {
    "field_name": "u",
    "n_points": len(coords),
    "coordinates": coords,
    "values": [],  # Dirichlet side does not own the trace
    "normal_fluxes": q_own
}

# Self-check: verify finite values
chk_vals = np.array([], dtype=float)
chk_flux = np.array(q_own, dtype=float)

if not np.isfinite(chk_flux).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface fluxes")

json.dump(exports, open("exports.json", "w"), indent=2)

# ============================================================================
# WRITE PER-LEVEL FILES
# ============================================================================
# field_level{LEVEL}_A.csv
with open(f"field_level{LEVEL}_A.csv", "w") as f:
    f.write("x,y,u\n")
    for (x, y), u_val in zip(nodes, u):
        f.write(f"{x:.11e},{y:.11e},{u_val:.11e}\n")

# interface_level{LEVEL}_A.csv
with open(f"interface_level{LEVEL}_A.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for (x, y), nid, q in zip(coords, interior, q_own):
        u_val = u[nid - 1]
        f.write(f"{x:.11e},{y:.11e},{u_val:.11e},{q:.11e}\n")

# ============================================================================
# PRINT NDOF
# ============================================================================
print(f"NDOF = {N_NODES}")
print(f"4C Dirichlet participant: NDOF = {N_NODES}, max|u| = {max(abs(t) for t in u) if u else 0:.6e}")
