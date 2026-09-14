#!/usr/bin/env python3
"""4C as NEUMANN side of a partitioned coupling (Scalar_Transport).

Side B: Subdomain (0,1) x (0.625,1.5), k=2, interface at y=0.625 (bottom edge).
Imports flux from partner (Dirichlet side A), applies it as Neumann BC,
exports field values and consistent outward flux.
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

# ── READ CONFIG ────────────────────────────────────────────────────────────
CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV = CFG["k"]  # diffusivity k=2 for subdomain B
IF = CFG.get("iface", "bottom")  # interface is bottom edge (y=0.625)
LEVEL = CFG.get("level", 1)
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
FOURC_LD = CFG.get("fourc_ld", "")

# Source term for subdomain B (converted to 4C syntax: ^ instead of **, lowercase pi)
# Original: x**3*y/2 - 25*x**3/24 - 4*x**2*y/5 + 31*x**2/15 + x*y**3/2 - 25*x*y**2/8 + 1397*x*y/640 + 1913*x/1280 - 4*y**3/15 + 31*y**2/15 - 65*y/48 - 55/32
SRC_EXPR = "x^3*y/2 - 25*x^3/24 - 4*x^2*y/5 + 31*x^2/15 + x*y^3/2 - 25*x*y^2/8 + 1397*x*y/640 + 1913*x/1280 - 4*y^3/15 + 31*y^2/15 - 65*y/48 - 55/32"
HAS_SRC = True

# ── PARTNER INTERFACE DATA (handshake) ─────────────────────────────────────
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())

def _partner(key, x_or_y):
    """Interpolate partner's data onto this side's interface points."""
    for _n, d in imp.items():
        co = d.get("coordinates") or []
        q = d.get(key) or []
        if co and q and len(q) == len(co):
            # Interface is horizontal (bottom/top), so use x coordinate
            ax = 0  # x-axis for horizontal interface
            pts = sorted(zip([c[ax] for c in co], q))
            xs = [p[0] for p in pts]
            qs = [float(p[1]) for p in pts]
            t = min(max(x_or_y, xs[0]), xs[-1])
            for a, b, qa, qb in zip(xs, xs[1:], qs, qs[1:]):
                if a <= t <= b:
                    w = 0.0 if b == a else (t - a) / (b - a)
                    return qa + w * (qb - qa)
            return qs[-1]
    return 0.0

def partner_flux(x_coord):
    """Partner's outward flux at interface point (this is our inward load)."""
    return _partner("normal_fluxes", x_coord)

# SIGN CONVENTION: Partner's outward flux = our inward load (opposite normals)
# Apply the number unchanged; no extra minus sign.

# ── MESH GENERATION ────────────────────────────────────────────────────────
# Uniform mesh on subdomain B: (0,1) x (0.625,1.5)
# NX divisions in x, NY divisions in y
hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY

# Generate node coordinates (1-based indexing for 4C)
nodes = []  # nodes[i-1] = (x, y) for node i
node_id = {}  # (row, col) -> node_id

nid = 1
for j in range(NY + 1):
    y = Y0 + j * hy
    for i in range(NX + 1):
        x = X0 + i * hx
        nodes.append((x, y))
        node_id[(j, i)] = nid
        nid += 1

N_NODES = len(nodes)

# Identify interface nodes (bottom edge: y = Y0 = 0.625)
# These are nodes with row index 0 (j=0)
interface_nodes_all = [node_id[(0, i)] for i in range(NX + 1)]

# Interior interface nodes: exclude endpoints (they're also on outer Dirichlet boundary)
interior = interface_nodes_all[1:-1]  # exclude first and last

# ── BUILD 4C DECK ─────────────────────────────────────────────────────────
deck_lines = []

# TITLE
deck_lines.append('TITLE:')
deck_lines.append(f'  - "Side B: Neumann side coupling, level {LEVEL}"')
deck_lines.append('')

# PROBLEM SIZE
deck_lines.append('PROBLEM SIZE:')
deck_lines.append(f'  ELEMENTS: {NX * NY}')
deck_lines.append(f'  NODES: {N_NODES}')
deck_lines.append('')

# PROBLEM TYPE
deck_lines.append('PROBLEM TYPE:')
deck_lines.append('  PROBLEMTYPE: "Scalar_Transport"')
deck_lines.append('')

# SCALAR TRANSPORT DYNAMIC
deck_lines.append('SCALAR TRANSPORT DYNAMIC:')
deck_lines.append('  TIMEINTEGR: "Stationary"')
deck_lines.append('  SOLVERTYPE: "linear_full"')
deck_lines.append('  NUMSTEP: 1')
deck_lines.append('  TIMESTEP: 1.0')
deck_lines.append('  MAXTIME: 1.0')
deck_lines.append('  LINEAR_SOLVER: 1')
deck_lines.append('  CALCFLUX_BOUNDARY: "diffusive"')
deck_lines.append('')

# SOLVER 1
deck_lines.append('SOLVER 1:')
deck_lines.append('  SOLVER: "UMFPACK"')
deck_lines.append('')

# MATERIALS
deck_lines.append('MATERIALS:')
deck_lines.append(f'  - MAT: 1')
deck_lines.append('    MAT_scatra:')
deck_lines.append(f'      DIFFUSIVITY: {KV}')
deck_lines.append('')

# FUNCT1: Source term
if HAS_SRC:
    deck_lines.append('FUNCT1:')
    deck_lines.append(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"')
    deck_lines.append('')

# NODE COORDS
deck_lines.append('NODE COORDS:')
for idx, (x, y) in enumerate(nodes, start=1):
    deck_lines.append(f'  - "NODE {idx} COORD {x:.15e} {y:.15e} 0.0"')
deck_lines.append('')

# TRANSPORT ELEMENTS (QUAD4 elements)
deck_lines.append('TRANSPORT ELEMENTS:')
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n1 = node_id[(j, i)]
        n2 = node_id[(j, i+1)]
        n3 = node_id[(j+1, i+1)]
        n4 = node_id[(j+1, i)]
        deck_lines.append(f'  - "{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
        elem_id += 1
deck_lines.append('')

# ── TOPOLOGY AND BOUNDARY CONDITIONS ──────────────────────────────────────
# DLINE ids for each boundary segment:
# DLINE 1: Top edge (y=Y1) - Dirichlet u=0
# DLINE 2: Left edge (x=X0) - Dirichlet u=0
# DLINE 3: Right edge (x=X1) - Dirichlet u=0
# DLINE 4: Bottom edge (y=Y0) - Interface for flux calculation

# DLINE-NODE TOPOLOGY (single section for all lines)
deck_lines.append('DLINE-NODE TOPOLOGY:')
# Top edge (DLINE 1)
for i in range(NX + 1):
    nid_top = node_id[(NY, i)]
    deck_lines.append(f'  - "NODE {nid_top} DLINE 1"')
# Left edge (DLINE 2)
for j in range(1, NY):
    nid_left = node_id[(j, 0)]
    deck_lines.append(f'  - "NODE {nid_left} DLINE 2"')
# Right edge (DLINE 3)
for j in range(1, NY):
    nid_right = node_id[(j, NX)]
    deck_lines.append(f'  - "NODE {nid_right} DLINE 3"')
# Bottom edge (DLINE 4) - interface
for i in range(NX + 1):
    nid_bottom = node_id[(0, i)]
    deck_lines.append(f'  - "NODE {nid_bottom} DLINE 4"')
deck_lines.append('')

# DESIGN LINE DIRICH CONDITIONS for outer boundaries (u=0)
deck_lines.append('DESIGN LINE DIRICH CONDITIONS:')
deck_lines.append('  - E: 1')  # Top
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('  - E: 2')  # Left
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('  - E: 3')  # Right
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('')

# ── NEUMANN CONDITIONS ON INTERFACE ───────────────────────────────────────
# For Neumann side: apply imported flux as POINT NEUMANN on interior interface nodes
# The VAL should be the nodal load: integral of flux over node's tributary length
# Using Simpson's rule: VAL = (h/6) * (q(y-h) + 4*q(y) + q(y+h))

# Calculate nodal loads for interior interface nodes
interface_nodal_loads = []
for n_idx, node_id_int in enumerate(interior):
    x_coord = nodes[node_id_int - 1][0]
    # Get flux values at this node and neighbors
    # h = hx for horizontal interface
    q_center = partner_flux(x_coord)
    q_left = partner_flux(x_coord - hx) if n_idx > 0 else q_center
    q_right = partner_flux(x_coord + hx) if n_idx < len(interior) - 1 else q_center
    # Simpson's rule for nodal load
    val = (hx / 6.0) * (q_left + 4.0 * q_center + q_right)
    interface_nodal_loads.append(val)

# DNODE-NODE TOPOLOGY for interface nodes (Neumann)
deck_lines.append('DNODE-NODE TOPOLOGY:')
for e_id, node_id_int in enumerate(interior, start=1):
    deck_lines.append(f'  - "NODE {node_id_int} DNODE {e_id}"')
deck_lines.append('')

# DESIGN POINT NEUMANN CONDITIONS for interface
deck_lines.append('DESIGN POINT NEUMANN CONDITIONS:')
for e_id, val in enumerate(interface_nodal_loads, start=1):
    deck_lines.append(f'  - E: {e_id}')
    deck_lines.append('    NUMDOF: 1')
    deck_lines.append('    ONOFF: [1]')
    deck_lines.append(f'    VAL: [{val:.15e}]')
    deck_lines.append('    FUNCT: [0]')
deck_lines.append('')

# ── VOLUME SOURCE TERM ────────────────────────────────────────────────────
# In 2D, body source goes in DESIGN SURF TRANSPORT NEUMANN CONDITIONS
if HAS_SRC:
    # DSURF-NODE TOPOLOGY: all nodes belong to the surface
    deck_lines.append('DSURF-NODE TOPOLOGY:')
    for idx in range(1, N_NODES + 1):
        deck_lines.append(f'  - "NODE {idx} DSURFACE 1"')
    deck_lines.append('')
    
    deck_lines.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
    deck_lines.append('  - E: 1')
    deck_lines.append('    NUMDOF: 1')
    deck_lines.append('    ONOFF: [1]')
    deck_lines.append('    VAL: [1.0]')
    deck_lines.append('    FUNCT: [1]')
    deck_lines.append('')

# ── FLUX CALCULATION ON INTERFACE ─────────────────────────────────────────
# Need SCATRA FLUX CALC LINE CONDITIONS for the interface line
# The interface is the bottom edge (y=Y0), which is DLINE 4
deck_lines.append('SCATRA FLUX CALC LINE CONDITIONS:')
deck_lines.append('  - E: 4')  # Bottom edge (interface)
deck_lines.append('')

DECK = 'deck.4C.yaml'
with open(DECK, 'w') as f:
    f.write('\n'.join(deck_lines))

print(f"Wrote deck to {DECK}", flush=True)

# ── RUN 4C BINARY ─────────────────────────────────────────────────────────
env = os.environ.copy()
if FOURC_LD:
    env['LD_LIBRARY_PATH'] = f"{FOURC_LD}:{env.get('LD_LIBRARY_PATH', '')}"

cmd = ['stdbuf', '-oL', '-eL', FOURC_BIN, DECK, 'out']
print(f"Running: {' '.join(cmd)}", flush=True)

result = subprocess.run(cmd, capture_output=True, text=True, env=env)
stdout = result.stdout
stderr = result.stderr

# Write logs
with open('run.log', 'w') as f:
    f.write(stdout)
    f.write(stderr)

print(stdout, flush=True)
if stderr:
    print(stderr, file=sys.stderr, flush=True)

if result.returncode != 0:
    print(f"4C exited with code {result.returncode}", file=sys.stderr, flush=True)
    # Don't exit yet - let the diagnostic below run

# ── DID 4C FINISH? ────────────────────────────────────────────────────────
vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not vtus:
    print("ERROR: No VTU files found in out-vtk-files/", file=sys.stderr, flush=True)
    print("Check run.log for 4C error messages", file=sys.stderr, flush=True)
    sys.exit(1)

# Take the last VTU (step 1 for stationary solve)
vtu = vtus[-1]
print(f"Reading solution from {vtu}", flush=True)

# Read VTU file
mesh = meshio.read(vtu)
vpts = [(float(p[0]), float(p[1])) for p in mesh.points]
phi = [float(x) for x in mesh.point_data["phi_1"].ravel()]

# Collapse by coordinate (4C QUAD4 VTU repeats nodes per element)
val_map = {}
for (x, y), u_val in zip(vpts, phi):
    key = (round(x, 12), round(y, 12))
    val_map[key] = u_val

# Extract field values at all nodes
u = [val_map[(round(nodes[i][0], 12), round(nodes[i][1], 12))] for i in range(N_NODES)]

# ── FLUX RECOVERY ─────────────────────────────────────────────────────────
fbname = next((da for da in mesh.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("no flux_boundary field in the VTU -- set CALCFLUX_BOUNDARY and add SCATRA FLUX CALC condition")

fb = mesh.point_data[fbname]
# Outward normal for bottom edge is (0, -1)
nrm = (0, -1)

fbmap = {}
for (x, y), vec in zip(vpts, fb):
    key = (round(x, 10), round(y, 10))
    flux_normal = float(vec[0] * nrm[0] + vec[1] * nrm[1])
    fbmap[key] = flux_normal

# Extract flux at interior interface nodes
q_own = [fbmap[(round(nodes[n-1][0], 10), round(nodes[n-1][1], 10))] for n in interior]
co = [list(nodes[n-1]) for n in interior]
vals = [u[n-1] for n in interior]

# ── EXPORT SELF-CHECK ─────────────────────────────────────────────────────
_chk_vals = np.asarray(vals, float).ravel()
_chk_flux = np.asarray(q_own, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes")

_chk_imp = json.loads(Path("imports.json").read_text() or "{}") if Path("imports.json").is_file() else {}
_chk_qin = np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel() for _d in _chk_imp.values()]) if _chk_imp else np.zeros(0)

if _chk_qin.size and np.abs(_chk_qin).max() > 0 and np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max():
    raise SystemExit("EXPORT SELF-CHECK: recovered flux ~0 against nonzero imported flux")

# ── WRITE EXPORTS ─────────────────────────────────────────────────────────
json.dump({
    "field_name": "u",
    "coordinates": co,
    "values": vals,  # Neumann side exports field values
    "normal_fluxes": q_own,
    "n_points": len(co)
}, open("exports.json", "w"), indent=2)

print("Wrote exports.json", flush=True)

# ── PER-LEVEL FILES ───────────────────────────────────────────────────────
with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u\n")
    for (px, py), u_val in zip(nodes, u):
        f.write(f"{px:.11e},{py:.11e},{float(u_val):.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for (px, py), n, q in zip(co, interior, q_own):
        f.write(f"{px:.11e},{py:.11e},{float(u[n-1]):.11e},{float(q):.11e}\n")

print(f"Wrote field_level{LEVEL}.csv and interface_level{LEVEL}.csv", flush=True)

# ── NDOF LINE ─────────────────────────────────────────────────────────────
print(f"NDOF = {len(nodes)}")
print(f"4C Neumann participant: NDOF = {len(nodes)} max|u| = {max(abs(t) for t in vals) if vals else 0:.6e}")
