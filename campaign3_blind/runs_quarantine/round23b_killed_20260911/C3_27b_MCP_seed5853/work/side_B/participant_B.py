"""4C as the NEUMANN side - simplified version"""
import json
from pathlib import Path
import numpy as np
import subprocess
import glob
import meshio
import os
import sys

CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV = CFG["k"]
IF = "bottom"
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
FOURC_LD = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")

SRC_EXPR_4C = str(CFG.get("source_expr", "0.0"))
HAS_SRC = SRC_EXPR_4C.strip() not in ("", "0", "0.", "0.0")

imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())

def partner_flux(x_coord):
    for _n, d in imp.items():
        co = d.get("coordinates") or []
        q = d.get("normal_fluxes") or []
        if co and q and len(q) == len(co):
            pts = sorted(zip([c[0] for c in co], [float(v) for v in q]))
            xs = [p[0] for p in pts]; qs = [p[1] for p in pts]
            t = min(max(x_coord, xs[0]), xs[-1])
            for a, b, qa, qb in zip(xs, xs[1:], qs, qs[1:]):
                if a <= t <= b:
                    w = 0.0 if b == a else (t - a) / (b - a)
                    return qa + w * (qb - qa)
            return qs[-1]
    return 0.0

HX = (X1 - X0) / NX
HY = (Y1 - Y0) / NY
nodes = []
for j in range(NY + 1):
    y = Y0 + j * HY
    for i in range(NX + 1):
        x = X0 + i * HX
        nodes.append((x, y))

_EPS = 1e-9
iface_nodes = [i+1 for i, (x, y) in enumerate(nodes) if abs(y - Y0) < _EPS]
iface_nodes.sort(key=lambda n: nodes[n-1][0])
interior = iface_nodes[1:-1]

elements = []
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n_bl = j * (NX + 1) + i + 1
        n_br = n_bl + 1
        n_tr = n_br + (NX + 1)
        n_tl = n_tr - 1
        elements.append((elem_id, n_bl, n_br, n_tr, n_tl))
        elem_id += 1

# Build deck with single sections
deck_lines = ['TITLE:', '  - "Coupled B"']
deck_lines.extend(['PROBLEM SIZE:', f'  ELEMENTS: {len(elements)}', f'  NODES: {len(nodes)}'])
deck_lines.extend(['PROBLEM TYPE:', '  PROBLEMTYPE: "Scalar_Transport"'])
deck_lines.extend(['SCALAR TRANSPORT DYNAMIC:', '  TIMEINTEGR: "Stationary"', '  SOLVERTYPE: "linear_full"',
                   '  VELOCITYFIELD: "zero"', '  TIMESTEP: 1.0', '  NUMSTEP: 1', '  MAXTIME: 1.0',
                   '  LINEAR_SOLVER: 1', '  CALCFLUX_BOUNDARY: "diffusive"'])
deck_lines.extend(['SOLVER 1:', '  SOLVER: "UMFPACK"'])
deck_lines.extend(['MATERIALS:', '  - MAT: 1', '    MAT_scatra:', f'      DIFFUSIVITY: {KV}'])

if HAS_SRC:
    deck_lines.extend(['FUNCT1:', f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR_4C}"'])

# Collect all Dirichlet nodes (left, right, top)
dirich_nodes = []
for i, (x, y) in enumerate(nodes):
    if abs(x - X0) < _EPS or abs(x - X1) < _EPS or abs(y - Y1) < _EPS:
        dirich_nodes.append(i+1)

# Single DESIGN POINT DIRICH CONDITIONS section
deck_lines.append('DESIGN POINT DIRICH CONDITIONS:')
dnode_id = 1
for n in dirich_nodes:
    deck_lines.extend([f'  - E: {dnode_id}', '    NUMDOF: 1', '    ONOFF: [1]', '    VAL: [0.0]', '    FUNCT: [0]'])
    dnode_id += 1

# Interface Neumann conditions
h_if = HX
deck_lines.append('DESIGN POINT NEUMANN CONDITIONS:')
neum_dnode_start = dnode_id
for idx, n in enumerate(interior):
    x, y = nodes[n-1]
    q_val = partner_flux(x)
    nodal_load = h_if * q_val  # Simplified
    deck_lines.extend([f'  - E: {dnode_id}', '    NUMDOF: 1', '    ONOFF: [1]', 
                      f'    VAL: [{nodal_load}]', '    FUNCT: [0]'])
    dnode_id += 1

if HAS_SRC:
    deck_lines.extend(['DESIGN SURF NEUMANN CONDITIONS:', '  - E: 1', '    NUMDOF: 1', 
                      '    ONOFF: [1]', '    VAL: [1.0]', '    FUNCT: [1]'])

deck_lines.extend(['SCATRA FLUX CALC LINE CONDITIONS:', '  - E: 1'])

deck_lines.append('NODE COORDS:')
for i, (x, y) in enumerate(nodes):
    deck_lines.append(f'  - "NODE {i+1} COORD {x:.10f} {y:.10f} 0.0"')

deck_lines.append('TRANSPORT ELEMENTS:')
for eid, n_bl, n_br, n_tr, n_tl in elements:
    deck_lines.append(f'  - "{eid} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')

# DNODE-NODE TOPOLOGY - combine all
deck_lines.append('DNODE-NODE TOPOLOGY:')
dn_counter = 1
for n in dirich_nodes:
    deck_lines.append(f'  - "NODE {n} DNODE {dn_counter}"')
    dn_counter += 1
for n in interior:
    deck_lines.append(f'  - "NODE {n} DNODE {dn_counter}"')
    dn_counter += 1

# DLINE for flux calc
deck_lines.append('DLINE-NODE TOPOLOGY:')
for n in interior:
    deck_lines.append(f'  - "NODE {n} DLINE 1"')

if HAS_SRC:
    deck_lines.append('DSURF-NODE TOPOLOGY:')
    for i in range(len(nodes)):
        deck_lines.append(f'  - "NODE {i+1} DSURFACE 1"')

with open('problem.4C.yaml', 'w') as f:
    f.write('\n'.join(deck_lines))

env = os.environ.copy()
env['LD_LIBRARY_PATH'] = f"{FOURC_LD}:{env.get('LD_LIBRARY_PATH', '')}"
cmd = ['stdbuf', '-oL', '-eL', FOURC_BIN, 'problem.4C.yaml', 'out']
result = subprocess.run(cmd, capture_output=True, text=True, env=env)
with open('run.log', 'w') as f:
    f.write(result.stdout + result.stderr)

vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not vtus:
    print("ERROR: No VTU files found", file=sys.stderr)
    sys.exit(1)

vtu = vtus[-1]
m = meshio.read(vtu)
vpts = [(float(p[0]), float(p[1])) for p in m.points]
phi = [float(x) for x in m.point_data["phi_1"].ravel()]

val_map = {}
for (x, y), u in zip(vpts, phi):
    val_map[(round(x, 12), round(y, 12))] = u

u_vals = [val_map[(round(x, 12), round(y, 12))] for (x, y) in nodes]

fbname = next((da for da in m.point_data if "flux_boundary" in da), None)
if fbname is None:
    print("ERROR: No flux_boundary field", file=sys.stderr)
    sys.exit(1)

fb = m.point_data[fbname]
nrm = (0, -1)  # bottom normal points down
fbmap = {}
for (x, y), vec in zip(vpts, fb):
    fbmap[(round(x, 10), round(y, 10))] = float(vec[0] * nrm[0] + vec[1] * nrm[1])

q_own = [fbmap[(round(nodes[n-1][0], 10), round(nodes[n-1][1], 10))] for n in interior]
co = [list(nodes[n-1]) for n in interior]
vals = [u_vals[n-1] for n in interior]

json.dump({"field_name": "u", "coordinates": co, "values": vals,
           "normal_fluxes": q_own, "n_points": len(co)}, open("exports.json", "w"))

_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as f:
    f.write("x,y,u\n")
    for (px, py), u in zip(nodes, u_vals):
        f.write(f"{px:.11e},{py:.11e},{float(u):.11e}\n")

with open(f"interface_level{_LVL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for (px, py), n, q in zip(co, interior, q_own):
        f.write(f"{px:.11e},{py:.11e},{float(u_vals[n-1]):.11e},{float(q):.11e}\n")

print(f"NDOF = {len(nodes)}")
print(f"4C Neumann participant: NDOF = {len(nodes)}  max|u| = {max(abs(t) for t in u_vals):.6e}")
