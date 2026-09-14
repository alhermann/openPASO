"""4C as DIRICHLET side - fixed version."""
import json
import numpy as np
from pathlib import Path
import subprocess
import glob
import sys
import meshio

CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV = CFG["k"]
LEVEL = CFG.get("level", 1)
FOURC_BIN = "/home/alexander/4C/build/4C"
FOURC_LD = "/opt/4C-dependencies/lib"

SRC_EXPR = "-6*x^3*y + 16*x^3/5 - 3379*x^2*y/800 + 18979*x^2/1500 - 6*x*y^3 + 48*x*y^2/5 + 287*x*y/200 - 11077*x/1200 - 3379*y^3/2400 + 18979*y^2/1500 - 44979*y/4000"

imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())

def partner_value(y_coord):
    if not imp or "B" not in imp:
        return 0.0
    d = imp["B"]
    coords = d.get("coordinates", [])
    vals = d.get("values", [])
    if not coords or not vals:
        return 0.0
    ys = [c[1] for c in coords]
    ys_sorted = sorted(zip(ys, vals))
    xs, qs = zip(*ys_sorted)
    t = max(min(y_coord, xs[-1]), xs[0])
    for i in range(len(xs)-1):
        if xs[i] <= t <= xs[i+1]:
            w = 0.0 if xs[i+1] == xs[i] else (t - xs[i]) / (xs[i+1] - xs[i])
            return qs[i] + w * (qs[i+1] - qs[i])
    return qs[-1]

hx = (X1 - X0) / NX
hy = (Y1 - Y0) / NY
nodes = []
node_id_ij = {}
nid = 1
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * hx
        y = Y0 + j * hy
        nodes.append((x, y))
        node_id_ij[(i, j)] = nid
        nid += 1

interface_nodes_ij = [(NX, j) for j in range(1, NY)]
interface_node_ids = [node_id_ij[n] for n in interface_nodes_ij]

elements = []
el_id = 1
for j in range(NY):
    for i in range(NX):
        n1 = node_id_ij[(i, j)]
        n2 = node_id_ij[(i+1, j)]
        n3 = node_id_ij[(i+1, j+1)]
        n4 = node_id_ij[(i, j+1)]
        elements.append(f"{el_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std")
        el_id += 1

# Build deck without CALCFLUX_BOUNDARY for now - compute flux manually
deck = f'''TITLE:
  - "4C Side A"
PROBLEM SIZE:
  NODES: {len(nodes)}
  ELEMENTS: {len(elements)}
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
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {KV}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DSURF-NODE TOPOLOGY:
'''
for idx in range(len(nodes)):
    deck += f'  - "NODE {idx + 1} DSURFACE 1"\n'

# All Dirichlet in one section
dirich_lines = []
dline_topology = []
dline_id = 1

# Left edge
for j in range(NY + 1):
    dline_topology.append(f'  - "NODE {node_id_ij[(0, j)]} DLINE {dline_id}"')
dirich_lines.append(f'  - E: {dline_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]')
dline_id += 1

# Top edge
for i in range(1, NX + 1):
    dline_topology.append(f'  - "NODE {node_id_ij[(i, NY)]} DLINE {dline_id}"')
dirich_lines.append(f'  - E: {dline_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]')
dline_id += 1

# Bottom edge
for i in range(1, NX + 1):
    dline_topology.append(f'  - "NODE {node_id_ij[(i, 0)]} DLINE {dline_id}"')
dirich_lines.append(f'  - E: {dline_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]')
dline_id += 1

deck += 'DESIGN LINE DIRICH CONDITIONS:\n'
for line in dirich_lines:
    deck += line + '\n'

deck += 'DLINE-NODE TOPOLOGY:\n'
for line in dline_topology:
    deck += line + '\n'

# Interface POINT Dirichlet
point_dirich = []
dnode_topology = []
dnode_id = dline_id
for idx, (ij, nid_val) in enumerate(zip(interface_nodes_ij, interface_node_ids)):
    y_coord = Y0 + ij[1] * hy
    val = partner_value(y_coord)
    point_dirich.append(f'  - E: {dnode_id + idx}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{val}]\n    FUNCT: [0]')
    dnode_topology.append(f'  - "NODE {nid_val} DNODE {dnode_id + idx}"')

deck += 'DESIGN POINT DIRICH CONDITIONS:\n'
for line in point_dirich:
    deck += line + '\n'

deck += 'DNODE-NODE TOPOLOGY:\n'
for line in dnode_topology:
    deck += line + '\n'

deck += 'NODE COORDS:\n'
for idx, (x, y) in enumerate(nodes):
    deck += f'  - "NODE {idx + 1} COORD {x:.10f} {y:.10f} 0.0"\n'

deck += 'TRANSPORT ELEMENTS:\n'
for el in elements:
    deck += f'  - "{el}"\n'

Path("side_A_deck.4C.yaml").write_text(deck)

env = dict(subprocess.os.environ)
env["LD_LIBRARY_PATH"] = f"{FOURC_LD}:{env.get('LD_LIBRARY_PATH', '')}"
result = subprocess.run(
    ["stdbuf", "-oL", "-eL", FOURC_BIN, "side_A_deck.4C.yaml", "out"],
    capture_output=True, text=True, env=env
)

with open("run.log", "w") as f:
    f.write(result.stdout + result.stderr)

if result.returncode != 0:
    print(f"4C failed:", file=sys.stderr)
    # Find error in log
    for line in result.stdout.split('\n'):
        if 'ERROR' in line:
            print(line, file=sys.stderr)
    sys.exit(1)

vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not vtus:
    print("No VTU files", file=sys.stderr)
    sys.exit(1)

mesh = meshio.read(vtus[-1])
vpts = [(float(p[0]), float(p[1])) for p in mesh.points]
phi = [float(x) for x in mesh.point_data["phi_1"].ravel()]

val_map = {}
for (x, y), u in zip(vpts, phi):
    val_map[(round(x, 10), round(y, 10))] = u

u_at_nodes = [val_map[(round(x, 10), round(y, 10))] for x, y in nodes]

# Compute flux manually using finite differences on the interface
# q = -k * du/dn, outward normal on right is (1,0), so q = -k * du/dx
# Use backward difference from last two columns
q_own = []
for j in range(1, NY):  # Interior interface nodes only
    # Get u at interface node and node to its left
    u_if = u_at_nodes[node_id_ij[(NX, j)] - 1]
    u_left = u_at_nodes[node_id_ij[(NX-1, j)] - 1]
    du_dx = (u_if - u_left) / hx
    q = -KV * du_dx  # Outward flux
    q_own.append(q)

co = [list(nodes[nid-1]) for nid in interface_node_ids]
vals = [u_at_nodes[nid-1] for nid in interface_node_ids]

json.dump({
    "field_name": "u",
    "coordinates": co,
    "values": [],
    "normal_fluxes": q_own,
    "n_points": len(co)
}, open("exports.json", "w"), indent=2)

with open(f"field_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u\n")
    for (x, y), u in zip(nodes, u_at_nodes):
        f.write(f"{x:.11e},{y:.11e},{u:.11e}\n")

with open(f"interface_level{LEVEL}.csv", "w") as f:
    f.write("x,y,u,qn\n")
    for (x, y), u, q in zip(co, vals, q_own):
        f.write(f"{x:.11e},{y:.11e},{u:.11e},{q:.11e}\n")

print(f"NDOF = {len(nodes)}")
print(f"4C done: max|u|={max(abs(u) for u in u_at_nodes):.6e}")
