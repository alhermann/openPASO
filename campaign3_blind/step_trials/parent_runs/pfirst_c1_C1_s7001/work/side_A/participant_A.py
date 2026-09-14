#!/usr/bin/env python3
"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling."""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

CFG = json.loads(Path("config.json").read_text())
CFG.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))
NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = (float(CFG["x0"]), float(CFG["x1"]),
                  float(CFG["y0"]), float(CFG["y1"]))
KV = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IF = CFG.get("iface", "right")

# Material conversions
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)

# Sources as 4C expressions
SRC_T = str(CFG.get("source_T", "0.0")).replace("**", "^")
SRC_UX = str(CFG.get("source_ux", "0.0")).replace("**", "^")
SRC_UY = str(CFG.get("source_uy", "0.0")).replace("**", "^")
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")

# Partner's interface samples
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")

def partner_values(y_coord):
    for _n, d in imp.items():
        co = d.get("coordinates") or []
        va = d.get("values") or []
        if not (co and va and len(va) == len(co)):
            continue
        va = np.asarray(va, float)
        if va.ndim == 1:
            va = va.reshape(-1, 1)
        if va.shape[1] < 3:
            continue
        s = np.asarray([c[1] for c in co], float)
        o = np.argsort(s)
        t = min(max(float(y_coord), s[o][0]), s[o][-1])
        return tuple(float(np.interp(t, s[o], va[o, c])) for c in range(3))
    return (0.0, 0.0, 0.0)

# Build 2-D node layout
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes.append((x, y))

# Interface nodes (right side: i = NX)
interior = [NX + 1 + (NX + 1) * j for j in range(1, NY)]

# Slab thickness
TZ = (X1 - X0) / NX * 0.1
OUT_T, OUT_U = "run_T", "run_U"
N2D = (NX + 1) * (NY + 1)

# ============ DECK T (Scalar_Transport) ============
deck_T = f'''PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_scatra:
      DIFFUSIVITY: {KV}
'''

if SRC_T != "0.0":
    deck_T += f'''
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
'''

# Outer boundary Dirichlet (T=0 on left, bottom, top)
deck_T += '''
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
  - E: 4
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

# Interface Dirichlet (per-node)
deck_T += "\nDESIGN POINT DIRICH CONDITIONS:\n"
for n in interior:
    T_val, _, _ = partner_values(nodes[n-1][1])
    deck_T += f'''  - E: {n}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_val:.15e}]
    FUNCT: [0]
'''

# Node coords
deck_T += "\nNODE COORDS:\n"
for idx, (x, y) in enumerate(nodes, 1):
    deck_T += f'  - "NODE {idx} COORD {x:.15e} {y:.15e} 0.0"\n'

# Transport elements
deck_T += "\nTRANSPORT ELEMENTS:\n"
for j in range(NY):
    for i in range(NX):
        n0 = i + 1 + (NX + 1) * j
        elem_id = j * NX + i + 1
        deck_T += f'  - "{elem_id} TRANSP QUAD4 {n0} {n0+1} {n0+NX+2} {n0+NX+1} MAT 1 TYPE Std"\n'

# Topology
deck_T += "\nDLINE-NODE TOPOLOGY:\n"
for j in range(NY + 1):
    deck_T += f'  - "NODE {1+(NX+1)*j} DLINE 1"\n'  # left
    deck_T += f'  - "NODE {NX+1+(NX+1)*j} DLINE 2"\n'  # interface
for i in range(NX + 1):
    deck_T += f'  - "NODE {1+i} DLINE 3"\n'  # bottom
    deck_T += f'  - "NODE {1+i+(NX+1)*NY} DLINE 4"\n'  # top

# Volume source
if SRC_T != "0.0":
    deck_T += '''
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DSURF-NODE TOPOLOGY:
'''
    for idx in range(1, len(nodes)+1):
        deck_T += f'  - "NODE {idx} DSURFACE 1"\n'

# Point Dirichlet topology
deck_T += "\nDNODE-NODE TOPOLOGY:\n"
for n in interior:
    deck_T += f'  - "NODE {n} DNODE {n}"\n'

Path("deck_T.4C.yaml").write_text(deck_T)

# ============ DECK U (TSI slab) ============
deck_U = f'''PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
THERMAL DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
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
      DIFFUSIVITY: {KV}
CLONING MATERIAL MAP:
  - STRUCTMAT: 1
    THERMMAT: 2
'''

if SRC_UX != "0.0":
    deck_U += f'\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"\n'
if SRC_UY != "0.0":
    deck_U += f'\nFUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"\n'
if SRC_T != "0.0":
    deck_U += f'\nFUNCT3:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\n'

# Outer boundary Dirichlet (u=0)
deck_U += '''
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 3
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''

# Interface Dirichlet (per-node, both layers)
deck_U += "\nDESIGN POINT DIRICH CONDITIONS:\n"
for n in interior:
    T_val, ux_val, uy_val = partner_values(nodes[n-1][1])
    deck_U += f'''  - E: {n}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]
    FUNCT: [0, 0, 0]
    TAG: monitor_reaction
  - E: {n + N2D}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]
    FUNCT: [0, 0, 0]
    TAG: monitor_reaction
'''

# Node coords (both layers)
deck_U += "\nNODE COORDS:\n"
for idx, (x, y) in enumerate(nodes, 1):
    deck_U += f'  - "NODE {idx} COORD {x:.15e} {y:.15e} 0.0"\n'
    deck_U += f'  - "NODE {idx + N2D} COORD {x:.15e} {y:.15e} {TZ:.15e}"\n'

# Structure elements (HEX8)
deck_U += "\nSTRUCTURE ELEMENTS:\n"
for j in range(NY):
    for i in range(NX):
        n0 = i + 1 + (NX + 1) * j
        elem_id = j * NX + i + 1
        deck_U += f'  - "{elem_id} SOLIDSCATRA HEX8 {n0} {n0+1} {n0+NX+2} {n0+NX+1} {n0+N2D} {n0+1+N2D} {n0+NX+2+N2D} {n0+NX+1+N2D} MAT 1"\n'

# Topology (both layers)
deck_U += "\nDLINE-NODE TOPOLOGY:\n"
for j in range(NY + 1):
    n = 1 + (NX + 1) * j
    deck_U += f'  - "NODE {n} DLINE 1"\n'
    deck_U += f'  - "NODE {n+N2D} DLINE 1"\n'
    n = NX + 1 + (NX + 1) * j
    deck_U += f'  - "NODE {n} DLINE 2"\n'
    deck_U += f'  - "NODE {n+N2D} DLINE 2"\n'
for i in range(NX + 1):
    n = 1 + i
    deck_U += f'  - "NODE {n} DLINE 3"\n'
    deck_U += f'  - "NODE {n+N2D} DLINE 3"\n'
    n = 1 + i + (NX + 1) * NY
    deck_U += f'  - "NODE {n} DLINE 4"\n'
    deck_U += f'  - "NODE {n+N2D} DLINE 4"\n'

# Volume sources
if SRC_UX != "0.0" or SRC_UY != "0.0":
    deck_U += '''
DESIGN VOL STRUCTURE NEUMANN CONDITIONS:
'''
    if SRC_UX != "0.0":
        deck_U += '''  - E: 1
    NUMDOF: 3
    ONOFF: [1, 0, 0]
    VAL: [1.0, 0.0, 0.0]
    FUNCT: [1, 0, 0]
'''
    if SRC_UY != "0.0":
        deck_U += '''  - E: 2
    NUMDOF: 3
    ONOFF: [0, 1, 0]
    VAL: [0.0, 1.0, 0.0]
    FUNCT: [0, 2, 0]
DVOL-NODE TOPOLOGY:
'''
        for idx in range(1, 2*N2D+1):
            deck_U += f'  - "NODE {idx} DVOL 2"\n'
    else:
        for idx in range(1, 2*N2D+1):
            deck_U += f'  - "NODE {idx} DVOL 1"\n'

if SRC_T != "0.0":
    deck_U += '''
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 3
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [3]
DVOL-NODE TOPOLOGY:
'''
    for idx in range(1, 2*N2D+1):
        deck_U += f'  - "NODE {idx} DVOL 3"\n'

# Point Dirichlet topology
deck_U += "\nDNODE-NODE TOPOLOGY:\n"
for n in interior:
    deck_U += f'  - "NODE {n} DNODE {n}"\n'
    deck_U += f'  - "NODE {n+N2D} DNODE {n+N2D}"\n'

# Runtime output
deck_U += f'''
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  FILE_TYPE: vtu
  PREFIX: "{OUT_U}-vtk-files/scatra"
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  INTERVAL_STEPS: 1
  FILE_TYPE: vtu
  PREFIX: "{OUT_U}-vtk-files/structure"
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
  FILE_TYPE: vtu
  PREFIX: "{OUT_U}-vtk-files/thermo"
  OUTPUT_THERMO: true
  TEMPERATURE: true
IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
  PREFIX: "{OUT_U}"
'''

DECK_U = "deck_U.4C.yaml"
Path(DECK_U).write_text(deck_U)

# Run deck T
log_T = subprocess.run(["stdbuf", "-oL", "-eL", FOURC_BIN, "deck_T.4C.yaml", OUT_T],
                       capture_output=True, text=True)
Path("run_T.log").write_text(log_T.stdout + log_T.stderr)

# Run deck U
log_U = subprocess.run(["stdbuf", "-oL", "-eL", FOURC_BIN, DECK_U, OUT_U],
                       capture_output=True, text=True)
Path("run_U.log").write_text(log_U.stdout + log_U.stderr)

# Recovery
import glob
import re
import meshio

def _latest(pattern):
    vs = sorted(glob.glob(pattern))
    if not vs:
        return None
    def _step(p):
        m = re.match(r".*-(\d+)-\d+\.vtu$", p)
        return int(m.group(1)) if m else -1
    return max(vs, key=_step)

def _nodal(m, field, comps, zlayer=None):
    pts = np.asarray(m.points)
    val = np.asarray(m.point_data[field])
    if val.ndim == 1:
        val = val.reshape(-1, 1)
    rows = np.arange(len(pts)) if zlayer is None else np.where(np.abs(pts[:, 2] - zlayer) < 1e-9)[0]
    key = {}
    for r in rows:
        key.setdefault((round(float(pts[r, 0]), 9), round(float(pts[r, 1]), 9)), []).append(r)
    out = np.zeros((len(nodes), comps))
    for i, (x, y) in enumerate(nodes):
        rr = key.get((round(float(x), 9), round(float(y), 9)))
        if not rr:
            raise SystemExit(f"4C VTU field {field!r} has no node at {(x, y)}")
        out[i] = val[rr, :comps].mean(axis=0)
    return out

vtu_T = _latest(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None:
    raise SystemExit("run T wrote no VTU")
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]

vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None:
    raise SystemExit("run U wrote no VTU")
U2d = _nodal(meshio.read(vtu_U), "displacement", 2, zlayer=0.0)

# Flux recovery from reactions
_gid_xy = {}
for ln in deck_U.splitlines():
    m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', ln)
    if m:
        _gid_xy[int(m.group(1))] = (round(float(m.group(2)), 9), round(float(m.group(3)), 9))

_F = {}
for yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    txt = Path(yf).read_text(errors="ignore")
    gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", txt.split("dbc monitor condition data", 1)[0], re.M)]
    fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt, re.M)
    if not (gids and fs):
        continue
    fx, fy = float(fs[-1][0]), float(fs[-1][1])
    xy = _gid_xy.get(gids[0] + 1)
    if xy is None:
        continue
    _acc = _F.setdefault(xy, [0.0, 0.0])
    _acc[0] += fx
    _acc[1] += fy

if not _F:
    raise SystemExit("run U wrote no monitor_dbc.yaml")

_h = (Y1 - Y0) / NY
_share = _h * TZ
q_t = []
for n in interior:
    xy = (round(float(nodes[n-1][0]), 9), round(float(nodes[n-1][1]), 9))
    if xy not in _F:
        raise SystemExit(f"no reaction for node at {xy}")
    q_t.append([_F[xy][0] / _share, _F[xy][1] / _share])

# Heat flux from consistent residual (simplified - use zero for now)
q_n = [0.0] * len(interior)

co = [[float(nodes[n-1][0]), float(nodes[n-1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# Export
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))

# Per-level files
LVL = CFG.get("level", "X")
with open(f"field_level{LVL}.csv", "w") as f:
    f.write("x,y,T,ux,uy\n")
    for (px, py), t, (ux, uy) in zip(nodes, T2d, U2d):
        f.write(f"{px:.11e},{py:.11e},{t:.11e},{ux:.11e},{uy:.11e}\n")

with open(f"interface_level{LVL}.csv", "w") as f:
    f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for (px, py), n, (qn, qx, qy) in zip(co, interior, Q):
        f.write(f"{px:.11e},{py:.11e},{T2d[n-1]:.11e},{U2d[n-1,0]:.11e},{U2d[n-1,1]:.11e},{qn:.11e},{-qx:.11e},{-qy:.11e}\n")

print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}")
