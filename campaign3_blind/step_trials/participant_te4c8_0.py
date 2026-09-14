"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
The traction uses the heat flux's sign rule (minus the flux through n_out): the two sides' exports
cancel and the Neumann partner applies them UNCHANGED. The task's OUTWARD traction is MINUS (qx, qy).
TWO 4C RUNS PER ITERATION: run T (Scalar_Transport with CALCFLUX_BOUNDARY "diffusive"), run U
(Thermo_Structure_Interaction on a one-element-thick SOLIDSCATRA HEX8 slab with u_z pinned).
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

CFG = json.loads(Path("config.json").read_text())
# A multi-level coupling call hands this level's keys in the environment
# (OASIS_CONFIG_JSON, a JSON object) instead of writing this file.
CFG.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))
NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = (float(CFG["x0"]), float(CFG["x1"]),
                  float(CFG["y0"]), float(CFG["y1"]))
KV = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IF = CFG.get("iface", "right")
SIDE = CFG.get("side", "dirichlet")
if SIDE != "dirichlet":
    raise SystemExit("this contract is the DIRICHLET role of a thermo-elastic "
                     "exchange (T, ux, uy in; qn, qx, qy out). 4C on the NEUMANN "
                     "side of a thermo-elastic exchange is not served: give that "
                     "role to the partner code, or apply the imported [qn, qx, qy] "
                     "yourself as POINT NEUMANN loads on both slab layers.")
# 4C's material takes E, nu and the linear expansion coefficient; the task
# gives Lame parameters and beta. These are the exact conversions.
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)
# SOURCES AS 4C EXPRESSION STRINGS (not Python functions): '^' for powers,
# lowercase 'pi', 'x', 'y'. "0.0" means none. They reach the solve only through
# FUNCT blocks in YOUR decks; a Python src() that never reaches a deck does
# nothing (the classic 4C trap).
SRC_T = str(CFG.get("source_T", "0.0")).replace("**", "^")
SRC_UX = str(CFG.get("source_ux", "0.0")).replace("**", "^")
SRC_UY = str(CFG.get("source_uy", "0.0")).replace("**", "^")

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")


def partner_values(y_or_x):
    """The partner's (T, ux, uy) at one of THIS side's interface points, each
    component interpolated on its own along the interface (the driver does not
    interpolate between the two meshes; one np.interp over a flattened (N, 3)
    array interleaves the components -- right length, every number wrong).
    Empty on iteration 1: fall back to (0, 0, 0)."""
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
        ax = 1 if IF in ("left", "right") else 0
        s = np.asarray([c[ax] for c in co], float)
        o = np.argsort(s)
        t = min(max(float(y_or_x), s[o][0]), s[o][-1])
        return tuple(float(np.interp(t, s[o], va[o, c])) for c in range(3))
    return (0.0, 0.0, 0.0)


# ── DID 4C FINISH? (served: when a run leaves no output, name the cause) ─
def why_4c_did_not_finish(tag=""):
    """When a run leaves no usable output: 4C's own error line, then the deck defects measured on worker decks."""
    _why = []
    try:
        for _lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
            try:
                _lines = open(_lg, errors="ignore").read().splitlines()
            except OSError:
                continue
            for _i, _ln in enumerate(_lines):
                if "PROC 0 ERROR" in _ln:
                    _said = [l.strip() for l in _lines[_i + 1:_i + 12] if l.strip() and not l.startswith("---") and "MPI_ABORT" not in l]
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said[:8]))
                    break
        for _deck in sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml")):
            _txt = open(_deck, errors="ignore").read()
            _secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
            _dup = sorted({x for x in _secs if _secs.count(x) > 1})
            if _dup:
                _why.append(f"{_deck}: section(s) written twice: " + ", ".join(_dup))
            if "Thermo_Structure_Interaction" in _txt:
                if "CLONING MATERIAL MAP" not in _txt:
                    _why.append(f"{_deck}: TSI needs a CLONING MATERIAL MAP pairing the structure material with the MAT_Fourier thermal material")
                if "COUPVARIABLE" not in _txt or "Temperature" not in _txt.split("COUPVARIABLE", 1)[-1][:40]:
                    _why.append(f"{_deck}: TSI DYNAMIC/PARTITIONED needs COUPVARIABLE \"Temperature\" (the default gives zero thermal strain)")
                if re.search(r"\b(WALL|SOLID) QUAD4\b|\bTRI3\b", _txt):
                    _why.append(f"{_deck}: 4C has no 2-D thermo-elastic element; use the one-element-thick SOLIDSCATRA HEX8 slab")
                if "monitor_reaction" in _txt and "IO/MONITOR STRUCTURE DBC" not in _txt:
                    _why.append(f"{_deck}: TAG: monitor_reaction writes nothing without an IO/MONITOR STRUCTURE DBC section")
                if "DESIGN VOL THERMO DIRICH" in _txt:
                    _why.append(f"{_deck}: DESIGN VOL THERMO DIRICH imposes the temperature volume-wide; use SURF (outer) and POINT (interface) THERMO DIRICH")
            if "Scalar_Transport" in _txt:
                if "THERMAL DYNAMIC:" in _txt and "SCALAR TRANSPORT DYNAMIC:" not in _txt:
                    _why.append(f"{_deck}: Scalar_Transport needs `SCALAR TRANSPORT DYNAMIC`, not `THERMAL DYNAMIC`")
                if "CALCFLUX_BOUNDARY" not in _txt or "FLUX CALC" not in _txt:
                    _why.append(f"{_deck}: needs CALCFLUX_BOUNDARY \"diffusive\" AND a `SCATRA FLUX CALC LINE CONDITIONS` entry (E 2) for the flux recovery")
                if re.search(r"^IO:\s*$", _txt, re.M):
                    _why.append(f"{_deck}: an `IO:` section in a Scalar_Transport deck is rejected; the VTU appears without it")
            _badkw = sorted({w for w in re.findall(r'"NODE\s+\d+\s+(D[A-Z]+)\s+\d+"', _txt) if w not in ("DNODE", "DLINE", "DSURFACE", "DVOL")})
            if _badkw:
                _why.append(f"{_deck}: topology entries use {', '.join(_badkw)} -- the entity words are DNODE, DLINE, DSURFACE, DVOL (anything else defines nothing and the conditions on it are silently dropped)")
            _topo = set(re.findall(r"D(?:NODE|LINE|SURF|VOL)\s+(\d+)", _txt))
            for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
                _head = _b.split(":", 1)[0].strip()
                if _head.startswith("DESIGN") and _head.endswith("CONDITIONS"):
                    _missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", _b) if x not in _topo}, key=int)
                    if _missing:
                        _why.append(f"{_deck}: {_head} names E id(s) {', '.join(_missing[:6])} that no *-NODE TOPOLOGY section defines")
    except Exception as _e:
        _why.append(f"(diagnosis failed: {_e!r})")
    if not _why:
        _why.append("no 4C error line in any *.log here -- run the binary line-buffered (stdbuf -oL -eL) with its console in a log next to the deck")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


atexit.register(_diagnose_at_exit)

# ── THE HOLE (yours): the mesh, the two decks and the two runs ──
# Build the 2-D node layout (NX x NY on [X0, X1] x [Y0, Y1])
h_x = (X1 - X0) / NX
h_y = (Y1 - Y0) / NY
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * h_x
        y = Y0 + j * h_y
        nodes.append((x, y))
n_nodes = len(nodes)

# Interface nodes: right side (x = X1), interior (excluding corners at y=0 and y=Y1)
if IF == "right":
    interior = [j * (NX + 1) + 1 + NX + 1 for j in range(1, NY)]  # 1-based ids
    interior = [n + 1 for n in interior]  # convert to 1-based
    interior = [nodes.index((X1, Y0 + j * h_y)) + 1 for j in range(1, NY)]
elif IF == "left":
    interior = [j * (NX + 1) + 1 for j in range(1, NY)]
    interior = [nodes.index((X0, Y0 + j * h_y)) + 1 for j in range(1, NY)]
elif IF == "top":
    interior = [i + 1 for i in range(1, NX + 1)]
    interior = [nodes.index((X0 + i * h_x, Y1)) + 1 for i in range(1, NX + 1)]
elif IF == "bottom":
    interior = [i + 1 + NX + 1 for i in range(1, NX + 1)]
    interior = [nodes.index((X0 + i * h_x, Y0)) + 1 for i in range(1, NX + 1)]

# Slab thickness for the TSI run (one element thick in z)
TZ = 0.1  # arbitrary small thickness

# FOURC binary and library path
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
FOURC_LD = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")

# ── RUN T: Scalar_Transport (2-D) ──
# Build deck T
deck_t = f'''TITLE:
  - "Subdomain A - Temperature"
PROBLEM SIZE:
  ELEMENTS: {NX * NY}
  NODES: {n_nodes}
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
'''

# Add outer boundary Dirichlet conditions (T=0 on x=0, y=0, y=1)
outer_surf_ids = []
surf_id = 2
if IF == "right":
    # x=0 (left)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # y=0 (bottom)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # y=Y1 (top)
    outer_surf_ids.append(surf_id)
    surf_id += 1
elif IF == "left":
    # x=X1 (right)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # y=0 (bottom)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # y=Y1 (top)
    outer_surf_ids.append(surf_id)
    surf_id += 1
elif IF == "top":
    # x=0 (left)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # x=X1 (right)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # y=0 (bottom)
    outer_surf_ids.append(surf_id)
    surf_id += 1
elif IF == "bottom":
    # x=0 (left)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # x=X1 (right)
    outer_surf_ids.append(surf_id)
    surf_id += 1
    # y=Y1 (top)
    outer_surf_ids.append(surf_id)
    surf_id += 1

for sid in outer_surf_ids:
    deck_t += f'''DESIGN SURF TRANSPORT DIRICH CONDITIONS:
  - E: {sid}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

# Add interface Dirichlet conditions (one per interior node)
interface_line_id = surf_id
deck_t += f'''DESIGN LINE TRANSPORT DIRICH CONDITIONS:
  - E: {interface_line_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

# Interface point Dirichlet conditions (one per interior node)
for idx, node_id in enumerate(interior):
    t_val = partner_values(nodes[node_id - 1][1 if IF in ("left", "right") else 0])[0]
    deck_t += f'''DESIGN POINT TRANSPORT DIRICH CONDITIONS:
  - E: {node_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{t_val}]
    FUNCT: [0]
'''

# Flux calc line conditions
deck_t += f'''SCATRA FLUX CALC LINE CONDITIONS:
  - E: {interface_line_id}
'''

# Node coords
deck_t += "NODE COORDS:\n"
for i, (x, y) in enumerate(nodes):
    deck_t += f'  - "NODE {i + 1} COORD {x:.15e} {y:.15e} 0.0"\n'

# Transport elements (QUAD4)
deck_t += "TRANSPORT ELEMENTS:\n"
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n1 = i + j * (NX + 1) + 1
        n2 = i + 1 + j * (NX + 1) + 1
        n3 = i + 1 + (j + 1) * (NX + 1) + 1
        n4 = i + (j + 1) * (NX + 1) + 1
        deck_t += f'  - "{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n'
        elem_id += 1

# Topology: DSURFACE for outer boundaries and volume source
deck_t += "DSURF-NODE TOPOLOGY:\n"
# All nodes for volume source (E: 1)
for n in range(1, n_nodes + 1):
    deck_t += f'  - "NODE {n} DSURFACE 1"\n'

# DSURFACE for outer boundaries
for sid, btype in enumerate(zip(outer_surf_ids, ["left", "bottom", "top"])):
    sid = btype[0]
    if btype[1] == "left":
        for j in range(NY + 1):
            n = j * (NX + 1) + 1
            deck_t += f'  - "NODE {n} DSURFACE {sid}"\n'
    elif btype[1] == "bottom":
        for i in range(NX + 1):
            n = i + 1
            deck_t += f'  - "NODE {n} DSURFACE {sid}"\n'
    elif btype[1] == "top":
        for i in range(NX + 1):
            n = i + (NY) * (NX + 1) + 1
            deck_t += f'  - "NODE {n} DSURFACE {sid}"\n'

# DLINE for interface
deck_t += "DLINE-NODE TOPOLOGY:\n"
for node_id in interior:
    deck_t += f'  - "NODE {node_id} DLINE {interface_line_id}"\n'

# DNODE for interface point conditions
deck_t += "DNODE-NODE TOPOLOGY:\n"
for node_id in interior:
    deck_t += f'  - "NODE {node_id} DNODE {node_id}"\n'

# Write deck T
Path("deck_T.4C.yaml").write_text(deck_t)

# Run T
OUT_T = "out_T"
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = FOURC_LD
try:
    result = subprocess.run(
        ["stdbuf", "-oL", "-eL", FOURC_BIN, "deck_T.4C.yaml", OUT_T],
        env=env,
        capture_output=True,
        text=True,
        timeout=300
    )
    Path("run_T.log").write_text(result.stdout + result.stderr)
except Exception as e:
    raise SystemExit(f"run T failed: {e}")

# ── RUN U: Thermo_Structure_Interaction (slab) ──
# Build the slab mesh (2 layers in z)
slab_nodes = []
for layer in range(2):
    z = layer * TZ
    for x, y in nodes:
        slab_nodes.append((x, y, z))
n_slab_nodes = len(slab_nodes)

# Slab elements (HEX8)
slab_elements = []
elem_id = 1
for j in range(NY):
    for i in range(NX):
        # Bottom layer nodes
        n1_b = i + j * (NX + 1) + 1
        n2_b = i + 1 + j * (NX + 1) + 1
        n3_b = i + 1 + (j + 1) * (NX + 1) + 1
        n4_b = i + (j + 1) * (NX + 1) + 1
        # Top layer nodes (offset by n_nodes)
        n1_t = n1_b + n_nodes
        n2_t = n2_b + n_nodes
        n3_t = n3_b + n_nodes
        n4_t = n4_b + n_nodes
        slab_elements.append(f"{elem_id} SOLIDSCATRA HEX8 {n1_b} {n2_b} {n3_b} {n4_b} {n1_t} {n2_t} {n3_t} {n4_t} MAT 1 TYPE Std")
        elem_id += 1

deck_u = f'''TITLE:
  - "Subdomain A - Thermo-Elastic"
PROBLEM SIZE:
  ELEMENTS: {len(slab_elements)}
  NODES: {n_slab_nodes}
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
THERMO STRUCTURE INTERACTION DYNAMIC:
  COUPALGO: "tsi_oneway"
  COUPVARIABLE: "Temperature"
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
SOLVER 1:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      LAM: {LAM}
      MU: {MU}
      ALPHA: {ALPHA}
      INITTEMP: 0.0
    MAT_Fourier:
      CONDUCTIVITY: {KV}
CLONING MATERIAL MAP:
  - MAT: 1
    STRUCTURE_MAT: 1
    THERMAL_MAT: 1
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN VOL STRUCTURE NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [1.0, 1.0]
    FUNCT: [1, 2]
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [3]
'''

# Outer boundary conditions (u=0, T=0)
outer_surf_ids_u = []
surf_id_u = 2
if IF == "right":
    outer_surf_ids_u.extend([surf_id_u, surf_id_u + 1, surf_id_u + 2])
    surf_id_u += 3
elif IF == "left":
    outer_surf_ids_u.extend([surf_id_u, surf_id_u + 1, surf_id_u + 2])
    surf_id_u += 3
elif IF == "top":
    outer_surf_ids_u.extend([surf_id_u, surf_id_u + 1, surf_id_u + 2])
    surf_id_u += 3
elif IF == "bottom":
    outer_surf_ids_u.extend([surf_id_u, surf_id_u + 1, surf_id_u + 2])
    surf_id_u += 3

for sid in outer_surf_ids_u:
    deck_u += f'''DESIGN SURF STRUCTURE DIRICH CONDITIONS:
  - E: {sid}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: {sid}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

# Interface conditions (one per interior node, both layers)
interface_line_id_u = surf_id_u
deck_u += f'''DESIGN LINE STRUCTURE DIRICH CONDITIONS:
  - E: {interface_line_id_u}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN LINE THERMO DIRICH CONDITIONS:
  - E: {interface_line_id_u}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

for idx, node_id in enumerate(interior):
    t_val, ux_val, uy_val = partner_values(nodes[node_id - 1][1 if IF in ("left", "right") else 0])
    # Bottom layer node
    deck_u += f'''DESIGN POINT STRUCTURE DIRICH CONDITIONS:
  - E: {node_id}
    TAG: monitor_reaction
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [{ux_val}, {uy_val}, 0.0]
    FUNCT: [0, 0, 0]
DESIGN POINT THERMO DIRICH CONDITIONS:
  - E: {node_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{t_val}]
    FUNCT: [0]
'''
    # Top layer node
    deck_u += f'''DESIGN POINT STRUCTURE DIRICH CONDITIONS:
  - E: {node_id + {n_nodes}}
    TAG: monitor_reaction
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [{ux_val}, {uy_val}, 0.0]
    FUNCT: [0, 0, 0]
DESIGN POINT THERMO DIRICH CONDITIONS:
  - E: {node_id + {n_nodes}}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{t_val}]
    FUNCT: [0]
'''

# Pin z-displacement on all nodes (plane strain)
deck_u += "DESIGN POINT STRUCTURE DIRICH CONDITIONS:\n"
for n in range(1, n_slab_nodes + 1):
    deck_u += f'  - E: {n}\n    TAG: monitor_reaction\n    NUMDOF: 3\n    ONOFF: [0, 0, 1]\n    VAL: [0.0, 0.0, 0.0]\n    FUNCT: [0, 0, 0]\n'

# IO/MONITOR STRUCTURE DBC
deck_u += '''IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
'''

# Node coords (slab)
deck_u += "NODE COORDS:\n"
for i, (x, y, z) in enumerate(slab_nodes):
    deck_u += f'  - "NODE {i + 1} COORD {x:.15e} {y:.15e} {z:.15e}"\n'

# Slab elements
deck_u += "SOLIDSCATRA ELEMENTS:\n"
for elem in slab_elements:
    deck_u += f'  - "{elem}"\n'

# Topology
deck_u += "DVOL-NODE TOPOLOGY:\n"
for n in range(1, n_slab_nodes + 1):
    deck_u += f'  - "NODE {n} DVOL 1"\n'

# DSURFACE for outer boundaries
for sid, btype in enumerate(zip(outer_surf_ids_u, ["left", "bottom", "top"])):
    sid = btype[0]
    if btype[1] == "left":
        for j in range(NY + 1):
            for layer in range(2):
                n = j * (NX + 1) + 1 + layer * n_nodes
                deck_u += f'  - "NODE {n} DSURFACE {sid}"\n'
    elif btype[1] == "bottom":
        for i in range(NX + 1):
            for layer in range(2):
                n = i + 1 + layer * n_nodes
                deck_u += f'  - "NODE {n} DSURFACE {sid}"\n'
    elif btype[1] == "top":
        for i in range(NX + 1):
            for layer in range(2):
                n = i + (NY) * (NX + 1) + 1 + layer * n_nodes
                deck_u += f'  - "NODE {n} DSURFACE {sid}"\n'

# DLINE for interface
deck_u += "DLINE-NODE TOPOLOGY:\n"
for node_id in interior:
    for layer in range(2):
        n = node_id + layer * n_nodes
        deck_u += f'  - "NODE {n} DLINE {interface_line_id_u}"\n'

# DNODE for interface point conditions
deck_u += "DNODE-NODE TOPOLOGY:\n"
for node_id in interior:
    for layer in range(2):
        n = node_id + layer * n_nodes
        deck_u += f'  - "NODE {n} DNODE {n}"\n'

# Write deck U
DECK_U = "deck_U.4C.yaml"
Path(DECK_U).write_text(deck_u)

# Run U
OUT_U = "out_U"
try:
    result = subprocess.run(
        ["stdbuf", "-oL", "-eL", FOURC_BIN, DECK_U, OUT_U],
        env=env,
        capture_output=True,
        text=True,
        timeout=300
    )
    Path("run_U.log").write_text(result.stdout + result.stderr)
except Exception as e:
    raise SystemExit(f"run U failed: {e}")

# ── RECOVERY FROM 4C's OWN OUTPUTS (served) ──
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
    """Field values on the 2-D node layout. A 4C VTU repeats every node once
    per element (and the slab has two layers): collapse by coordinate."""
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
            raise SystemExit(f"4C VTU field {field!r} has no node at {(x, y)}: the deck's nodes and "
                             f"`nodes` disagree")
        out[i] = val[rr, :comps].mean(axis=0)
    return out


vtu_T = _latest(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None:
    raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("run T wrote no flux_boundary field -- set CALCFLUX_BOUNDARY \"diffusive\" and "
                     "add a SCATRA FLUX CALC LINE CONDITIONS entry on the interface line")
FB = _nodal(mT, fbname, 2)
nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IF]
q_n = [float(FB[n - 1, 0] * nrm[0] + FB[n - 1, 1] * nrm[1]) for n in interior]

vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None:
    raise SystemExit(why_4c_did_not_finish("run U"))
U2d = _nodal(meshio.read(vtu_U), "displacement", 2, zlayer=0.0)
Ttsi = _nodal(meshio.read(vtu_UT), "temperature", 1, zlayer=0.0)[:, 0]
# the two runs solve the same heat problem up to the load rule (O(h^2)); a larger gap is a deck defect
_dT = float(np.max(np.abs(Ttsi - T2d)) / max(float(np.max(np.abs(T2d))), 1e-300))
if _dT > 0.05:
    raise SystemExit(f"the TSI deck's temperature differs from the scatra deck's by {_dT:.3f} relative: "
                     f"the two decks do not solve the same heat problem (check FUNCT3 = the heat source, "
                     f"the POINT THERMO DIRICH values and the outer SURF THERMO DIRICH of run U)")

# REACTIONS -> TRACTION: one yaml per monitored condition (node gid ZERO-based, force f)
_gid_xy = {}
_deck_u_txt = ""
try:
    if isinstance(DECK_U, str) and len(DECK_U) < 400 and Path(DECK_U).is_file():
        _deck_u_txt = Path(DECK_U).read_text(errors="ignore")
    elif isinstance(DECK_U, str) and "NODE COORDS" in DECK_U:
        _deck_u_txt = DECK_U
except OSError:
    _deck_u_txt = ""
if not _deck_u_txt:
    for _cand in sorted(glob.glob("*.yaml")):
        _t = Path(_cand).read_text(errors="ignore")
        if "Thermo_Structure_Interaction" in _t and "NODE COORDS" in _t:
            _deck_u_txt = _t
            break
for _ln in _deck_u_txt.splitlines():
    _m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', _ln)
    if _m:
        _gid_xy[int(_m.group(1))] = (round(float(_m.group(2)), 9), round(float(_m.group(3)), 9))
_F = {}
for _yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    _txt = Path(_yf).read_text(errors="ignore")
    _gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", _txt.split("dbc monitor condition data", 1)[0], re.M)]
    _fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", _txt, re.M)
    if not (_gids and _fs):
        continue
    fx, fy = float(_fs[-1][0]), float(_fs[-1][1])
    _xy = _gid_xy.get(_gids[0] + 1)
    if _xy is None:
        continue
    _acc = _F.setdefault(_xy, [0.0, 0.0])
    _acc[0] += fx
    _acc[1] += fy
if not _F:
    raise SystemExit("run U wrote no <OUT_U>-*_monitor_dbc.yaml: every interface DESIGN POINT DIRICH "
                     "entry needs `TAG: monitor_reaction` and the deck an IO/MONITOR STRUCTURE DBC "
                     "section with FILE_TYPE yaml and WRITE_CONDITION_INFORMATION true")
_ax = 1 if IF in ("left", "right") else 0
_coord = [float(nodes[n - 1][_ax]) for n in interior]
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else (Y1 - Y0)
_share = _h * float(TZ)
q_t = []
for n in interior:
    _xy = (round(float(nodes[n - 1][0]), 9), round(float(nodes[n - 1][1]), 9))
    if _xy not in _F:
        raise SystemExit(f"no reaction file for the interface node at {_xy}: its POINT DIRICH entries "
                         f"(both layers) need TAG: monitor_reaction")
    q_t.append([_F[_xy][0] / _share, _F[_xy][1] / _share])
co = [[float(nodes[n - 1][0]), float(nodes[n - 1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# ── EXPORT SELF-CHECK ─
_chk = np.asarray(Q, float)
if not np.isfinite(_chk).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface fluxes; nothing was exported")
_chk_imp = imp if imp else {}
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()]) if _chk_imp else np.zeros(0))
if _chk_qin.size == _chk.size and _chk.size and np.array_equal(_chk.ravel(), -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: the exported fluxes are the partner's array negated, bit for "
                     "bit: a copy, not a recovery from this side's own runs")
if np.abs(_chk[:, 1:]).max() == 0.0 and np.abs(U2d).max() > 0:
    raise SystemExit("EXPORT SELF-CHECK: a zero traction against a nonzero displacement field: the "
                     "reaction files carried nothing -- check TAG: monitor_reaction on the interface "
                     "POINT DIRICH entries of BOTH layers")
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))
# PER-LEVEL PERSISTENCE
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy\n")
    for (_px, _py), _t, (_ux, _uy) in zip(nodes, T2d, U2d):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_t):.11e},{float(_ux):.11e},{float(_uy):.11e}\n")
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for (_px, _py), n, (_qn, _qx, _qy) in zip(co, interior, Q):
        _f.write(f"{_px:.11e},{_py:.11e},{float(T2d[n-1]):.11e},{float(U2d[n-1,0]):.11e},"
                 f"{float(U2d[n-1,1]):.11e},{_qn:.11e},{-_qx:.11e},{-_qy:.11e}\n")
# THE RUN-LOG CONTRACT LINE
print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
