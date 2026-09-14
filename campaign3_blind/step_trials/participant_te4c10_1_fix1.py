"""Side A participant for steady thermoelastic coupling using 4C.
    Role: DIRICHLET side (imposes T, ux, uy; exports qn, qx, qy).
    Two 4C runs per iteration: Scalar_Transport for heat flux, TSI slab for traction.
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

# ---- CONFIG ----
CFG = json.loads(Path("config.json").read_text())
CFG.update(json.loads(os.environ.get("OASIS_CONFIG_JSON") or "{}"))

NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = float(CFG["x0"]), float(CFG["x1"]), float(CFG["y0"]), float(CFG["y1"])
K_VAL = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IFACE = CFG.get("iface", "right")
SIDE = CFG.get("side", "dirichlet")
if SIDE != "dirichlet":
    raise SystemExit("This script is the DIRICHLET role only.")

# Material conversions
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)

# Source terms: convert Python ** to 4C ^, uppercase PI to lowercase pi
SRC_T = str(CFG.get("source_T", "0.0")).replace("**", "^").replace("PI", "pi")
SRC_UX = str(CFG.get("source_ux", "0.0")).replace("**", "^").replace("PI", "pi")
SRC_UY = str(CFG.get("source_uy", "0.0")).replace("**", "^").replace("PI", "pi")

# Partner data
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")


def partner_values(coord):
    """Interpolate partner's [T, ux, uy] at coordinate."""
    if not imp:
        return (0.0, 0.0, 0.0)
    for _d in imp.values():
        co = _d.get("coordinates") or []
        va = _d.get("values") or []
        if not (co and va and len(va) == len(co)):
            continue
        va = np.asarray(va, float)
        if va.ndim == 1:
            va = va.reshape(-1, 1)
        if va.shape[1] < 3:
            continue
        ax = 1 if IFACE in ("left", "right") else 0
        s = np.asarray([c[ax] for c in co], float)
        o = np.argsort(s)
        t = min(max(float(coord), s[o][0]), s[o][-1])
        return tuple(float(np.interp(t, s[o], va[o, c])) for c in range(3))
    return (0.0, 0.0, 0.0)


def why_4c_did_not_finish(tag=""):
    """Diagnose when a run leaves no usable output."""
    _why = []
    try:
        for _lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
            try:
                _lines = open(_lg, errors="ignore").read().splitlines()
            except OSError:
                continue
            for _i, _ln in enumerate(_lines):
                if "PROC 0 ERROR" in _ln:
                    _said = [l.strip() for l in _lines[_i + 1:_i + 12]
                             if l.strip() and not l.startswith("---") and "MPI_ABORT" not in l]
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said[:8]))
                    break
        for _deck in sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml")):
            if "monitor_dbc" in _deck:
                continue
            _txt = open(_deck, errors="ignore").read()
            _secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
            _dup = sorted({x for x in _secs if _secs.count(x) > 1})
            if _dup:
                _why.append(f"{_deck}: section(s) written twice: " + ", ".join(_dup))
            if "Thermo_Structure_Interaction" in _txt:
                if "CLONING MATERIAL MAP" not in _txt:
                    _why.append(f"{_deck}: TSI needs CLONING MATERIAL MAP")
                if "COUPVARIABLE" not in _txt:
                    _why.append(f"{_deck}: TSI needs COUPVARIABLE Temperature")
            if "Scalar_Transport" in _txt:
                if "CALCFLUX_BOUNDARY" not in _txt:
                    _why.append(f"{_deck}: needs CALCFLUX_BOUNDARY diffusive")
    except Exception:
        pass
    if not _why:
        _why.append("no 4C error line found -- check log files")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


atexit.register(_diagnose_at_exit)

# ---- MESH GENERATION ----
dx = (X1 - X0) / NX
dy = (Y1 - Y0) / NY
TZ = max(dx, dy) / 4  # Slab thickness for plane strain

nodes = []  # 2D nodes: list of (x, y), 0-indexed
node_id_map = {}  # (x, y) -> 1-based ID for deck

nid = 1
for j in range(NY + 1):
    y = round(Y0 + j * dy, 12)
    for i in range(NX + 1):
        x = round(X0 + i * dx, 12)
        nodes.append((x, y))
        node_id_map[(x, y)] = nid
        nid += 1

N_NODES_2D = len(nodes)

# Interface nodes (1-based deck IDs, excluding endpoints)
if IFACE == "right":
    iface_x = X1
    iface_deck_ids = [node_id_map[(iface_x, round(Y0 + j * dy, 12))] for j in range(NY + 1)]
elif IFACE == "left":
    iface_x = X0
    iface_deck_ids = [node_id_map[(iface_x, round(Y0 + j * dy, 12))] for j in range(NY + 1)]
elif IFACE == "top":
    iface_y = Y1
    iface_deck_ids = [node_id_map[(round(X0 + i * dx, 12), iface_y)] for i in range(NX + 1)]
else:  # bottom
    iface_y = Y0
    iface_deck_ids = [node_id_map[(round(X0 + i * dx, 12), iface_y)] for i in range(NX + 1)]

interior = iface_deck_ids[1:-1]  # Exclude endpoints

# ---- DECK T: Scalar_Transport ----
OUT_T = "run_t"
DECK_T = f"{OUT_T}.4C.yaml"

# Build 2D QUAD4 mesh for scatra
elem_lines = []
for j in range(NY):
    for i in range(NX):
        n_bl = node_id_map[nodes[i + j * (NX + 1)]]
        n_br = node_id_map[nodes[i + 1 + j * (NX + 1)]]
        n_tr = node_id_map[nodes[i + 1 + (j + 1) * (NX + 1)]]
        n_tl = node_id_map[nodes[i + (j + 1) * (NX + 1)]]
        elem_lines.append(f'{j*NX+i+1} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std')

# Design entity IDs
DSURF_VOL = 1
DLINE_OUTER = 2
DLINE_IFACE = 3

# Topology: Outer lines
outer_lines = []
line_id = DLINE_OUTER
# Bottom (y=Y0)
for i in range(NX + 1):
    outer_lines.append(f'NODE {node_id_map[nodes[i]]} DLINE {line_id}')
# Top (y=Y1)
for i in range(NX + 1):
    outer_lines.append(f'NODE {node_id_map[nodes[i + NX * (NX + 1)]]} DLINE {line_id}')
# Left (x=X0) if not interface
if IFACE != "left":
    for j in range(NY + 1):
        outer_lines.append(f'NODE {node_id_map[nodes[j * (NX + 1)]]} DLINE {line_id}')
# Right (x=X1) if not interface
if IFACE != "right":
    for j in range(NY + 1):
        outer_lines.append(f'NODE {node_id_map[nodes[NX + j * (NX + 1)]]} DLINE {line_id}')

# Interface line topology
iface_lines = [f'NODE {n} DLINE {DLINE_IFACE}' for n in iface_deck_ids]

# DSURFACE for volume source (all nodes)
dsurf_lines = [f'NODE {node_id_map[n]} DSURFACE {DSURF_VOL}' for n in nodes]

# Point conditions for interface Dirichlet
point_conditions = []
dnode_id = 1
coord_axis = 1 if IFACE in ("left", "right") else 0
for idx, nid in enumerate(iface_deck_ids):
    coord = nodes[idx][coord_axis] if IFACE in ("left", "right") else nodes[idx][0]
    T_val, _, _ = partner_values(coord)
    point_conditions.append(f'- E: {dnode_id}\n      NUMDOF: 1\n      ONOFF: [1]\n      VAL: [{T_val:.15e}]\n      FUNCT: [0]')
    dnode_id += 1

dnode_topology = [f'NODE {nid} DNODE {idx + 1}' for idx, nid in enumerate(iface_deck_ids)]

deck_t = f'''TITLE:
  - "Scalar Transport for Side A"
PROBLEM SIZE:
  ELEMENTS: {len(elem_lines)}
  NODES: {N_NODES_2D}
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
      DIFFUSIVITY: {K_VAL}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: {DSURF_VOL}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN LINE DIRICH CONDITIONS:
  - E: {DLINE_OUTER}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN POINT DIRICH CONDITIONS:
{chr(10).join(point_conditions)}
SCATRA FLUX CALC LINE CONDITIONS:
  - E: {DLINE_IFACE}
NODE COORDS:
'''
for i, (x, y) in enumerate(nodes):
    deck_t += f'  - "NODE {i + 1} COORD {x:.12e} {y:.12e} 0.0"\n'

deck_t += 'TRANSPORT ELEMENTS:\n'
for el in elem_lines:
    deck_t += f'  - "{el}"\n'

deck_t += 'DSURF-NODE TOPOLOGY:\n'
for ln in dsurf_lines:
    deck_t += f'  - "{ln}"\n'

deck_t += 'DLINE-NODE TOPOLOGY:\n'
for ln in outer_lines + iface_lines:
    deck_t += f'  - "{ln}"\n'

deck_t += 'DNODE-NODE TOPOLOGY:\n'
for ln in dnode_topology:
    deck_t += f'  - "{ln}"\n'

Path(DECK_T).write_text(deck_t)

# ---- DECK U: TSI Plane Strain Slab ----
OUT_U = "run_u"
DECK_U = f"{OUT_U}.4C.yaml"

# Slab: two layers at z=0 and z=TZ
slab_nodes = []
slab_node_id = {}
slab_nid = 1
for (x, y) in nodes:
    slab_nodes.append((x, y, 0.0))
    slab_node_id[(x, y, 0.0)] = slab_nid
    slab_nid += 1
for (x, y) in nodes:
    slab_nodes.append((x, y, TZ))
    slab_node_id[(x, y, TZ)] = slab_nid
    slab_nid += 1

N_NODES_SLAB = len(slab_nodes)

# HEX8 elements
hex_elems = []
for j in range(NY):
    for i in range(NX):
        n0_bl = slab_node_id[(nodes[i + j*(NX+1)][0], nodes[i + j*(NX+1)][1], 0.0)]
        n0_br = slab_node_id[(nodes[i+1 + j*(NX+1)][0], nodes[i+1 + j*(NX+1)][1], 0.0)]
        n0_tr = slab_node_id[(nodes[i+1 + (j+1)*(NX+1)][0], nodes[i+1 + (j+1)*(NX+1)][1], 0.0)]
        n0_tl = slab_node_id[(nodes[i + (j+1)*(NX+1)][0], nodes[i + (j+1)*(NX+1)][1], 0.0)]
        n1_bl = slab_node_id[(nodes[i + j*(NX+1)][0], nodes[i + j*(NX+1)][1], TZ)]
        n1_br = slab_node_id[(nodes[i+1 + j*(NX+1)][0], nodes[i+1 + j*(NX+1)][1], TZ)]
        n1_tr = slab_node_id[(nodes[i+1 + (j+1)*(NX+1)][0], nodes[i+1 + (j+1)*(NX+1)][1], TZ)]
        n1_tl = slab_node_id[(nodes[i + (j+1)*(NX+1)][0], nodes[i + (j+1)*(NX+1)][1], TZ)]
        hex_elems.append(f'{j*NX+i+1} SOLIDSCATRA HEX8 {n0_bl} {n0_br} {n0_tr} {n0_tl} {n1_bl} {n1_br} {n1_tr} {n1_tl} MAT 1 KINEM linear TYPE Undefined')

# DVOL for volume thermal source and u_z constraint
dvollines = [f'NODE {nid} DVOL 1' for nid in range(1, N_NODES_SLAB + 1)]

# Interface point conditions for both layers
interface_point_cond = []
interface_dnode_topo = []
pdnid = 1

for idx, nid in enumerate(iface_deck_ids):
    coord = nodes[idx][coord_axis] if IFACE in ("left", "right") else nodes[idx][0]
    T_val, ux_val, uy_val = partner_values(coord)
    
    layer0_nid = slab_node_id[(nodes[idx][0], nodes[idx][1], 0.0)]
    layer1_nid = slab_node_id[(nodes[idx][0], nodes[idx][1], TZ)]
    
    interface_point_cond.append(f'- E: {pdnid}\n      NUMDOF: 3\n      ONOFF: [1, 1, 1]\n      VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]\n      FUNCT: [0, 0, 0]\n      TAG: monitor_reaction')
    interface_dnode_topo.append(f'NODE {layer0_nid} DNODE {pdnid}')
    pdnid += 1
    
    interface_point_cond.append(f'- E: {pdnid}\n      NUMDOF: 3\n      ONOFF: [1, 1, 1]\n      VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]\n      FUNCT: [0, 0, 0]\n      TAG: monitor_reaction')
    interface_dnode_topo.append(f'NODE {layer1_nid} DNODE {pdnid}')
    pdnid += 1

deck_u = f'''TITLE:
  - "TSI Plane Strain for Side A"
PROBLEM SIZE:
  ELEMENTS: {len(hex_elems)}
  NODES: {N_NODES_SLAB}
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLTEMP: 1e-8
  TOLRES: 1e-8
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  NUMSTEP: 1
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
SOLVER 1:
  SOLVER: "UMFPACK"
SOLVER 2:
  SOLVER: "UMFPACK"
MATERIALS:
  - MAT: 1
    MAT_Struct_ThermoStVenantK:
      YOUNG: [{E_MOD}]
      NUE: {NU}
      DENS: 1.0
      THEXPANS: {ALPHA}
      INITTEMP: 0.0
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 1.0
      CONDUCT:
        constant: [{K_VAL}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"
FUNCT3:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"
IO:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN VOL DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN POINT DIRICH CONDITIONS:
{chr(10).join(interface_point_cond)}
NODE COORDS:
'''
for i, (x, y, z) in enumerate(slab_nodes):
    deck_u += f'  - "NODE {i + 1} COORD {x:.12e} {y:.12e} {z:.12e}"\n'

deck_u += 'STRUCTURE ELEMENTS:\n'
for el in hex_elems:
    deck_u += f'  - "{el}"\n'

deck_u += 'DVOL-NODE TOPOLOGY:\n'
for ln in dvollines:
    deck_u += f'  - "{ln}"\n'

deck_u += 'DNODE-NODE TOPOLOGY:\n'
for ln in interface_dnode_topo:
    deck_u += f'  - "{ln}"\n'

Path(DECK_U).write_text(deck_u)

# ---- RUN BOTH DECKS ----
fourc_bin = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
fourc_ld = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")
env = dict(os.environ, LD_LIBRARY_PATH=fourc_ld)

def run_4c(deck, prefix):
    log_file = f"{prefix}.log"
    cmd = ["stdbuf", "-oL", "-eL", fourc_bin, deck, prefix]
    with open(log_file, "w") as log:
        result = subprocess.run(cmd, env=env, stdout=log, stderr=log)
    return result.returncode

rc_t = run_4c(DECK_T, OUT_T)
rc_u = run_4c(DECK_U, OUT_U)

if rc_t != 0:
    raise SystemExit(why_4c_did_not_finish("run T"))
if rc_u != 0:
    raise SystemExit(why_4c_did_not_finish("run U"))

# ---- RECOVERY ----
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
    raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("run T wrote no flux_boundary field")
FB = _nodal(mT, fbname, 2)
nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IFACE]
q_n = [float(FB[n - 1, 0] * nrm[0] + FB[n - 1, 1] * nrm[1]) for n in interior]

vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None:
    raise SystemExit(why_4c_did_not_finish("run U"))
U2d = _nodal(meshio.read(vtu_U), "displacement", 2, zlayer=0.0)
Ttsi = _nodal(meshio.read(vtu_UT), "temperature", 1, zlayer=0.0)[:, 0]
_dT = float(np.max(np.abs(Ttsi - T2d)) / max(float(np.max(np.abs(T2d))), 1e-300))
if _dT > 0.05:
    raise SystemExit(f"TSI temperature differs from scatra by {_dT:.3f} relative")

_gid_xy = {}
_deck_u_txt = Path(DECK_U).read_text(errors="ignore")
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
    raise SystemExit("run U wrote no monitor_dbc.yaml")
_ax = 1 if IFACE in ("left", "right") else 0
_coord = [float(nodes[n - 1][_ax]) for n in interior]
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else (Y1 - Y0)
_share = _h * float(TZ)
q_t = []
for n in interior:
    _xy = (round(float(nodes[n - 1][0]), 9), round(float(nodes[n - 1][1]), 9))
    if _xy not in _F:
        raise SystemExit(f"no reaction file for interface node at {_xy}")
    q_t.append([_F[_xy][0] / _share, _F[_xy][1] / _share])
co = [[float(nodes[n - 1][0]), float(nodes[n - 1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# ── EXPORT SELF-CHECK ─
_chk = np.asarray(Q, float)
if not np.isfinite(_chk).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface fluxes")
_chk_imp = imp if imp else {}
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                            for _d in _chk_imp.values()]) if _chk_imp else np.zeros(0))
if _chk_qin.size == _chk.size and _chk.size and np.array_equal(_chk.ravel(), -_chk_qin):
    raise SystemExit("EXPORT SELF-CHECK: exported fluxes are bit-for-bit negation of import")
if np.abs(_chk[:, 1:]).max() == 0.0 and np.abs(U2d).max() > 0:
    raise SystemExit("EXPORT SELF-CHECK: zero traction against nonzero displacement")

json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))

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

print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
