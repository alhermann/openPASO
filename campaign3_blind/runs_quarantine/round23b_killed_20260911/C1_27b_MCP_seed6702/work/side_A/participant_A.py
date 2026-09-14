#!/usr/bin/env python3
"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
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

# Read config
CFG = json.loads(Path("config.json").read_text())
NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = (float(CFG["x0"]), float(CFG["x1"]),
                  float(CFG["y0"]), float(CFG["y1"]))
KV = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IF = CFG.get("iface", "right")
SIDE = CFG.get("side", "dirichlet")
if SIDE != "dirichlet":
    raise SystemExit("this contract is the DIRICHLET role of a thermo-elastic exchange")

# Material conversions from Lame parameters
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)

# Sources as 4C expressions (^ for powers, lowercase pi)
SRC_T = str(CFG.get("source_T", "0.0")).replace("**", "^")
SRC_UX = str(CFG.get("source_ux", "0.0")).replace("**", "^")
SRC_UY = str(CFG.get("source_uy", "0.0")).replace("**", "^")

# 4C binary path
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
FOURC_LD = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")

# Read imports
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")


def partner_values(y_or_x):
    """Interpolate partner's (T, ux, uy) at one of THIS side's interface points."""
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


def why_4c_did_not_finish(tag=""):
    """Diagnose when 4C run fails."""
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
            _txt = open(_deck, errors="ignore").read()
            _secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
            _dup = sorted({x for x in _secs if _secs.count(x) > 1})
            if _dup:
                _why.append(f"{_deck}: section(s) written twice: " + ", ".join(_dup))
            if "Thermo_Structure_Interaction" in _txt:
                if "CLONING MATERIAL MAP" not in _txt:
                    _why.append(f"{_deck}: TSI needs CLONING MATERIAL MAP")
                if "COUPVARIABLE" not in _txt or "Temperature" not in _txt.split("COUPVARIABLE", 1)[-1][:40]:
                    _why.append(f"{_deck}: TSI needs COUPVARIABLE Temperature")
            if "Scalar_Transport" in _txt:
                if "CALCFLUX_BOUNDARY" not in _txt or "FLUX CALC" not in _txt:
                    _why.append(f"{_deck}: needs CALCFLUX_BOUNDARY and SCATRA FLUX CALC LINE CONDITIONS")
    except Exception as _e:
        _why.append(f"(diagnosis failed: {_e!r})")
    if not _why:
        _why.append("no 4C error line in any *.log here")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


atexit.register(_diagnose_at_exit)

# ── HOLE 1: Build the structured grid mesh ─────────────────────────────────
dx = (X1 - X0) / NX
dy = abs(Y1 - Y0) / NY

# Build nodes list (row-major: vary y first, then x)
nodes = []
for j in range(NY + 1):
    y = Y0 + j * (Y1 - Y0) / NY
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        nodes.append((x, y))

N2 = len(nodes)

# Build quads (counter-clockwise: bottom-left, bottom-right, top-right, top-left)
quads = []
quad_id = 1
for j in range(NY):
    for i in range(NX):
        # Node indices (1-based for 4C)
        bl = j * (NX + 1) + i + 1           # bottom-left
        br = j * (NX + 1) + i + 2           # bottom-right  
        tr = (j + 1) * (NX + 1) + i + 2     # top-right
        tl = (j + 1) * (NX + 1) + i + 1     # top-left
        quads.append((bl, br, tr, tl))
        quad_id += 1

# ── INTERFACE HANDSHAKE ────────────────────────────────────────────────────
_TOL = 1e-9 * max(X1 - X0, Y1 - Y0)
_AX, _VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]

def _on_edge(n):
    x, y = nodes[n - 1]
    return abs(x - X0) < _TOL or abs(x - X1) < _TOL or abs(y - Y0) < _TOL or abs(y - Y1) < _TOL

iface_all = sorted([n for n in range(1, N2 + 1) if abs(nodes[n - 1][_AX] - _VAL) < _TOL],
                   key=lambda n: nodes[n - 1][1 - _AX])

if len(iface_all) < 3:
    raise SystemExit(f"only {len(iface_all)} node(s) lie on the {IF} edge")

interior = iface_all[1:-1]  # exclude endpoints (they are outer boundary)
outer = [n for n in range(1, N2 + 1) if _on_edge(n) and n not in set(interior)]

TZ = min((X1 - X0) / NX, abs(Y1 - Y0) / NY)  # slab thickness

def gid3(n, layer):
    """4C node id of 2-D node n on slab layer 0 or 1."""
    return n + layer * N2

_g = {n: partner_values(nodes[n - 1][1 - _AX]) for n in interior}


def scatra_point_conditions():
    """Deck T: DESIGN POINT DIRICH CONDITIONS for interface nodes."""
    cond, topo = [], []
    for d, n in enumerate(interior, 1):
        cond.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{_g[n][0]:.17g}]\n    FUNCT: [0]\n")
        topo.append(f'  - "NODE {n} DNODE {d}"\n')
    return "".join(cond), "".join(topo)


def slab_point_conditions():
    """Deck U: POINT DIRICH for both layers of each interior node."""
    struct, thermo, topo = [], [], []
    d = 0
    for n in interior:
        T, ux, uy = _g[n]
        for layer in (0, 1):
            d += 1
            struct.append(f"  - E: {d}\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [{ux:.17g}, {uy:.17g}, 0.0]\n"
                          f"    FUNCT: [0, 0, 0]\n    TAG: monitor_reaction\n")
            thermo.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{T:.17g}]\n    FUNCT: [0]\n")
            topo.append(f'  - "NODE {gid3(n, layer)} DNODE {d}"\n')
    return "".join(struct), "".join(thermo), "".join(topo)


def scatra_topology():
    """Deck T: DLINE and DSURF topology."""
    dline_outer = "".join(f'  - "NODE {n} DLINE 1"\n' for n in outer)
    dline_iface = "".join(f'  - "NODE {n} DLINE 2"\n' for n in iface_all)
    dsurf = "".join(f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, N2 + 1))
    return dline_outer, dline_iface, dsurf


def slab_topology():
    """Deck U: DSURF and DVOL topology."""
    dsurf = "".join(f'  - "NODE {gid3(n, layer)} DSURFACE 1"\n' for layer in (0, 1) for n in outer)
    dvol = "".join(f'  - "NODE {n} DVOL 1"\n' for n in range(1, 2 * N2 + 1))
    return dsurf, dvol


def deck_T_tables():
    """Deck T's TABLES section."""
    cond, dn = scatra_point_conditions()
    dline_outer, dline_iface, dsurf = scatra_topology()
    
    out = "DESIGN POINT DIRICH CONDITIONS:\n" + cond
    out += "NODE COORDS:\n" + "".join(f'  - "NODE {n} COORD {x:.15f} {y:.15f} 0.0"\n' 
                                       for n, (x, y) in enumerate(nodes, 1))
    out += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e} TRANSP QUAD4 {a} {b} {c} {d} MAT 1 TYPE Std"\n'
                                             for e, (a, b, c, d) in enumerate(quads, 1))
    out += "DLINE-NODE TOPOLOGY:\n" + dline_outer + dline_iface
    out += "DNODE-NODE TOPOLOGY:\n" + dn
    out += "DSURF-NODE TOPOLOGY:\n" + dsurf
    return out


def deck_U_tables():
    """Deck U's TABLES section."""
    struct, thermo, dn = slab_point_conditions()
    dsurf, dvol = slab_topology()
    
    out = "DESIGN POINT DIRICH CONDITIONS:\n" + struct
    out += "DESIGN POINT THERMO DIRICH CONDITIONS:\n" + thermo
    out += "NODE COORDS:\n"
    for n, (x, y) in enumerate(nodes, 1):
        out += f'  - "NODE {gid3(n, 0)} COORD {x:.15f} {y:.15f} 0.0"\n'
        out += f'  - "NODE {gid3(n, 1)} COORD {x:.15f} {y:.15f} {TZ:.15f}"\n'
    out += "STRUCTURE ELEMENTS:\n" + "".join(
        f'  - "{e} SOLIDSCATRA HEX8 {a} {b} {c} {d} {gid3(a, 1)} {gid3(b, 1)} {gid3(c, 1)} {gid3(d, 1)} '
        f'MAT 1 KINEM linear TYPE Undefined"\n' for e, (a, b, c, d) in enumerate(quads, 1))
    out += "DNODE-NODE TOPOLOGY:\n" + dn
    out += "DSURF-NODE TOPOLOGY:\n" + dsurf
    out += "DVOL-NODE TOPOLOGY:\n" + dvol
    return out


# ── HOLE 2: Deck HEADERS and RUNS ─────────────────────────────────────────
HEADER_T = f'''TITLE:
  - "Scalar Transport for Temperature"
PROBLEM SIZE:
  DIM: 2
  ELEMENTS: {len(quads)}
  NODES: {N2}
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
      DIFFUSIVITY: {KV}
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN SURF NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
SCATRA FLUX CALC LINE CONDITIONS:
  - E: 2
'''

# FIX: Use correct section names for IO/RUNTIME VTK OUTPUT/STRUCTURE and THERMAL
HEADER_U = f'''TITLE:
  - "Thermo Structure Interaction for Displacement"
PROBLEM SIZE:
  DIM: 3
  ELEMENTS: {len(quads)}
  NODES: {2 * N2}
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
IO:
  STRUCT_STRESS: "No"
  STRUCT_STRAIN: "No"
IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
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
      YOUNGNUM: 1
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
        constant: [{KV}]
CLONING MATERIAL MAP:
  - SRC_FIELD: "structure"
    SRC_MAT: 1
    TAR_FIELD: "thermo"
    TAR_MAT: 2
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"
FUNCT2:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"
FUNCT3:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1.0, 1.0, 0.0]
    FUNCT: [1, 2, 0]
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [3]
DESIGN VOL DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
'''

# Write decks
DECK_T = HEADER_T + "\n" + deck_T_tables()
DECK_U = HEADER_U + "\n" + deck_U_tables()

Path("deck_T.4C.yaml").write_text(DECK_T)
Path("deck_U.4C.yaml").write_text(DECK_U)

OUT_T = "out_T"
OUT_U = "out_U"
DECK_U_FILE = "deck_U.4C.yaml"

# Run both decks
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = FOURC_LD

cmd_T = ["stdbuf", "-oL", "-eL", FOURC_BIN, "deck_T.4C.yaml", OUT_T]
cmd_U = ["stdbuf", "-oL", "-eL", FOURC_BIN, "deck_U.4C.yaml", OUT_U]

print(f"Running deck T with command: {' '.join(cmd_T)}", flush=True)
result_T = subprocess.run(cmd_T, env=env, capture_output=True, text=True)
with open("deck_T.log", "w") as f:
    f.write(result_T.stdout + result_T.stderr)
if result_T.returncode != 0:
    print(f"Deck T failed with code {result_T.returncode}", flush=True)
    print(result_T.stderr, flush=True)

print(f"Running deck U with command: {' '.join(cmd_U)}", flush=True)
result_U = subprocess.run(cmd_U, env=env, capture_output=True, text=True)
with open("deck_U.log", "w") as f:
    f.write(result_U.stdout + result_U.stderr)
if result_U.returncode != 0:
    print(f"Deck U failed with code {result_U.returncode}", flush=True)
    print(result_U.stderr, flush=True)

# ── RECOVERY FROM 4C OUTPUTS ──────────────────────────────────────────────
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
    """Field values on the 2-D node layout."""
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


# Recover temperature field and flux
vtu_T = _latest(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None:
    raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]

fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("run T wrote no flux_boundary field")
FB = _nodal(mT, fbname, 2)

nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IF]
q_n = [float(FB[n - 1, 0] * nrm[0] + FB[n - 1, 1] * nrm[1]) for n in interior]

# Recover displacement field
vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None:
    raise SystemExit(why_4c_did_not_finish("run U"))
U2d = _nodal(meshio.read(vtu_U), "displacement", 2, zlayer=0.0)
Ttsi = _nodal(meshio.read(vtu_UT), "temperature", 1, zlayer=0.0)[:, 0]

# Check T consistency between runs
_dT = float(np.max(np.abs(Ttsi - T2d)) / max(float(np.max(np.abs(T2d))), 1e-300))
if _dT > 0.05:
    raise SystemExit(f"TSI temperature differs from scatra by {_dT:.3f} relative")

# Recover reactions -> traction
# The yaml files have node gids that are 0-based, while deck NODE ids are 1-based
# Each interface node has TWO conditions (one per slab layer), so we need to sum forces from both
_gid_xy = {}
_deck_u_txt = Path(DECK_U_FILE).read_text(errors="ignore")
for _ln in _deck_u_txt.splitlines():
    _m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', _ln)
    if _m:
        # gid in yaml is 0-based, deck node id is 1-based
        _deck_node_id = int(_m.group(1))
        _xy = (round(float(_m.group(2)), 9), round(float(_m.group(3)), 9))
        # Store mapping from 0-based gid to xy
        _gid_xy[_deck_node_id - 1] = _xy  # yaml uses 0-based gids

_F = {}  # keyed by (x, y) rounded
for _yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    _txt = Path(_yf).read_text(errors="ignore")
    # Extract node gids (0-based in yaml)
    _gids_section = _txt.split("dbc monitor condition data", 1)[0]
    _gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", _gids_section, re.M)]
    # Extract force vector (3 components: fx, fy, fz)
    _fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", _txt, re.M)
    if not (_gids and _fs):
        continue
    fx, fy, fz = float(_fs[-1][0]), float(_fs[-1][1]), float(_fs[-1][2])
    # Get xy from gid (yaml gid is 0-based, matches our _gid_xy keys)
    _xy = _gid_xy.get(_gids[0])
    if _xy is None:
        continue
    _acc = _F.setdefault(_xy, [0.0, 0.0])
    _acc[0] += fx
    _acc[1] += fy

if not _F:
    raise SystemExit("run U wrote no usable monitor_dbc.yaml files")

# Compute share (tributary area for each interface node)
# For vertical interface (left/right), use y-spacing; for horizontal (top/bottom), use x-spacing
if IF in ("left", "right"):
    _coord = [float(nodes[n - 1][1]) for n in interior]  # y-coordinates
else:
    _coord = [float(nodes[n - 1][0]) for n in interior]  # x-coordinates
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else abs(Y1 - Y0)
_share = _h * float(TZ)

if _share <= 0:
    raise SystemExit(f"Zero share computed: h={_h}, TZ={TZ}")

q_t = []
for n in interior:
    _xy = (round(float(nodes[n - 1][0]), 9), round(float(nodes[n - 1][1]), 9))
    if _xy not in _F:
        raise SystemExit(f"no reaction file for interface node at {_xy}")
    q_t.append([_F[_xy][0] / _share, _F[_xy][1] / _share])

co = [[float(nodes[n - 1][0]), float(nodes[n - 1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# Export self-check
_chk = np.asarray(Q, float)
if not np.isfinite(_chk).all():
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface fluxes")

# Write exports.json LAST
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))

# Per-level persistence
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

# Print NDOF line
print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
