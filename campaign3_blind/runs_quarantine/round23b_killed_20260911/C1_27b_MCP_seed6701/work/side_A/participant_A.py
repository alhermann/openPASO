#!/usr/bin/env python3
"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
The traction uses the heat flux's sign rule (minus the flux through n_out): the two sides' exports
cancel and the Neumann partner applies them UNCHANGED. The task's OUTWARD traction is MINUS (qx, qy).
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
    raise SystemExit("this contract is the DIRICHLET role of a thermo-elastic "
                     "exchange (T, ux, uy in; qn, qx, qy out).")

# Material conversions from Lame parameters
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)

# Source terms for subdomain A (already converted to 4C syntax with ^ and lowercase pi)
SRC_T = "-6*x^3*y + 8*x^3/5 - x^2*y/2 - 2*x^2/3 - 6*x*y^3 + 24*x*y^2/5 + 67*x*y/10 - 11*x/15 - y^3/6 - 2*y^2/3 + 5*y/6"
SRC_UX = "-33*x^2*y^3/5 - 312*x^2*y^2/25 + 453*x^2*y/25 - 18*x^2/5 + 599*x*y^3/10 + 4754*x*y^2/75 - 17597*x*y/150 + 112*x/5 - 84*y^5/25 - 147*y^4/25 + 18277*y^3/300 + 893*y^2/30 - 1549*y/20 + 15"
SRC_UY = "3*x^3*y^2 - 8*x^3*y/5 - x^3/5 + 2693*x^2*y^2/20 + 7106*x^2*y/75 - 26333*x^2/300 - 12*x*y^4 - 84*x*y^3/5 + 4241*x*y^2/20 + 301*x*y/3 - 2173*x/20 + 56*y^4/15 + 392*y^3/75 - 364*y^2/25 + 28*y/5"

# 4C binary path
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")


def partner_values(y_or_x):
    """The partner's (T, ux, uy) at one of THIS side's interface points."""
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


# ── DID 4C FINISH? ─
def why_4c_did_not_finish(tag=""):
    """When a run leaves no usable output: 4C's own error line, then the deck defects."""
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
                    _why.append(f"{_deck}: TSI needs a CLONING MATERIAL MAP")
                if "COUPVARIABLE" not in _txt or "Temperature" not in _txt.split("COUPVARIABLE", 1)[-1][:40]:
                    _why.append(f"{_deck}: TSI DYNAMIC/PARTITIONED needs COUPVARIABLE \"Temperature\"")
                if re.search(r"\b(WALL|SOLID) QUAD4\b|\bTRI3\b", _txt):
                    _why.append(f"{_deck}: use SOLIDSCATRA HEX8 slab, not 2-D elements")
                if "monitor_reaction" in _txt and "IO/MONITOR STRUCTURE DBC" not in _txt:
                    _why.append(f"{_deck}: TAG: monitor_reaction needs IO/MONITOR STRUCTURE DBC section")
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

# ── HOLE 1: Build the 2-D structured mesh ─────────────────────────────────────
# nodes: list of (x, y) tuples; 4C node id = list index + 1
# quads: list of 4-tuples of node ids, counter-clockwise
dx = (X1 - X0) / NX
dy = (Y1 - Y0) / NY
nodes = []
for j in range(NY + 1):
    y = Y0 + j * dy
    for i in range(NX + 1):
        x = X0 + i * dx
        nodes.append((x, y))

quads = []
for j in range(NY):
    for i in range(NX):
        # Counter-clockwise: (i,j), (i+1,j), (i+1,j+1), (i,j+1)
        n00 = i + j * (NX + 1) + 1
        n10 = i + 1 + j * (NX + 1) + 1
        n11 = i + 1 + (j + 1) * (NX + 1) + 1
        n01 = i + (j + 1) * (NX + 1) + 1
        quads.append((n00, n10, n11, n01))

N2 = len(nodes)
_TOL = 1e-9 * max(X1 - X0, Y1 - Y0)
_AX, _VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]

def _on_edge(n):
    x, y = nodes[n - 1]
    return abs(x - X0) < _TOL or abs(x - X1) < _TOL or abs(y - Y0) < _TOL or abs(y - Y1) < _TOL

iface_all = sorted([n for n in range(1, N2 + 1) if abs(nodes[n - 1][_AX] - _VAL) < _TOL],
                   key=lambda n: nodes[n - 1][1 - _AX])
interior = iface_all[1:-1]
outer = [n for n in range(1, N2 + 1) if _on_edge(n) and n not in set(interior)]

TZ = min((X1 - X0) / NX, (Y1 - Y0) / NY)

def gid3(n, layer):
    return n + layer * N2

_g = {n: partner_values(nodes[n - 1][1 - _AX]) for n in interior}


def scatra_point_conditions():
    cond, topo = [], []
    for d, n in enumerate(interior, 1):
        cond.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{_g[n][0]:.17g}]\n    FUNCT: [0]\n")
        topo.append(f'  - "NODE {n} DNODE {d}"\n')
    return "".join(cond), "".join(topo)


def slab_point_conditions():
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
    dline = "".join(f'  - "NODE {n} DLINE 1"\n' for n in outer) + "".join(f'  - "NODE {n} DLINE 2"\n' for n in iface_all)
    dsurf = "".join(f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, N2 + 1))
    return dline, dsurf


def slab_topology():
    dsurf = "".join(f'  - "NODE {gid3(n, layer)} DSURFACE 1"\n' for layer in (0, 1) for n in outer)
    dvol = "".join(f'  - "NODE {n} DVOL 1"\n' for n in range(1, 2 * N2 + 1))
    return dsurf, dvol


def deck_T_tables():
    cond, dn = scatra_point_conditions()
    dline, dsurf = scatra_topology()
    out = "DESIGN POINT DIRICH CONDITIONS:\n" + cond
    out += "NODE COORDS:\n" + "".join(f'  - "NODE {n} COORD {x:.15f} {y:.15f} 0.0"\n' for n, (x, y) in enumerate(nodes, 1))
    out += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e} TRANSP QUAD4 {a} {b} {c} {d} MAT 1 TYPE Std"\n'
                                             for e, (a, b, c, d) in enumerate(quads, 1))
    out += "DLINE-NODE TOPOLOGY:\n" + dline + "DNODE-NODE TOPOLOGY:\n" + dn + "DSURF-NODE TOPOLOGY:\n" + dsurf
    return out


def deck_U_tables():
    struct, thermo, dn = slab_point_conditions()
    dsurf, dvol = slab_topology()
    out = "DESIGN POINT DIRICH CONDITIONS:\n" + struct + "DESIGN POINT THERMO DIRICH CONDITIONS:\n" + thermo
    out += "NODE COORDS:\n" + "".join(f'  - "NODE {gid3(n, 0)} COORD {x:.15f} {y:.15f} 0.0"\n' for n, (x, y) in enumerate(nodes, 1))
    out += "".join(f'  - "NODE {gid3(n, 1)} COORD {x:.15f} {y:.15f} {TZ:.15f}"\n' for n, (x, y) in enumerate(nodes, 1))
    out += "STRUCTURE ELEMENTS:\n" + "".join(
        f'  - "{e} SOLIDSCATRA HEX8 {a} {b} {c} {d} {gid3(a, 1)} {gid3(b, 1)} {gid3(c, 1)} {gid3(d, 1)} '
        f'MAT 1 KINEM linear TYPE Undefined"\n' for e, (a, b, c, d) in enumerate(quads, 1))
    out += "DNODE-NODE TOPOLOGY:\n" + dn + "DSURF-NODE TOPOLOGY:\n" + dsurf + "DVOL-NODE TOPOLOGY:\n" + dvol
    return out


# ── HOLE 2: Deck headers and runs ────────────────────────────────────────────
HEADER_T = f'''TITLE:
  - "Side A Temperature Solve"
PROBLEM SIZE:
  ELEMENTS: {len(quads)}
  NODES: {N2}
PROBLEM TYPE:
  PROBLEMTYPE: "Scalar_Transport"
SCALAR TRANSPORT DYNAMIC:
  TIMEINTEGR: "Stationary"
  SOLVERTYPE: "linear_full"
  VELOCITYFIELD: "zero"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
  CALCFLUX_BOUNDARY: "diffusive"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
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

HEADER_U = f'''TITLE:
  - "Side A Thermo-Elastic Solve"
PROBLEM SIZE:
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
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  NUMSTEP: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  ITEMAX: 1
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
THERMAL DYNAMIC/RUNTIME VTK OUTPUT:
  OUTPUT_THERMO: true
  TEMPERATURE: true
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
      DENS: 1
      THEXPANS: {ALPHA}
      INITTEMP: 0
      THERMOMAT: 2
  - MAT: 2
    MAT_Fourier:
      CAPA: 1
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
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0, 0, 0]
    FUNCT: [0, 0, 0]
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0]
    FUNCT: [0]
'''

# Write decks
DECK_T = HEADER_T + "\n" + deck_T_tables()
DECK_U = HEADER_U + "\n" + deck_U_tables()

Path("deck_T.4C.yaml").write_text(DECK_T)
Path("deck_U.4C.yaml").write_text(DECK_U)

OUT_T = "out_T"
OUT_U = "out_U"

# Run deck T
cmd_T = f"stdbuf -oL -eL {FOURC_BIN} deck_T.4C.yaml {OUT_T}"
result_T = subprocess.run(cmd_T, shell=True, capture_output=True, text=True)
if result_T.returncode != 0:
    print(f"Deck T failed: {result_T.stderr}", file=sys.stderr)

# Run deck U
cmd_U = f"stdbuf -oL -eL {FOURC_BIN} deck_U.4C.yaml {OUT_U}"
result_U = subprocess.run(cmd_U, shell=True, capture_output=True, text=True)
if result_U.returncode != 0:
    print(f"Deck U failed: {result_U.stderr}", file=sys.stderr)

# ── RECOVERY FROM 4C OUTPUTS ─────────────────────────────────────────────────
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
    raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("run T wrote no flux_boundary field")
FB = _nodal(mT, fbname, 2)
nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IF]
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

# Reactions -> Traction
_gid_xy = {}
_deck_u_txt = DECK_U
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
    raise SystemExit("run U wrote no monitor_dbc.yaml files")

_ax = 1 if IF in ("left", "right") else 0
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

print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
