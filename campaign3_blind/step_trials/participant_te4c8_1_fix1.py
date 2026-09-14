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

CFG = json.loads(Path("config.json").read_text())
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
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)
SRC_T = str(CFG.get("source_T", "0.0")).replace("**", "^")
SRC_UX = str(CFG.get("source_ux", "0.0")).replace("**", "^")
SRC_UY = str(CFG.get("source_uy", "0.0")).replace("**", "^")

imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text() or "{}")


def partner_values(y_or_x):
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

h_x = (X1 - X0) / NX
h_y = (Y1 - Y0) / NY
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * h_x
        y = Y0 + j * h_y
        nodes.append((x, y))
if IF in ("left", "right"):
    iface_nodes = [j * (NX + 1) + (NX + 1) if IF == "right" else j * (NX + 1) + 1 for j in range(NY + 1)]
else:
    iface_nodes = [j * (NX + 1) + 1 + i * (NX + 1) for i in range(NX + 1)]
interior = [iface_nodes[i] for i in range(1, len(iface_nodes) - 1)]
TZ = h_y
OUT_T = "run_T"
OUT_U = "run_U"
DECK_T = "deck_T.4C.yaml"
DECK_U = "deck_U.4C.yaml"

# Identify outer boundary nodes (not on interface)
outer_nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        idx = j * (NX + 1) + i + 1
        is_outer = False
        if j == 0 or j == NY:
            is_outer = True
        if IF == "right" and i == 0:
            is_outer = True
        if IF == "left" and i == NX:
            is_outer = True
        if is_outer:
            outer_nodes.append(idx)

with open(DECK_T, "w") as f:
    f.write('TITLE:\n')
    f.write('  - "4C thermo-elastic Dirichlet side - temperature run"\n')
    f.write('PROBLEM SIZE:\n')
    f.write(f'  ELEMENTS: {NX * NY}\n')
    f.write(f'  NODES: {(NX + 1) * (NY + 1)}\n')
    f.write('PROBLEM TYPE:\n')
    f.write('  PROBLEMTYPE: "Scalar_Transport"\n')
    f.write('SCALAR TRANSPORT DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n')
    f.write('  TIMESTEP: 1.0\n')
    f.write('  MAXTIME: 1.0\n')
    f.write('  LINEAR_SOLVER: 1\n')
    f.write('  CALCFLUX_BOUNDARY: "diffusive"\n')
    f.write('SOLVER 1:\n')
    f.write('  SOLVER: "UMFPACK"\n')
    f.write('MATERIALS:\n')
    f.write('  - MAT: 1\n')
    f.write(f'    MAT_scatra:\n')
    f.write(f'      DIFFUSIVITY: {KV}\n')
    f.write('FUNCT1:\n')
    f.write(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\n')
    # Outer boundary Dirichlet (single E id for all outer nodes)
    f.write('DESIGN LINE THERMO DIRICH CONDITIONS:\n')
    f.write('  - E: 1\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [0.0]\n')
    f.write('    FUNCT: [0]\n')
    # Interface point Dirichlet (one E id per interior interface node)
    f.write('DESIGN POINT THERMO DIRICH CONDITIONS:\n')
    for k, n in enumerate(interior, 1):
        T_int, _, _ = partner_values(nodes[n - 1][1] if IF in ("left", "right") else nodes[n - 1][0])
        f.write(f'  - E: {k + 10}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write(f'    VAL: [{T_int}]\n')
        f.write('    FUNCT: [0]\n')
    # Volume source
    f.write('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n')
    f.write('  - E: 99\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [1.0]\n')
    f.write('    FUNCT: [1]\n')
    f.write('NODE COORDS:\n')
    for j in range(NY + 1):
        for i in range(NX + 1):
            idx = j * (NX + 1) + i + 1
            x = X0 + i * h_x
            y = Y0 + j * h_y
            f.write(f'  - "NODE {idx} COORD {x:.10e} {y:.10e} 0.0"\n')
    f.write('TRANSPORT ELEMENTS:\n')
    for j in range(NY):
        for i in range(NX):
            e = i + j * NX + 1
            n1 = j * (NX + 1) + i + 1
            n2 = j * (NX + 1) + i + 2
            n3 = (j + 1) * (NX + 1) + i + 2
            n4 = (j + 1) * (NX + 1) + i + 1
            f.write(f'  - "{e} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n')
    # Topology: outer boundary as DLINE 1
    f.write('DLINE-NODE TOPOLOGY:\n')
    for n in outer_nodes:
        f.write(f'  - "NODE {n} DLINE 1"\n')
    # Interface line for flux calculation
    f.write('DLINE-NODE TOPOLOGY:\n')
    for n in iface_nodes:
        f.write(f'  - "NODE {n} DLINE 2"\n')
    # Point Dirichlet topology
    f.write('DNODE-NODE TOPOLOGY:\n')
    for k, n in enumerate(interior, 1):
        f.write(f'  - "NODE {n} DNODE {k + 10}"\n')
    # Volume source topology
    f.write('DSURF-NODE TOPOLOGY:\n')
    for j in range(NY + 1):
        for i in range(NX + 1):
            idx = j * (NX + 1) + i + 1
            f.write(f'  - "NODE {idx} DSURFACE 99"\n')
    # Flux calc line condition
    f.write('SCATRA FLUX CALC LINE CONDITIONS:\n')
    f.write('  - E: 2\n')

with open(DECK_U, "w") as f:
    f.write('TITLE:\n')
    f.write('  - "4C thermo-elastic Dirichlet side - structure run"\n')
    f.write('PROBLEM SIZE:\n')
    f.write(f'  ELEMENTS: {NX * NY}\n')
    f.write(f'  NODES: {2 * (NX + 1) * (NY + 1)}\n')
    f.write('PROBLEM TYPE:\n')
    f.write('  PROBLEMTYPE: "Thermo_Structure_Interaction"\n')
    f.write('THERMO STRUCTURE INTERACTION DYNAMIC:\n')
    f.write('  COUPALGO: "tsi_oneway"\n')
    f.write('  COUPVARIABLE: "Temperature"\n')
    f.write('STRUCTURAL DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n')
    f.write('  TIMESTEP: 1.0\n')
    f.write('  MAXTIME: 1.0\n')
    f.write('  LINEAR_SOLVER: 1\n')
    f.write('THERMAL DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n')
    f.write('  TIMESTEP: 1.0\n')
    f.write('  MAXTIME: 1.0\n')
    f.write('  LINEAR_SOLVER: 1\n')
    f.write('SOLVER 1:\n')
    f.write('  SOLVER: "UMFPACK"\n')
    f.write('MATERIALS:\n')
    f.write('  - MAT: 1\n')
    f.write('    MAT_Struct_ThermoStVenantK:\n')
    f.write(f'      YOUNG_MODULUS: {E_MOD}\n')
    f.write(f'      POISSON_RATIO: {NU}\n')
    f.write(f'      LINEAR_EXPANSION_COEFFICIENT: {ALPHA}\n')
    f.write(f'      INITTEMP: 0.0\n')
    f.write('  - MAT: 2\n')
    f.write('    MAT_Fourier:\n')
    f.write(f'      DIFFUSIVITY: {KV}\n')
    f.write('CLONING MATERIAL MAP:\n')
    f.write('  - STRUCTURE_MATERIAL: 1\n')
    f.write('    THERMAL_MATERIAL: 2\n')
    f.write('FUNCT1:\n')
    f.write(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"\n')
    f.write('FUNCT2:\n')
    f.write(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"\n')
    f.write('FUNCT3:\n')
    f.write(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\n')
    # Outer boundary structure Dirichlet (single E id for all outer nodes, both layers)
    f.write('DESIGN LINE STRUCTURE DIRICH CONDITIONS:\n')
    f.write('  - E: 1\n')
    f.write('    NUMDOF: 3\n')
    f.write('    ONOFF: [1, 1, 1]\n')
    f.write('    VAL: [0.0, 0.0, 0.0]\n')
    f.write('    FUNCT: [0, 0, 0]\n')
    # Interface point structure Dirichlet (one E id per interior interface node pair)
    f.write('DESIGN POINT STRUCTURE DIRICH CONDITIONS:\n')
    for k, n in enumerate(interior, 1):
        _, ux_int, uy_int = partner_values(nodes[n - 1][1] if IF in ("left", "right") else nodes[n - 1][0])
        idx0 = n
        idx1 = n + (NX + 1) * (NY + 1)
        eid = k + 10
        f.write(f'  - E: {eid}\n')
        f.write('    NUMDOF: 3\n')
        f.write('    ONOFF: [1, 1, 1]\n')
        f.write(f'    VAL: [{ux_int}, {uy_int}, 0.0]\n')
        f.write('    FUNCT: [0, 0, 0]\n')
        f.write(f'    TAG: monitor_reaction\n')
        f.write(f'  - E: {eid + 100}\n')
        f.write('    NUMDOF: 3\n')
        f.write('    ONOFF: [1, 1, 1]\n')
        f.write(f'    VAL: [{ux_int}, {uy_int}, 0.0]\n')
        f.write('    FUNCT: [0, 0, 0]\n')
        f.write(f'    TAG: monitor_reaction\n')
    # Outer boundary thermal Dirichlet
    f.write('DESIGN LINE THERMO DIRICH CONDITIONS:\n')
    f.write('  - E: 1\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [0.0]\n')
    f.write('    FUNCT: [0]\n')
    # Interface point thermal Dirichlet
    f.write('DESIGN POINT THERMO DIRICH CONDITIONS:\n')
    for k, n in enumerate(interior, 1):
        T_int, _, _ = partner_values(nodes[n - 1][1] if IF in ("left", "right") else nodes[n - 1][0])
        idx0 = n
        idx1 = n + (NX + 1) * (NY + 1)
        eid = k + 10
        f.write(f'  - E: {eid}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write(f'    VAL: [{T_int}]\n')
        f.write('    FUNCT: [0]\n')
        f.write(f'  - E: {eid + 100}\n')
        f.write('    NUMDOF: 1\n')
        f.write('    ONOFF: [1]\n')
        f.write(f'    VAL: [{T_int}]\n')
        f.write('    FUNCT: [0]\n')
    # Volume source thermal
    f.write('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n')
    f.write('  - E: 99\n')
    f.write('    NUMDOF: 1\n')
    f.write('    ONOFF: [1]\n')
    f.write('    VAL: [1.0]\n')
    f.write('    FUNCT: [3]\n')
    # Volume source structural
    f.write('DESIGN SURF STRUCTURE NEUMANN CONDITIONS:\n')
    f.write('  - E: 99\n')
    f.write('    NUMDOF: 3\n')
    f.write('    ONOFF: [1, 1, 1]\n')
    f.write('    VAL: [1.0, 1.0, 0.0]\n')
    f.write('    FUNCT: [1, 2, 0]\n')
    f.write('NODE COORDS:\n')
    for j in range(NY + 1):
        for i in range(NX + 1):
            idx = j * (NX + 1) + i + 1
            x = X0 + i * h_x
            y = Y0 + j * h_y
            f.write(f'  - "NODE {idx} COORD {x:.10e} {y:.10e} 0.0"\n')
            f.write(f'  - "NODE {idx + (NX + 1) * (NY + 1)} COORD {x:.10e} {y:.10e} {TZ}"\n')
    # Topology: outer boundary DLINE 1 (both layers)
    f.write('DLINE-NODE TOPOLOGY:\n')
    for n in outer_nodes:
        f.write(f'  - "NODE {n} DLINE 1"\n')
        f.write(f'  - "NODE {n + (NX + 1) * (NY + 1)} DLINE 1"\n')
    # Interface point topology (layer 0)
    f.write('DNODE-NODE TOPOLOGY:\n')
    for k, n in enumerate(interior, 1):
        eid = k + 10
        f.write(f'  - "NODE {n} DNODE {eid}"\n')
        f.write(f'  - "NODE {n + (NX + 1) * (NY + 1)} DNODE {eid}"\n')
    # Interface point topology (layer 1)
    for k, n in enumerate(interior, 1):
        eid = k + 10 + 100
        f.write(f'  - "NODE {n + (NX + 1) * (NY + 1)} DNODE {eid}"\n')
    # Volume source topology (both layers)
    f.write('DSURF-NODE TOPOLOGY:\n')
    for j in range(NY + 1):
        for i in range(NX + 1):
            idx = j * (NX + 1) + i + 1
            f.write(f'  - "NODE {idx} DSURFACE 99"\n')
            f.write(f'  - "NODE {idx + (NX + 1) * (NY + 1)} DSURFACE 99"\n')
    f.write('SOLIDSCATRA ELEMENTS:\n')
    for j in range(NY):
        for i in range(NX):
            e = i + j * NX + 1
            n1 = j * (NX + 1) + i + 1
            n2 = j * (NX + 1) + i + 2
            n3 = (j + 1) * (NX + 1) + i + 2
            n4 = (j + 1) * (NX + 1) + i + 1
            n5 = n1 + (NX + 1) * (NY + 1)
            n6 = n2 + (NX + 1) * (NY + 1)
            n7 = n3 + (NX + 1) * (NY + 1)
            n8 = n4 + (NX + 1) * (NY + 1)
            f.write(f'  - "{e} SOLIDSCATRA HEX8 {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8} MAT 1 TYPE Std"\n')
    f.write('IO/MONITOR STRUCTURE DBC:\n')
    f.write('  INTERVAL_STEPS: 1\n')
    f.write('  FILE_TYPE: yaml\n')
    f.write('  WRITE_CONDITION_INFORMATION: true\n')

env = os.environ.copy()
env['LD_LIBRARY_PATH'] = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")
bin_path = CFG['fourc_bin']
with open(f"{DECK_T}.log", "w") as log_T:
    result_T = subprocess.run(["stdbuf", "-oL", "-eL", bin_path, DECK_T, OUT_T], stdout=log_T, stderr=log_T, env=env)
with open(f"{DECK_U}.log", "w") as log_U:
    result_U = subprocess.run(["stdbuf", "-oL", "-eL", bin_path, DECK_U, OUT_U], stdout=log_U, stderr=log_U, env=env)

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
_dT = float(np.max(np.abs(Ttsi - T2d)) / max(float(np.max(np.abs(T2d))), 1e-300))
if _dT > 0.05:
    raise SystemExit(f"the TSI deck's temperature differs from the scatra deck's by {_dT:.3f} relative: "
                     f"the two decks do not solve the same heat problem (check FUNCT3 = the heat source, "
                     f"the POINT THERMO DIRICH values and the outer SURF THERMO DIRICH of run U)")

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
