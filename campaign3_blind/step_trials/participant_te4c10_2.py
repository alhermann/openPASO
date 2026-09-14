"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
The traction uses the heat flux's sign rule (minus the flux through n_out): the two sides' exports
cancel and the Neumann partner applies them UNCHANGED. The task's OUTWARD traction is MINUS (qx, qy).
TWO 4C RUNS PER ITERATION (your hole, both under a second): run T, a 2-D Scalar_Transport deck with
CALCFLUX_BOUNDARY "diffusive" (4C's consistent boundary flux); run U, a Thermo_Structure_Interaction
deck on a ONE-ELEMENT-THICK SOLIDSCATRA HEX8 slab with u_z pinned (exact plane strain), tsi_oneway,
COUPVARIABLE Temperature, MAT_Struct_ThermoStVenantK (alpha = beta/(3 lambda + 2 mu), INITTEMP 0),
`TAG: monitor_reaction` on the interface point conditions: 4C writes <out>-<id>_monitor_dbc.yaml per
condition (node gid ZERO-based, force f) and (f_layer0 + f_layer1)/(h*t_z) IS the exported traction
(measured order 2.0). SERVED: the handshake, the finish diagnosis, the recovery, self-checks and
the exports. YOURS: the mesh (the 2-D layout and the slab), both decks and both runs.
config.json: {"level","nx","ny","x0","x1","y0","y1","k","lam","mu","beta","iface":"left|right|
bottom|top","source_T","source_ux","source_uy" (4C expressions: '^', lowercase pi),"fourc_bin","fourc_ld"}
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
            try:   # the grammar judgement stands on its own: no binary, no judgement, other findings kept
                # section names judged by the BINARY's own grammar (`4C -p`, read once); the closest known
                # names ride along (measured: invented names like IO/RUNTIME VTK OUTPUT/THERMO stop 4C first)
                if not globals().get("_VALID"):
                    _dump = subprocess.run([str(CFG.get("fourc_bin") or os.environ.get("FOURC_BIN", "4C")), "-p"],
                                           capture_output=True, text=True, timeout=180,
                                           env=dict(os.environ, LD_LIBRARY_PATH=f"{CFG.get('fourc_ld') or ''}:{os.environ.get('LD_LIBRARY_PATH', '')}")).stdout
                    globals()["_VALID"] = set(re.findall(r"^    - name: (.+?)\s*$", _dump, re.M)) | set(
                        re.findall(r"^  - ([A-Z][A-Z0-9 _/.:-]*?)\s*$", _dump.split("legacy_string_sections:", 1)[-1], re.M)) | {"TITLE"}
                if len(_VALID) > 100:
                    import difflib
                    _tok = lambda nm: [w for w in re.split(r"[ /_-]+", nm.upper()) if w]
                    _best = lambda a, bs: max((difflib.SequenceMatcher(None, a, b).ratio() for b in bs), default=0.0)
                    for _s in dict.fromkeys(_secs):
                        if _s not in _VALID and not re.fullmatch(r"FUNCT\d+", _s):
                            _u = _tok(_s)   # word-wise similarity both ways: THERMO finds THERMAL, SOLIDSCATRA finds STRUCTURE
                            _sc = sorted(((sum(_best(a, _tok(c)) for a in _u) + sum(_best(b, _u) for b in _tok(c))) / (len(_u) + len(_tok(c))), c)
                                         for c in _VALID if _tok(c))
                            _close = [c for v, c in _sc[::-1][:5] if v >= 0.45]
                            _why.append(f"{_deck}: section '{_s}' is not in the binary's grammar (`4C -p`)"
                                        + (f"; closest known: {', '.join(repr(c) for c in _close)}" if _close else ""))
            except Exception:                # noqa: BLE001
                pass
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
            _topo = set(re.findall(r"\b(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)", _txt))
            for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
                _head = _b.split(":", 1)[0].strip()
                if _head.startswith("DESIGN") and _head.endswith("CONDITIONS"):
                    _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head)
                    _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[_kw.group(1)] if _kw else ""
                    _missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", _b) if (_kind, x) not in _topo}, key=int)
                    if _missing:
                        _why.append(f"{_deck}: {_head} names E id(s) {', '.join(_missing[:6])} that no *-NODE TOPOLOGY section defines")
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(diagnosis failed: {_e!r})")
    if not _why:
        _why.append("no 4C error line in any *.log here -- run the binary line-buffered (stdbuf -oL -eL) with its console in a log next to the deck")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


atexit.register(_diagnose_at_exit)

# ── THE HOLE (yours): the mesh, the two decks and the two runs. OASiS serves the
#    handshake above and the recovery below; what 4C solves is YOUR deck. Build this
#    subdomain's 2-D node layout (NX x NY on [X0, X1] x [Y0, Y1]) and, for deck U, the
#    one-element-thick slab (a second node layer at z = TZ). Classify your interface
#    nodes yourself: partner_values(coord) returns the partner's (T, ux, uy) at one
#    interface coordinate, and the two interface ENDPOINTS are outer nodes that keep the
#    outer datum. Write deck T (Scalar_Transport, 2-D, with 4C's consistent boundary flux
#    on the interface line) and deck U (the TSI slab, with the partner's T and (ux, uy)
#    as point conditions on both layers and the reactions of those conditions monitored)
#    for the problem you were given: the grammar is `4C -p`,
#    prepare_simulation(solver='fourc', physics=...) and knowledge(solver='fourc').
#    Run each deck line-buffered with its console in a log
#    (stdbuf -oL -eL <bin> <deck> <prefix> > <deck>.log 2>&1); on a non-zero exit FALL
#    THROUGH, the served check reads the log. LEAVE BEHIND exactly these names:
#      nodes     the list of (x, y) of the 2-D layout; deck node id on the z = 0 layer = index + 1
#      interior  the 1-based ids of the interface nodes WITHOUT the two endpoints, in order along it
#      TZ        the slab thickness (one well-shaped HEX8 layer)
#      OUT_T, OUT_U   the two output prefixes;  DECK_U   the file name you wrote deck U to
# ─────────────────────────────────────────────────────────────────────────
# THE SOLVE ITSELF IS YOURS AND IS NOT SERVED HERE.
#
# Build the mesh, the function space, the weak form and the linear solve for
# the problem you were given, in this backend, however you judge best. That is
# ordinary finite-element work and OASiS has no business dictating it.
#
# What OASiS does document — because you cannot guess it and it is what the
# interface check compares against — is everything AROUND the solve: the
# imports/exports handshake above, the interface sign convention, and the flux
# recovery below. Those are this tool's own interface, not your method.
#
# At this point you are expected to have produced:
#   * the discrete solution on this subdomain, with the partner's interface
#     data applied according to SIDE, and
#   * the assembled operator and the VOLUME load separately, because the flux
#     recovery below subtracts the volume load alone.
# ─────────────────────────────────────────────────────────────────────────

# ── BUILD MESH ──
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes.append((x, y))
NZ = 2  # one element thick in z
TZ = (X1 - X0) / NX  # same size as h in x
# Slab nodes: layer 0 at z=0, layer 1 at z=TZ
slab_nodes = []
layer_map = {}  # (x,y,z) -> deck node id (1-based)
nid = 1
for lz in range(NZ):
    z = lz * TZ
    for (x, y) in nodes:
        slab_nodes.append((x, y, z))
        layer_map[(round(x, 9), round(y, 9), round(z, 9))] = nid
        nid += 1
NUM_NODES = len(nodes)
NUM_SLAB_NODES = len(slab_nodes)
NUM_ELEMENTS = NX * NY

# Interface nodes (on THIS side's interface, x=X1 for "right", etc.)
if IF == "right":
    iface_nodes_2d = [(X1, Y0 + j * (Y1 - Y0) / NY) for j in range(NY + 1)]
elif IF == "left":
    iface_nodes_2d = [(X0, Y0 + j * (Y1 - Y0) / NY) for j in range(NY + 1)]
elif IF == "top":
    iface_nodes_2d = [(X0 + i * (X1 - X0) / NX, Y1) for i in range(NX + 1)]
else:  # bottom
    iface_nodes_2d = [(X0 + i * (X1 - X0) / NX, Y0) for i in range(NX + 1)]
# Find their deck IDs (1-based) on layer 0
iface_ids_2d = []
for (ix, iy) in iface_nodes_2d:
    for idx, (nx, ny) in enumerate(nodes):
        if abs(nx - ix) < 1e-9 and abs(ny - iy) < 1e-9:
            iface_ids_2d.append(idx + 1)
            break
# Interior interface nodes (exclude the two endpoints that are also outer BC corners)
interior = iface_ids_2d[1:-1] if len(iface_ids_2d) > 2 else []

# Outer boundary lines/surfaces (all except the interface)
outer_dsuf_ids = []
# We'll assign DSURFACE ids sequentially: 1..k for outer, k+1 for interface
# For simplicity, just use one DSURFACE for all outer nodes
outer_surf_id = 1
interface_surf_id = 2

# ── DECK T: Scalar_Transport (2-D, steady) ──
OUT_T = "run_t"
deck_t = f'''TITLE:
  - "Subdomain A: heat equation, Dirichlet side"
PROBLEM SIZE:
  ELEMENTS: {NUM_ELEMENTS}
  NODES: {NUM_NODES}
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
'''
# Source term
if SRC_T.strip() != "0.0":
    deck_t += f'''
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
DSURF-NODE TOPOLOGY:
'''
    for nid2d in range(1, NUM_NODES + 1):
        deck_t += f'  - "NODE {nid2d} DSURFACE 1"\n'
else:
    deck_t += '''
DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DSURF-NODE TOPOLOGY:
'''
    for nid2d in range(1, NUM_NODES + 1):
        deck_t += f'  - "NODE {nid2d} DSURFACE 1"\n'

# Boundary conditions
deck_t += f'''
SCATRA FLUX CALC LINE CONDITIONS:
  - E: {interface_surf_id}
NODE COORDS:
'''
for idx, (x, y) in enumerate(nodes):
    deck_t += f'  - "NODE {idx + 1} COORD {x:.10f} {y:.10f} 0.0"\n'

deck_t += f'''
TRANSPORT ELEMENTS:
'''
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n1 = j * (NX + 1) + i + 1
        n2 = n1 + 1
        n3 = n1 + (NX + 1) + 1
        n4 = n1 + (NX + 1)
        deck_t += f'  - "{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n'
        elem_id += 1

# Outer boundary Dirichlet (T=0)
deck_t += f'''
DESIGN SURF DIRICH CONDITIONS:
  - E: {outer_surf_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DSURF-NODE TOPOLOGY:
'''
# Mark outer vs interface nodes
for nid2d in range(1, NUM_NODES + 1):
    (x, y) = nodes[nid2d - 1]
    is_iface = False
    if IF == "right" and abs(x - X1) < 1e-9:
        is_iface = True
    elif IF == "left" and abs(x - X0) < 1e-9:
        is_iface = True
    elif IF == "top" and abs(y - Y1) < 1e-9:
        is_iface = True
    elif IF == "bottom" and abs(y - Y0) < 1e-9:
        is_iface = True
    surf_id = interface_surf_id if is_iface else outer_surf_id
    deck_t += f'  - "NODE {nid2d} DSURFACE {surf_id}"\n'

# Interface Dirichlet (per-node values from partner)
deck_t += f'''
DESIGN POINT DIRICH CONDITIONS:
'''
for nid2d in iface_ids_2d:
    (x, y) = nodes[nid2d - 1]
    if IF == "right" or IF == "left":
        coord = y
    else:
        coord = x
    T_val, _, _ = partner_values(coord)
    deck_t += f'''  - E: {nid2d}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_val}]
    FUNCT: [0]
'''
deck_t += f'''
DNODE-NODE TOPOLOGY:
'''
for nid2d in iface_ids_2d:
    deck_t += f'  - "NODE {nid2d} DNODE {nid2d}"\n'

Path(f"{OUT_T}.4C.yaml").write_text(deck_t)

# ── DECK U: Thermo_Structure_Interaction (slab) ──
OUT_U = "run_u"
DECK_U = f"{OUT_U}.4C.yaml"

deck_u = f'''TITLE:
  - "Subdomain A: thermo-elasticity, Dirichlet side, plane strain slab"
PROBLEM SIZE:
  DIM: 3
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 2
THERMAL DYNAMIC:
  INITIALFIELD: "field_by_function"
  INITFUNCNO: 1
  TIMESTEP: 1.0
  MAXTIME: 1.0
  LINEAR_SOLVER: 1
TSI DYNAMIC:
  COUPALGO: "tsi_oneway"
  MAXTIME: 1.0
  TIMESTEP: 1.0
  ITEMAX: 1
TSI DYNAMIC/PARTITIONED:
  COUPVARIABLE: "Temperature"
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "Thermal_Solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Structure_Solver"
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
IO:
  STRUCT_STRESS: "Cauchy"
  STRUCT_STRAIN: "GL"
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
  STRESS_STRAIN: true
IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
'''

# Thermal source terms (body forces in thermo)
# In TSI, thermal body sources go in as DESIGN VOL THERMO NEUMANN
if SRC_T.strip() != "0.0":
    deck_u += f'''
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
DVOL-NODE TOPOLOGY:
'''
    for snid in range(1, NUM_SLAB_NODES + 1):
        deck_u += f'  - "NODE {snid} DVOL 1"\n'
else:
    deck_u += '''
FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DVOL-NODE TOPOLOGY:
'''
    for snid in range(1, NUM_SLAB_NODES + 1):
        deck_u += f'  - "NODE {snid} DVOL 1"\n'

# Mechanical body forces
if SRC_UX.strip() != "0.0" or SRC_UY.strip() != "0.0":
    deck_u += f'''
FUNCT3:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"
FUNCT4:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"
DESIGN VOL NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1.0, 1.0, 0.0]
    FUNCT: [3, 4, 0]
DVOL-NODE TOPOLOGY:
'''
    for snid in range(1, NUM_SLAB_NODES + 1):
        deck_u += f'  - "NODE {snid} DVOL 2"\n'
else:
    deck_u += '''
FUNCT3:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT4:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
DESIGN VOL NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 3
    ONOFF: [0, 0, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [3, 4, 0]
DVOL-NODE TOPOLOGY:
'''
    for snid in range(1, NUM_SLAB_NODES + 1):
        deck_u += f'  - "NODE {snid} DVOL 2"\n'

# Outer boundary Dirichlet (T=0, u=0)
# Thermal outer BC
deck_u += f'''
DESIGN SURF THERMO DIRICH CONDITIONS:
  - E: {outer_surf_id}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DSURF-NODE TOPOLOGY:
'''
# Structural outer BC (u=0)
deck_u += f'''
DESIGN SURF DIRICH CONDITIONS:
  - E: {outer_surf_id}
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
'''
# Plane strain: u_z = 0 on ALL nodes
deck_u += f'''
DESIGN VOL DIRICH CONDITIONS:
  - E: 3
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DVOL-NODE TOPOLOGY:
'''
for snid in range(1, NUM_SLAB_NODES + 1):
    deck_u += f'  - "NODE {snid} DVOL 3"\n'

# Now add DSURF-NODE TOPOLOGY for outer surfaces
# We need to identify which slab nodes are on outer boundaries
outer_surf_nodes = []
interface_surf_nodes = []
for snid in range(1, NUM_SLAB_NODES + 1):
    (x, y, z) = slab_nodes[snid - 1]
    is_outer = False
    is_iface = False
    if IF == "right":
        if abs(x - X1) < 1e-9:
            is_iface = True
        elif abs(x - X0) < 1e-9 or abs(y - Y0) < 1e-9 or abs(y - Y1) < 1e-9:
            is_outer = True
    elif IF == "left":
        if abs(x - X0) < 1e-9:
            is_iface = True
        elif abs(x - X1) < 1e-9 or abs(y - Y0) < 1e-9 or abs(y - Y1) < 1e-9:
            is_outer = True
    elif IF == "top":
        if abs(y - Y1) < 1e-9:
            is_iface = True
        elif abs(y - Y0) < 1e-9 or abs(x - X0) < 1e-9 or abs(x - X1) < 1e-9:
            is_outer = True
    else:  # bottom
        if abs(y - Y0) < 1e-9:
            is_iface = True
        elif abs(y - Y1) < 1e-9 or abs(x - X0) < 1e-9 or abs(x - X1) < 1e-9:
            is_outer = True
    
    if is_outer:
        outer_surf_nodes.append(snid)
    elif is_iface:
        interface_surf_nodes.append(snid)

for snid in outer_surf_nodes:
    deck_u += f'  - "NODE {snid} DSURFACE {outer_surf_id}"\n'

# Interface Point Dirichlet (per-node values from partner) on BOTH layers
deck_u += f'''
DESIGN POINT DIRICH CONDITIONS:
'''
# Thermal interface BC
for lid in iface_ids_2d:
    (x, y) = nodes[lid - 1]
    if IF == "right" or IF == "left":
        coord = y
    else:
        coord = x
    T_val, ux_val, uy_val = partner_values(coord)
    
    # Layer 0
    z0 = 0.0
    snid_0 = layer_map[(round(x, 9), round(y, 9), round(z0, 9))]
    deck_u += f'''  - E: {snid_0}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_val}]
    FUNCT: [0]
    TAG: monitor_reaction
'''
    # Layer 1
    z1 = TZ
    snid_1 = layer_map[(round(x, 9), round(y, 9), round(z1, 9))]
    deck_u += f'''  - E: {snid_1}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{T_val}]
    FUNCT: [0]
    TAG: monitor_reaction
'''

# Structural interface BC (ux, uy)
for lid in iface_ids_2d:
    (x, y) = nodes[lid - 1]
    if IF == "right" or IF == "left":
        coord = y
    else:
        coord = x
    _, ux_val, uy_val = partner_values(coord)
    
    # Layer 0
    z0 = 0.0
    snid_0 = layer_map[(round(x, 9), round(y, 9), round(z0, 9))]
    deck_u += f'''  - E: {snid_0}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{ux_val}, {uy_val}, 0.0]
    FUNCT: [0, 0, 0]
    TAG: monitor_reaction
'''
    # Layer 1
    z1 = TZ
    snid_1 = layer_map[(round(x, 9), round(y, 9), round(z1, 9))]
    deck_u += f'''  - E: {snid_1}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{ux_val}, {uy_val}, 0.0]
    FUNCT: [0, 0, 0]
    TAG: monitor_reaction
'''

deck_u += f'''
DNODE-NODE TOPOLOGY:
'''
for lid in iface_ids_2d:
    (x, y) = nodes[lid - 1]
    z0 = 0.0
    z1 = TZ
    snid_0 = layer_map[(round(x, 9), round(y, 9), round(z0, 9))]
    snid_1 = layer_map[(round(x, 9), round(y, 9), round(z1, 9))]
    deck_u += f'  - "NODE {snid_0} DNODE {snid_0}"\n'
    deck_u += f'  - "NODE {snid_1} DNODE {snid_1}"\n'

# Node coords for slab
deck_u += f'''
NODE COORDS:
'''
for snid, (x, y, z) in enumerate(slab_nodes, 1):
    deck_u += f'  - "NODE {snid} COORD {x:.10f} {y:.10f} {z:.10f}"\n'

# Structure elements (HEX8)
deck_u += f'''
STRUCTURE ELEMENTS:
'''
elem_id = 1
for lz in range(NZ - 1):  # only one layer in z
    for j in range(NY):
        for i in range(NX):
            # Get node indices for this hex8 element
            # Bottom face (z=lz*TZ)
            n1 = lz * NUM_NODES + j * (NX + 1) + i + 1
            n2 = n1 + 1
            n3 = n1 + (NX + 1) + 1
            n4 = n1 + (NX + 1)
            # Top face (z=(lz+1)*TZ)
            n5 = n1 + NUM_NODES
            n6 = n2 + NUM_NODES
            n7 = n3 + NUM_NODES
            n8 = n4 + NUM_NODES
            
            # Verify nodes exist
            if n8 > NUM_SLAB_NODES:
                continue
                
            deck_u += f'  - "{elem_id} SOLIDSCATRA HEX8 {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8} MAT 1 KINEM linear TYPE Undefined"\n'
            elem_id += 1

Path(DECK_U).write_text(deck_u)

# ── RUN BOTH DECKS ──
fourc_bin = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
fourc_ld = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")

env = dict(os.environ, LD_LIBRARY_PATH=f"{fourc_ld}:{os.environ.get('LD_LIBRARY_PATH', '')}")

# Run T
cmd_t = ["stdbuf", "-oL", "-eL", fourc_bin, f"{OUT_T}.4C.yaml", OUT_T]
try:
    result_t = subprocess.run(cmd_t, capture_output=True, text=True, timeout=300, env=env)
    Path(f"{OUT_T}.log").write_text(result_t.stdout + "\n" + result_t.stderr)
except Exception as e:
    Path(f"{OUT_T}.error.log").write_text(str(e))
    raise SystemExit(why_4c_did_not_finish("run T"))

# Run U
cmd_u = ["stdbuf", "-oL", "-eL", fourc_bin, f"{OUT_U}.4C.yaml", OUT_U]
try:
    result_u = subprocess.run(cmd_u, capture_output=True, text=True, timeout=300, env=env)
    Path(f"{OUT_U}.log").write_text(result_u.stdout + "\n" + result_u.stderr)
except Exception as e:
    Path(f"{OUT_U}.error.log").write_text(str(e))
    raise SystemExit(why_4c_did_not_finish("run U"))

# ── RECOVERY FROM 4C's OWN OUTPUTS (served): boundary flux VTU, displacement VTU, reaction yaml ──
import meshio  # noqa: E402

# ── YOUR DECKS, CHECKED BEFORE ANYTHING IS READ (served): 4C drops a condition whose E id no
#    topology section defines and RUNS THE WRONG PROBLEM to 'finished normally' (measured on a
#    worker deck: every boundary condition and the source gone, rc 0). A run like that is refused here.
for _dk in sorted(glob.glob("*.4C.yaml")) or [p for p in sorted(glob.glob("*.yaml")) if "monitor_dbc" not in p]:
    _txt = Path(_dk).read_text(errors="ignore")
    _topo = set(re.findall(r"\b(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)", _txt))
    _lost = []
    for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
        _head = _b.split(":", 1)[0].strip()
        if _head.startswith("DESIGN") and _head.endswith("CONDITIONS"):
            _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head)
            _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[_kw.group(1)] if _kw else ""
            _lost += [f"{_head} E {x}" for x in re.findall(r"\bE:\s*(\d+)", _b) if (_kind, x) not in _topo]
    if _lost:
        raise SystemExit(f"DECK CHECK: {_dk} puts conditions on E ids that no *-NODE TOPOLOGY section defines "
                         f"({'; '.join(_lost[:6])}): 4C dropped them silently, so the run solved a different "
                         f"problem. Add the DNODE/DLINE/DSURF/DVOL-NODE TOPOLOGY entries for those ids.")


def _latest(pattern):
    vs = sorted(glob.glob(pattern))
    if not vs:
        return None
    # <field>-<step>-<rank>.vtu: the TRAILING number is the MPI rank; the step
    # is the middle one, and step 0 is the all-zero initial state
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
# DECK_U may be the deck's file name or the deck text itself (measured: a worker
# left the assembled text in it); a third way is any TSI deck file next to us.
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
    fx, fy = float(_fs[-1][0]), float(_fs[-1][1])          # the last step
    # the yaml's node gids are ZERO-based; the deck's NODE ids are one-based
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
    q_t.append([_F[_xy][0] / _share, _F[_xy][1] / _share])   # MEASURED sign: this IS -(sigma.n_out)
co = [[float(nodes[n - 1][0]), float(nodes[n - 1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# ── EXPORT SELF-CHECK ─ keep this block ───────────────────────────────────
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
# exports.json LAST (the driver takes its existence as proof of success): the
# Dirichlet side imposed the trace, it does not own one -> values = [].
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))
# PER-LEVEL PERSISTENCE: this level's field, named by the config level, never
# overwritten by the next level. Build the task's per-level files from these.
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy\n")
    for (_px, _py), _t, (_ux, _uy) in zip(nodes, T2d, U2d):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_t):.11e},{float(_ux):.11e},{float(_uy):.11e}\n")
with open(f"interface_level{_LVL}.csv", "w") as _f:
    # the task's OUTWARD traction sigma_tot.n_out is MINUS the exported (qx, qy)
    _f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for (_px, _py), n, (_qn, _qx, _qy) in zip(co, interior, Q):
        _f.write(f"{_px:.11e},{_py:.11e},{float(T2d[n-1]):.11e},{float(U2d[n-1,0]):.11e},"
                 f"{float(U2d[n-1,1]):.11e},{_qn:.11e},{-_qx:.11e},{-_qy:.11e}\n")
# THE RUN-LOG CONTRACT LINE: `NDOF = <integer>` on a line of its own (T, ux, uy
# per node of the 2-D discretisation), then the descriptive line.
print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
