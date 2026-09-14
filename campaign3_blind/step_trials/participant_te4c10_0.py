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
SRC_T = str(CFG.get("source_T", "0.0"))
SRC_UX = str(CFG.get("source_ux", "0.0"))
SRC_UY = str(CFG.get("source_uy", "0.0"))

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

# MESH GENERATION
nodes = []
nx, ny = NX, NY
hx = (X1 - X0) / nx
hy = (Y1 - Y0) / ny
node_id = 1
for j in range(ny + 1):
    y = Y0 + j * hy
    for i in range(nx + 1):
        x = X0 + i * hx
        nodes.append((x, y))
        node_id += 1
# nodes is 0-indexed list of (x, y). 4C node id = index + 1.

# Identify interface nodes (x = X1 if iface == "right")
if IF == "right":
    iface_nodes = [i + 1 for i, (x, y) in enumerate(nodes) if abs(x - X1) < 1e-9]
    # Sort by y
    iface_nodes.sort(key=lambda n: nodes[n - 1][1])
elif IF == "left":
    iface_nodes = [i + 1 for i, (x, y) in enumerate(nodes) if abs(x - X0) < 1e-9]
    iface_nodes.sort(key=lambda n: nodes[n - 1][1])
elif IF == "top":
    iface_nodes = [i + 1 for i, (x, y) in enumerate(nodes) if abs(y - Y1) < 1e-9]
    iface_nodes.sort(key=lambda n: nodes[n - 1][0])
elif IF == "bottom":
    iface_nodes = [i + 1 for i, (x, y) in enumerate(nodes) if abs(y - Y0) < 1e-9]
    iface_nodes.sort(key=lambda n: nodes[n - 1][0])

# Interior interface nodes (exclude endpoints)
interior = iface_nodes[1:-1]

# Slab thickness (one element thick)
TZ = hy / 2.0  # Reasonable thickness

# 3D Slab Nodes (Layer 0: z=0, Layer 1: z=TZ)
# Node IDs: Layer 0 = 1..N, Layer 1 = N+1..2N
N2D = len(nodes)
# 4C expects specific ordering for HEX8.
# Bottom face (z=0) CCW: 1 2 6 5 (in local quad).
# Top face (z=TZ) CCW: 17 18 22 21.
# Global ordering: 1..N (z=0), N+1..2N (z=TZ).
# Element connectivity for element e (bottom-left node i):
# n1=i, n2=i+1, n3=i+ny+1, n4=i+ny, n5=i+N2D, n6=i+N2D+1, n7=i+N2D+ny+1, n8=i+N2D+ny

# WRITE DECK T (Scalar_Transport)
OUT_T = "out_T"
DECK_T = "deck_T.4C.yaml"
# Line IDs for topology
line_id_outer = 1
line_id_iface = 2
# Surface ID for source
surf_id_vol = 1

deck_t_lines = [
    "TITLE:",
    '  - "Side A Thermo-Elastic T Field"',
    "PROBLEM SIZE:",
    f"  ELEMENTS: {nx * ny}",
    f"  NODES: {N2D}",
    "PROBLEM TYPE:",
    '  PROBLEMTYPE: "Scalar_Transport"',
    "SCALAR TRANSPORT DYNAMIC:",
    '  TIMEINTEGR: "Stationary"',
    '  SOLVERTYPE: "linear_full"',
    "  NUMSTEP: 1",
    "  TIMESTEP: 1.0",
    "  MAXTIME: 1.0",
    "  LINEAR_SOLVER: 1",
    "SOLVER 1:",
    '  SOLVER: "UMFPACK"',
    "MATERIALS:",
    f"  - MAT: 1",
    "    MAT_scatra:",
    f"      DIFFUSIVITY: {KV}",
    "FUNCT1:",
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"',
    "DESIGN SURF TRANSPORT NEUMANN CONDITIONS:",
    f"  - E: {surf_id_vol}",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [1.0]",
    "    FUNCT: [1]",
    "DESIGN LINE TRANSPORT DIRICH CONDITIONS:",
    f"  - E: {line_id_outer}",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    "DESIGN POINT TRANSPORT DIRICH CONDITIONS:",
]
# Add interface point conditions
for n_id in iface_nodes:
    _, _, T_val = partner_values(nodes[n_id - 1][0] if IF in ("left", "right") else nodes[n_id - 1][1])
    deck_t_lines.append(f"  - E: {n_id}")
    deck_t_lines.append("    NUMDOF: 1")
    deck_t_lines.append("    ONOFF: [1]")
    deck_t_lines.append(f"    VAL: [{T_val:.15e}]")
    deck_t_lines.append("    FUNCT: [0]")

deck_t_lines.extend([
    "SCATRA FLUX CALC LINE CONDITIONS:",
    f"  - E: {line_id_iface}",
    "CALCFLUX_BOUNDARY: \"diffusive\"",
    "NODE COORDS:",
])
for i, (x, y) in enumerate(nodes):
    deck_t_lines.append(f'  - "NODE {i + 1} COORD {x:.15e} {y:.15e} 0.0"')

deck_t_lines.append("TRANSPORT ELEMENTS:")
el_id = 1
for j in range(ny):
    for i in range(nx):
        n1 = j * (nx + 1) + i + 1
        n2 = n1 + 1
        n3 = n1 + nx + 1
        n4 = n1 + nx
        deck_t_lines.append(f'  - "{el_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
        el_id += 1

deck_t_lines.append("DSURF-NODE TOPOLOGY:")
for i in range(N2D):
    deck_t_lines.append(f'  - "NODE {i + 1} DSURFACE {surf_id_vol}"')

deck_t_lines.append("DLINE-NODE TOPOLOGY:")
# Outer lines
lid = line_id_outer
if IF == "right":
    # Left (x=0), Bottom (y=0), Top (y=1)
    for i, (x, y) in enumerate(nodes):
        if abs(x - X0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(y - Y0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(y - Y1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
    # Interface line
    lid = line_id_iface
    for i, (x, y) in enumerate(nodes):
        if abs(x - X1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
elif IF == "left":
    # Right, Bottom, Top
    for i, (x, y) in enumerate(nodes):
        if abs(x - X1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(y - Y0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(y - Y1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
    lid = line_id_iface
    for i, (x, y) in enumerate(nodes):
        if abs(x - X0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
elif IF == "top":
    # Left, Right, Bottom
    for i, (x, y) in enumerate(nodes):
        if abs(x - X0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(x - X1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(y - Y0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
    lid = line_id_iface
    for i, (x, y) in enumerate(nodes):
        if abs(y - Y1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
elif IF == "bottom":
    # Left, Right, Top
    for i, (x, y) in enumerate(nodes):
        if abs(x - X0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(x - X1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
        elif abs(y - Y1) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')
    lid = line_id_iface
    for i, (x, y) in enumerate(nodes):
        if abs(y - Y0) < 1e-9:
            deck_t_lines.append(f'  - "NODE {i + 1} DLINE {lid}"')

with open(DECK_T, "w") as f:
    f.write("\n".join(deck_t_lines))

# WRITE DECK U (TSI)
OUT_U = "out_U"
DECK_U = "deck_U.4C.yaml"
# Slab thickness
TZ = hy / 2.0

deck_u_lines = [
    "TITLE:",
    '  - "Side A Thermo-Elastic U Field (TSI Slab)"',
    "PROBLEM SIZE:",
    f"  DIM: 3",
    f"  ELEMENTS: {nx * ny}",
    f"  NODES: {2 * N2D}",
    "PROBLEM TYPE:",
    '  PROBLEMTYPE: "Thermo_Structure_Interaction"',
    "STRUCTURAL DYNAMIC:",
    '  DYNAMICTYPE: "Statics"',
    "  TIMESTEP: 1.0",
    "  NUMSTEP: 1",
    "  MAXTIME: 1.0",
    "  TOLDISP: 1e-8",
    "  TOLRES: 1e-8",
    "  MAXITER: 20",
    "  LINEAR_SOLVER: 2",
    "THERMAL DYNAMIC:",
    '  DYNAMICTYPE: "Statics"',
    "  TIMESTEP: 1.0",
    "  MAXTIME: 1.0",
    "  TOLTEMP: 1e-8",
    "  TOLRES: 1e-8",
    "  LINEAR_SOLVER: 1",
    "TSI DYNAMIC:",
    '  COUPALGO: "tsi_oneway"',
    "  MAXTIME: 1.0",
    "  TIMESTEP: 1.0",
    "  ITEMAX: 1",
    "TSI DYNAMIC/PARTITIONED:",
    '  COUPVARIABLE: "Temperature"',
    "SOLVER 1:",
    '  SOLVER: "UMFPACK"',
    '  NAME: "Thermal_Solver"',
    "SOLVER 2:",
    '  SOLVER: "UMFPACK"',
    '  NAME: "Structure_Solver"',
    "MATERIALS:",
    f"  - MAT: 1",
    "    MAT_Struct_ThermoStVenantK:",
    f"      YOUNG: [{E_MOD:.15e}]",
    f"      NUE: {NU}",
    f"      DENS: 1.0",
    f"      THEXPANS: {ALPHA}",
    "      INITTEMP: 0.0",
    "      THERMOMAT: 2",
    f"  - MAT: 2",
    "    MAT_Fourier:",
    "      CAPA: 1.0",
    f"      CONDUCT:",
    f"        constant: [{KV}]",
    "CLONING MATERIAL MAP:",
    "  - SRC_FIELD: \"structure\"",
    "    SRC_MAT: 1",
    "    TAR_FIELD: \"thermo\"",
    "    TAR_MAT: 2",
    "FUNCT1:",
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"',
    "FUNCT2:",
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"',
    "FUNCT3:",
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"',
    "IO/MONITOR STRUCTURE DBC:",
    "  INTERVAL_STEPS: 1",
    "  FILE_TYPE: yaml",
    "  WRITE_CONDITION_INFORMATION: true",
    "IO/RUNTIME VTK OUTPUT:",
    "  INTERVAL_STEPS: 1",
    "IO/RUNTIME VTK OUTPUT/STRUCTURE:",
    "  OUTPUT_STRUCTURE: true",
    "  DISPLACEMENT: true",
    "DESIGN SURF THERMO DIRICH CONDITIONS:",
    f"  - E: 1",
    "    NUMDOF: 1",
    "    ONOFF: [1]",
    "    VAL: [0.0]",
    "    FUNCT: [0]",
    "DESIGN SURF DIRICH CONDITIONS:",
    f"  - E: 1",
    "    NUMDOF: 3",
    "    ONOFF: [1, 1, 1]",
    "    VAL: [0.0, 0.0, 0.0]",
    "    FUNCT: [0, 0, 0]",
    "DESIGN POINT THERMO DIRICH CONDITIONS:",
]
# Interface thermal BCs (both layers)
for n_id in iface_nodes:
    T_val, _, _ = partner_values(nodes[n_id - 1][0] if IF in ("left", "right") else nodes[n_id - 1][1])
    # Layer 0
    deck_u_lines.append(f"  - E: {n_id}")
    deck_u_lines.append("    NUMDOF: 1")
    deck_u_lines.append("    ONOFF: [1]")
    deck_u_lines.append(f"    VAL: [{T_val:.15e}]")
    deck_u_lines.append("    FUNCT: [0]")
    # Layer 1
    deck_u_lines.append(f"  - E: {n_id + N2D}")
    deck_u_lines.append("    NUMDOF: 1")
    deck_u_lines.append("    ONOFF: [1]")
    deck_u_lines.append(f"    VAL: [{T_val:.15e}]")
    deck_u_lines.append("    FUNCT: [0]")

deck_u_lines.append("DESIGN POINT DIRICH CONDITIONS:")
# Interface structural BCs (both layers) + u_z=0 on all nodes (plane strain)
for n_id in iface_nodes:
    _, ux_val, uy_val = partner_values(nodes[n_id - 1][0] if IF in ("left", "right") else nodes[n_id - 1][1])
    # Layer 0
    deck_u_lines.append(f"  - E: {n_id}")
    deck_u_lines.append("    NUMDOF: 3")
    deck_u_lines.append("    ONOFF: [1, 1, 1]")
    deck_u_lines.append(f"    VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]")
    deck_u_lines.append("    FUNCT: [0, 0, 0]")
    deck_u_lines.append("    TAG: monitor_reaction")
    # Layer 1
    deck_u_lines.append(f"  - E: {n_id + N2D}")
    deck_u_lines.append("    NUMDOF: 3")
    deck_u_lines.append("    ONOFF: [1, 1, 1]")
    deck_u_lines.append(f"    VAL: [{ux_val:.15e}, {uy_val:.15e}, 0.0]")
    deck_u_lines.append("    FUNCT: [0, 0, 0]")
    deck_u_lines.append("    TAG: monitor_reaction")

# u_z=0 on all nodes (plane strain)
deck_u_lines.append("DESIGN VOL DIRICH CONDITIONS:")
deck_u_lines.append(f"  - E: 1")
deck_u_lines.append("    NUMDOF: 3")
deck_u_lines.append("    ONOFF: [0, 0, 1]")
deck_u_lines.append("    VAL: [0.0, 0.0, 0.0]")
deck_u_lines.append("    FUNCT: [0, 0, 0]")

deck_u_lines.append("DESIGN VOL THERMO NEUMANN CONDITIONS:")
deck_u_lines.append(f"  - E: 1")
deck_u_lines.append("    NUMDOF: 1")
deck_u_lines.append("    ONOFF: [1]")
deck_u_lines.append("    VAL: [1.0]")
deck_u_lines.append("    FUNCT: [1]")

deck_u_lines.append("DESIGN VOL NEUMANN CONDITIONS:")
deck_u_lines.append(f"  - E: 1")
deck_u_lines.append("    NUMDOF: 3")
deck_u_lines.append("    ONOFF: [1, 1, 1]")
deck_u_lines.append("    VAL: [1.0, 1.0, 1.0]")
deck_u_lines.append("    FUNCT: [2, 3, 0]")

deck_u_lines.append("NODE COORDS:")
# Layer 0
for i, (x, y) in enumerate(nodes):
    deck_u_lines.append(f'  - "NODE {i + 1} COORD {x:.15e} {y:.15e} 0.0"')
# Layer 1
for i, (x, y) in enumerate(nodes):
    deck_u_lines.append(f'  - "NODE {i + 1 + N2D} COORD {x:.15e} {y:.15e} {TZ:.15e}"')

deck_u_lines.append("STRUCTURE ELEMENTS:")
el_id = 1
for j in range(ny):
    for i in range(nx):
        n1 = j * (nx + 1) + i + 1
        n2 = n1 + 1
        n3 = n1 + nx + 1
        n4 = n1 + nx
        # HEX8: bottom 4 (n1, n2, n3, n4) CCW, top 4 (n1+N2D, ...)
        # Ordering: 1 2 6 5 9 10 14 13 (in local quad numbering 1..8)
        # 1=n1, 2=n2, 3=n4, 4=n3 (bottom CCW) -> 1, 2, 4, 3
        # 5=n1+N2D, 6=n2+N2D, 7=n4+N2D, 8=n3+N2D (top CCW)
        # 4C HEX8 expects: 1 2 6 5 9 10 14 13 ?
        # Standard: 1 2 3 4 (bottom CCW), 5 6 7 8 (top CCW)
        # Let's use: n1, n2, n3, n4 (bottom CCW) -> n1, n2, n3, n4
        # Wait, n1, n2, n3, n4 is 1, 2, 6, 5 in my loop?
        # i,j -> n1. n2 = n1+1. n3 = n1+nx+1. n4 = n1+nx.
        # CCW: n1 (BL), n2 (BR), n3 (TR), n4 (TL).
        # So 1, 2, 3, 4 -> n1, n2, n3, n4.
        # Top: n1+N2D, n2+N2D, n3+N2D, n4+N2D.
        deck_u_lines.append(f'  - "{el_id} SOLIDSCATRA HEX8 {n1} {n2} {n3} {n4} {n1 + N2D} {n2 + N2D} {n3 + N2D} {n4 + N2D} MAT 1 KINEM linear TYPE Undefined"')
        el_id += 1

deck_u_lines.append("DSURF-NODE TOPOLOGY:")
# Outer surfaces (E: 1)
lid = 1
for i, (x, y) in enumerate(nodes):
    is_outer = False
    if IF == "right":
        if abs(x - X0) < 1e-9 or abs(y - Y0) < 1e-9 or abs(y - Y1) < 1e-9: is_outer = True
    elif IF == "left":
        if abs(x - X1) < 1e-9 or abs(y - Y0) < 1e-9 or abs(y - Y1) < 1e-9: is_outer = True
    elif IF == "top":
        if abs(x - X0) < 1e-9 or abs(x - X1) < 1e-9 or abs(y - Y0) < 1e-9: is_outer = True
    elif IF == "bottom":
        if abs(x - X0) < 1e-9 or abs(x - X1) < 1e-9 or abs(y - Y1) < 1e-9: is_outer = True
    if is_outer:
        # Layer 0 and 1
        deck_u_lines.append(f'  - "NODE {i + 1} DSURFACE {lid}"')
        deck_u_lines.append(f'  - "NODE {i + 1 + N2D} DSURFACE {lid}"')

deck_u_lines.append("DVOL-NODE TOPOLOGY:")
# All nodes (E: 1)
for i in range(2 * N2D):
    deck_u_lines.append(f'  - "NODE {i + 1} DVOL 1"')

with open(DECK_U, "w") as f:
    f.write("\n".join(deck_u_lines))

# RUN DECK T
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{CFG.get('fourc_ld') or ''}:{env.get('LD_LIBRARY_PATH', '')}"
cmd_t = ["stdbuf", "-oL", "-eL", CFG["fourc_bin"], DECK_T, OUT_T]
with open(f"{DECK_T}.log", "w") as log_t:
    proc_t = subprocess.run(cmd_t, stdout=log_t, stderr=subprocess.STDOUT, env=env)
if proc_t.returncode != 0:
    raise SystemExit(why_4c_did_not_finish("run T"))

# RUN DECK U
cmd_u = ["stdbuf", "-oL", "-eL", CFG["fourc_bin"], DECK_U, OUT_U]
with open(f"{DECK_U}.log", "w") as log_u:
    proc_u = subprocess.run(cmd_u, stdout=log_u, stderr=subprocess.STDOUT, env=env)
if proc_u.returncode != 0:
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
