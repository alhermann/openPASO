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
            _hit = False
            for _i, _ln in enumerate(_lines):
                if "PROC 0 ERROR" in _ln:
                    _said = [l.strip() for l in _lines[_i + 1:_i + 12] if l.strip() and not l.startswith("---") and "MPI_ABORT" not in l]
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said[:8]))
                    _hit = True
                    break
            if not _hit:   # a crash without an error message (measured: a zero-area flux boundary -> floating point exception)
                for _i, _ln in enumerate(_lines):
                    if "*** Process received signal ***" in _ln:
                        _sig = next((l.split("Signal:", 1)[1].strip() for l in _lines[_i:_i + 4] if "Signal:" in l), "a signal")
                        _bef = [l.strip() for l in _lines[max(0, _i - 12):_i] if l.strip() and not set(l.strip()) <= set("+-|=")]
                        _why.append(f"4C died on {_sig} ({_lg}) with no error message; the last thing it printed: " + " | ".join(_bef[-2:])
                                    + " -- a flux table dividing by a ZERO boundary area means the flux-calc DLINE shares no edge with any element (its node ids do not match the element numbering)")
                        break
        for _deck in sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml")):
            _txt = open(_deck, errors="ignore").read()
            _secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
            _dup = sorted({x for x in _secs if _secs.count(x) > 1})
            if _dup:
                _why.append(f"{_deck}: section(s) written twice: " + ", ".join(_dup))
            # section names, materials and their parameters are judged by the binary's own grammar in
            # check_input(solver='fourc', input_path=<deck>); run it before the binary -- this check stays short
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
                if "IO/RUNTIME VTK OUTPUT/STRUCTURE" not in _txt or not re.search(r"DISPLACEMENT:\s*true", _txt, re.I):
                    _why.append(f"{_deck}: no structure VTU without `IO/RUNTIME VTK OUTPUT` (INTERVAL_STEPS 1) and `IO/RUNTIME VTK OUTPUT/STRUCTURE` (OUTPUT_STRUCTURE true, DISPLACEMENT true) -- the run finishes 'normally' with nothing for the recovery to read")
                if "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" not in _txt or not re.search(r"TEMPERATURE:\s*true", _txt, re.I):
                    _why.append(f"{_deck}: no thermo VTU without `THERMAL DYNAMIC/RUNTIME VTK OUTPUT` (OUTPUT_THERMO true, TEMPERATURE true)")
                for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):   # a T value in the structural family: '1 DOFs given but 3 expected'
                    _hd = _b.split(":", 1)[0].strip()
                    if re.fullmatch(r"DESIGN (POINT|LINE|SURF|VOL) DIRICH CONDITIONS", _hd) and re.search(r"\bNUMDOF:\s*1\b", _b):
                        _why.append(f"{_deck}: {_hd} has an entry with NUMDOF 1 -- that family carries the slab's 3 displacement dofs; a temperature value belongs in DESIGN {_hd.split()[1]} THERMO DIRICH CONDITIONS")
            if "Scalar_Transport" in _txt:
                if "THERMAL DYNAMIC:" in _txt and "SCALAR TRANSPORT DYNAMIC:" not in _txt:
                    _why.append(f"{_deck}: Scalar_Transport needs `SCALAR TRANSPORT DYNAMIC`, not `THERMAL DYNAMIC`")
                if "CALCFLUX_BOUNDARY" not in _txt or "FLUX CALC" not in _txt:
                    _why.append(f"{_deck}: needs CALCFLUX_BOUNDARY \"diffusive\" AND a `SCATRA FLUX CALC LINE CONDITIONS` entry (E 2) for the flux recovery")
                if re.search(r"^IO:\s*$", _txt, re.M):
                    _why.append(f"{_deck}: an `IO:` section in a Scalar_Transport deck is rejected; the VTU appears without it")
            _badkw = sorted({w for w in re.findall(r'"NODE\s+\d+\s+(D[A-Z]+)\s+\d+"', _txt) if w not in ("DNODE", "DLINE", "DSURFACE", "DVOL")})
            if _badkw:
                _why.append(f"{_deck}: topology entries use {', '.join(_badkw)} -- the entity words are DNODE, DLINE, DSURFACE, DVOL (anything else defines nothing and the conditions on it are silently dropped); section `DSURF-NODE TOPOLOGY` takes entries `NODE <n> DSURFACE <id>` (section word DSURF, entry word DSURFACE)")
            _topo = set(re.findall(r"\b(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)", _txt))
            for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
                _head = _b.split(":", 1)[0].strip()
                _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head) if _head.endswith("CONDITIONS") else None
                if _kw:   # every condition family with a geometry word (SCATRA FLUX CALC LINE CONDITIONS too)
                    _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[_kw.group(1)]
                    _missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", _b) if (_kind, x) not in _topo}, key=int)
                    if _missing:
                        _why.append(f"{_deck}: {_head} names E id(s) {', '.join(_missing[:6])} that no *-NODE TOPOLOGY section defines")
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(diagnosis failed: {_e!r})")
    if not _why:
        _why.append("no 4C error line in any *.log here -- run the binary line-buffered (stdbuf -oL -eL) with its console in a log next to the deck")
    _why.append("check_input(solver='fourc', input_path=<deck>) names every defect the binary's grammar can see (section names with the closest known, materials and their parameters, condition ids, element geometry) in one call")
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
#    Before a run, check_input(solver='fourc', input_path=<deck>) names every defect the grammar
#    can see in one call (4C stops at the first). Run each deck line-buffered with its console in a log
#    (stdbuf -oL -eL <bin> <deck> <prefix> > <deck>.log 2>&1); on a non-zero exit FALL
#    THROUGH, the served check reads the log. LEAVE BEHIND exactly these names:
#      nodes     the list of (x, y) of the 2-D layout; deck node id on the z = 0 layer = index + 1
#      interior  the 1-based ids of the interface nodes WITHOUT the two endpoints, in order along it
#      TZ        the slab thickness (one well-shaped HEX8 layer)
#      OUT_T, OUT_U   the two output prefixes;  DECK_U   the file name you wrote deck U to
# ─────────────────────────────────────────────────────────────────────────

# Mesh Generation
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes.append((x, y))

# Interface Nodes (Right side x=X1)
# IDs are 1-based. Node ID = 1 + i + (NX+1)*j
# Interface line is i = NX. j ranges 0..NY.
# Endpoints (j=0, j=NY) are Outer BC. Interior are Dirichlet.
interface_nodes = []
for j in range(NY + 1):
    nid = 1 + NX + (NX + 1) * j
    interface_nodes.append(nid)

interior = [nid for j, nid in zip(range(NY + 1), interface_nodes) if 0 < j < NY]

# Slab Thickness
TZ = max(X1 - X0, Y1 - Y0) / 10.0  # Reasonable thickness for plane strain

# Output Prefixes
OUT_T = "run_t"
OUT_U = "run_u"

# Deck U File Name (required by recovery logic)
DECK_U = "deck_u.4C.yaml"

# --- Build Deck T (Scalar Transport) ---
deck_t_lines = [
    'TITLE:',
    '  - "Side A Heat"',
    'PROBLEM SIZE:',
    f'  ELEMENTS: {NX * NY}',
    f'  NODES: {(NX + 1) * (NY + 1)}',
    'PROBLEM TYPE:',
    '  PROBLEMTYPE: "Scalar_Transport"',
    'SCALAR TRANSPORT DYNAMIC:',
    '  TIMEINTEGR: "Stationary"',
    '  SOLVERTYPE: "linear_full"',
    '  NUMSTEP: 1',
    '  TIMESTEP: 1.0',
    '  MAXTIME: 1.0',
    '  LINEAR_SOLVER: 1',
    'SOLVER 1:',
    '  SOLVER: "UMFPACK"',
    'MATERIALS:',
    '  - MAT: 1',
    '    MAT_scatra:',
    f'      DIFFUSIVITY: {KV}',
    'FUNCT1:',
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"',
    'CALCFLUX_BOUNDARY: "diffusive"',
]

# Topology IDs
# Lines: Left(1), Bottom(2), Top(3), Right(4)
# Surfaces: Vol(1)
# Points: Interface Interior Nodes (5 + idx)

deck_t_lines.extend(['DESIGN LINE DIRICH CONDITIONS:'])
# Left (x=0)
deck_t_lines.append('  - E: 1')
deck_t_lines.append('    NUMDOF: 1')
deck_t_lines.append('    ONOFF: [1]')
deck_t_lines.append('    VAL: [0.0]')
deck_t_lines.append('    FUNCT: [0]')
# Bottom (y=0)
deck_t_lines.append('  - E: 2')
deck_t_lines.append('    NUMDOF: 1')
deck_t_lines.append('    ONOFF: [1]')
deck_t_lines.append('    VAL: [0.0]')
deck_t_lines.append('    FUNCT: [0]')
# Top (y=1)
deck_t_lines.append('  - E: 3')
deck_t_lines.append('    NUMDOF: 1')
deck_t_lines.append('    ONOFF: [1]')
deck_t_lines.append('    VAL: [0.0]')
deck_t_lines.append('    FUNCT: [0]')

deck_t_lines.append('DESIGN POINT DIRICH CONDITIONS:')
eid_pt = 5
for nid in interior:
    val = partner_values(nodes[nid-1][1])  # y-coord
    deck_t_lines.append(f'  - E: {eid_pt}')
    deck_t_lines.append('    NUMDOF: 1')
    deck_t_lines.append('    ONOFF: [1]')
    deck_t_lines.append(f'    VAL: [{val[0]}]')
    deck_t_lines.append('    FUNCT: [0]')
    eid_pt += 1

deck_t_lines.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
deck_t_lines.append('  - E: 1')
deck_t_lines.append('    NUMDOF: 1')
deck_t_lines.append('    ONOFF: [1]')
deck_t_lines.append('    VAL: [1.0]')
deck_t_lines.append('    FUNCT: [1]')

deck_t_lines.append('SCATRA FLUX CALC LINE CONDITIONS:')
deck_t_lines.append('  - E: 4')  # Right Line

deck_t_lines.append('NODE COORDS:')
for i, (x, y) in enumerate(nodes):
    deck_t_lines.append(f'  - "NODE {i+1} COORD {x:.11e} {y:.11e} 0.0"')

deck_t_lines.append('TRANSPORT ELEMENTS:')
eid_el = 1
for j in range(NY):
    for i in range(NX):
        n1 = 1 + i + (NX + 1) * j
        n2 = 1 + (i + 1) + (NX + 1) * j
        n3 = 1 + (i + 1) + (NX + 1) * (j + 1)
        n4 = 1 + i + (NX + 1) * (j + 1)
        deck_t_lines.append(f'  - "{eid_el} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
        eid_el += 1

deck_t_lines.append('DLINE-NODE TOPOLOGY:')
# Left Line (j=0..NY, i=0)
for j in range(NY + 1):
    nid = 1 + (NX + 1) * j
    deck_t_lines.append(f'  - "NODE {nid} DLINE 1"')
# Bottom Line (i=0..NX, j=0)
for i in range(NX + 1):
    nid = 1 + i
    deck_t_lines.append(f'  - "NODE {nid} DLINE 2"')
# Top Line (i=0..NX, j=NY)
for i in range(NX + 1):
    nid = 1 + i + (NX + 1) * NY
    deck_t_lines.append(f'  - "NODE {nid} DLINE 3"')
# Right Line (j=0..NY, i=NX)
for j in range(NY + 1):
    nid = 1 + NX + (NX + 1) * j
    deck_t_lines.append(f'  - "NODE {nid} DLINE 4"')

deck_t_lines.append('DNODE-NODE TOPOLOGY:')
eid_dn = 5
for nid in interior:
    deck_t_lines.append(f'  - "NODE {nid} DNODE {eid_dn}"')
    eid_dn += 1

deck_t_lines.append('DSURF-NODE TOPOLOGY:')
for i, _ in enumerate(nodes):
    deck_t_lines.append(f'  - "NODE {i+1} DSURFACE 1"')

Path("deck_t.4C.yaml").write_text("\n".join(deck_t_lines))

# --- Build Deck U (TSI) ---
# Materials
# Struct: MAT_Struct_ThermoStVenantK
# Thermo: MAT_Fourier
# Map: SRC_FIELD structure, TAR_FIELD thermo

deck_u_lines = [
    'TITLE:',
    '  - "Side A ThermoElastic"',
    'PROBLEM SIZE:',
    '  DIM: 3',
    'IO:',
    '  STRUCT_STRESS: "No"',
    '  STRUCT_STRAIN: "No"',
    'IO/MONITOR STRUCTURE DBC:',
    '  INTERVAL_STEPS: 1',
    '  FILE_TYPE: yaml',
    '  WRITE_CONDITION_INFORMATION: true',
    'PROBLEM TYPE:',
    '  PROBLEMTYPE: "Thermo_Structure_Interaction"',
    'STRUCTURAL DYNAMIC:',
    '  DYNAMICTYPE: "Statics"',
    '  TIMESTEP: 1',
    '  NUMSTEP: 1',
    '  MAXTIME: 1',
    '  LINEAR_SOLVER: 2',
    'THERMAL DYNAMIC:',
    '  DYNAMICTYPE: Statics',
    '  TIMESTEP: 1',
    '  NUMSTEP: 1',
    '  MAXTIME: 1',
    '  LINEAR_SOLVER: 1',
    'TSI DYNAMIC:',
    '  COUPALGO: "tsi_oneway"',
    '  NUMSTEP: 1',
    '  MAXTIME: 1',
    '  TIMESTEP: 1',
    '  ITEMAX: 1',
    'TSI DYNAMIC/PARTITIONED:',
    '  COUPVARIABLE: "Temperature"',
    'IO/RUNTIME VTK OUTPUT:',
    '  INTERVAL_STEPS: 1',
    '  OUTPUT_DATA_FORMAT: "ascii"',
    'IO/RUNTIME VTK OUTPUT/STRUCTURE:',
    '  OUTPUT_STRUCTURE: true',
    '  DISPLACEMENT: true',
    'THERMAL DYNAMIC/RUNTIME VTK OUTPUT:',
    '  OUTPUT_THERMO: true',
    '  TEMPERATURE: true',
    'SOLVER 1:',
    '  SOLVER: "UMFPACK"',
    '  NAME: "Thermal_Solver"',
    'SOLVER 2:',
    '  SOLVER: "UMFPACK"',
    '  NAME: "Structure_Solver"',
    'MATERIALS:',
    '  - MAT: 1',
    '    MAT_Struct_ThermoStVenantK:',
    '      YOUNGNUM: 1',
    f'      YOUNG: [{E_MOD}]',
    f'      NUE: {NU}',
    '      DENS: 1',
    f'      THEXPANS: {ALPHA}',
    '      INITTEMP: 0',
    '      THERMOMAT: 2',
    '  - MAT: 2',
    '    MAT_Fourier:',
    '      CAPA: 1',
    f'      CONDUCT:',
    f'        constant: [{KV}]',
    'CLONING MATERIAL MAP:',
    '  - SRC_FIELD: "structure"',
    '    SRC_MAT: 1',
    '    TAR_FIELD: "thermo"',
    '    TAR_MAT: 2',
    'FUNCT1:',
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"',
    'FUNCT2:',
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"',
    'FUNCT3:',
    f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"',
    'DESIGN VOL NEUMANN CONDITIONS:',
    '  - E: 1',
    '    NUMDOF: 3',
    '    ONOFF: [1, 1, 0]',
    '    VAL: [1.0, 1.0, 0.0]',
    '    FUNCT: [1, 2, 0]',
    'DESIGN VOL THERMO NEUMANN CONDITIONS:',
    '  - E: 1',
    '    NUMDOF: 1',
    '    ONOFF: [1]',
    '    VAL: [1.0]',
    '    FUNCT: [3]',
    'DESIGN VOL DIRICH CONDITIONS:',
    '  - E: 1',
    '    NUMDOF: 3',
    '    ONOFF: [0, 0, 1]',
    '    VAL: [0, 0, 0]',
    '    FUNCT: [0, 0, 0]',
    'DESIGN SURF DIRICH CONDITIONS:',
    '  - E: 1',
    '    NUMDOF: 3',
    '    ONOFF: [1, 1, 1]',
    '    VAL: [0, 0, 0]',
    '    FUNCT: [0, 0, 0]',
    'DESIGN SURF THERMO DIRICH CONDITIONS:',
    '  - E: 1',
    '    NUMDOF: 1',
    '    ONOFF: [1]',
    '    VAL: [0]',
    '    FUNCT: [0]',
    'DESIGN POINT DIRICH CONDITIONS:',
]
# Point Dirichlet on Interface (Interior nodes on BOTH layers)
# Layer 0 IDs: 1..N2D. Layer 1 IDs: N2D+1..2*N2D.
# Interface IDs on Layer 0: interior. On Layer 1: interior + N2D.
eid_pt_struct = 5
for nid in interior:
    # Layer 0
    deck_u_lines.append(f'  - E: {eid_pt_struct}')
    deck_u_lines.append('    NUMDOF: 3')
    deck_u_lines.append('    ONOFF: [1, 1, 1]')
    val = partner_values(nodes[nid-1][1])
    deck_u_lines.append(f'    VAL: [{val[1]}, {val[2]}, 0.0]')
    deck_u_lines.append('    FUNCT: [0, 0, 0]')
    deck_u_lines.append('    TAG: monitor_reaction')
    eid_pt_struct += 1
    # Layer 1
    nid_l1 = nid + len(nodes)
    deck_u_lines.append(f'  - E: {eid_pt_struct}')
    deck_u_lines.append('    NUMDOF: 3')
    deck_u_lines.append('    ONOFF: [1, 1, 1]')
    deck_u_lines.append(f'    VAL: [{val[1]}, {val[2]}, 0.0]')
    deck_u_lines.append('    FUNCT: [0, 0, 0]')
    deck_u_lines.append('    TAG: monitor_reaction')
    eid_pt_struct += 1

deck_u_lines.append('DESIGN POINT THERMO DIRICH CONDITIONS:')
eid_pt_thermo = eid_pt_struct
for nid in interior:
    val = partner_values(nodes[nid-1][1])
    # Layer 0
    deck_u_lines.append(f'  - E: {eid_pt_thermo}')
    deck_u_lines.append('    NUMDOF: 1')
    deck_u_lines.append('    ONOFF: [1]')
    deck_u_lines.append(f'    VAL: [{val[0]}]')
    deck_u_lines.append('    FUNCT: [0]')
    eid_pt_thermo += 1
    # Layer 1
    deck_u_lines.append(f'  - E: {eid_pt_thermo}')
    deck_u_lines.append('    NUMDOF: 1')
    deck_u_lines.append('    ONOFF: [1]')
    deck_u_lines.append(f'    VAL: [{val[0]}]')
    deck_u_lines.append('    FUNCT: [0]')
    eid_pt_thermo += 1

deck_u_lines.append('NODE COORDS:')
# Layer 0
for i, (x, y) in enumerate(nodes):
    deck_u_lines.append(f'  - "NODE {i+1} COORD {x:.11e} {y:.11e} 0.0"')
# Layer 1
for i, (x, y) in enumerate(nodes):
    deck_u_lines.append(f'  - "NODE {len(nodes)+i+1} COORD {x:.11e} {y:.11e} {TZ:.11e}"')

deck_u_lines.append('STRUCTURE ELEMENTS:')
eid_el = 1
for j in range(NY):
    for i in range(NX):
        n1 = 1 + i + (NX + 1) * j
        n2 = 1 + (i + 1) + (NX + 1) * j
        n3 = 1 + (i + 1) + (NX + 1) * (j + 1)
        n4 = 1 + i + (NX + 1) * (j + 1)
        n1_l1, n2_l1, n3_l1, n4_l1 = [n + len(nodes) for n in [n1, n2, n3, n4]]
        deck_u_lines.append(f'  - "{eid_el} SOLIDSCATRA HEX8 {n1} {n2} {n3} {n4} {n1_l1} {n2_l1} {n3_l1} {n4_l1} MAT 1 KINEM linear TYPE Undefined"')
        eid_el += 1

deck_u_lines.append('DNODE-NODE TOPOLOGY:')
eid_dn = 5
for nid in interior:
    # Layer 0
    deck_u_lines.append(f'  - "NODE {nid} DNODE {eid_dn}"')
    eid_dn += 1
    # Layer 1
    nid_l1 = nid + len(nodes)
    deck_u_lines.append(f'  - "NODE {nid_l1} DNODE {eid_dn}"')
    eid_dn += 1

deck_u_lines.append('DSURF-NODE TOPOLOGY:')
# Outer Boundary Faces (Left, Bottom, Top)
# For simplicity, we map all nodes on outer boundaries to Surface ID 1
# Left (x=0), Bottom (y=0), Top (y=1)
for j in range(NY + 1):
    nid = 1 + (NX + 1) * j
    deck_u_lines.append(f'  - "NODE {nid} DSURFACE 1"')
    deck_u_lines.append(f'  - "NODE {nid + len(nodes)} DSURFACE 1"')
for i in range(NX + 1):
    nid = 1 + i
    deck_u_lines.append(f'  - "NODE {nid} DSURFACE 1"')
    deck_u_lines.append(f'  - "NODE {nid + len(nodes)} DSURFACE 1"')
    nid_top = 1 + i + (NX + 1) * NY
    deck_u_lines.append(f'  - "NODE {nid_top} DSURFACE 1"')
    deck_u_lines.append(f'  - "NODE {nid_top + len(nodes)} DSURFACE 1"')

deck_u_lines.append('DVOL-NODE TOPOLOGY:')
for i in range(len(nodes)):
    deck_u_lines.append(f'  - "NODE {i+1} DVOL 1"')
    deck_u_lines.append(f'  - "NODE {i+1+len(nodes)} DVOL 1"')

Path(DECK_U).write_text("\n".join(deck_u_lines))

# --- Run 4C ---
BIN = CFG["fourc_bin"]
LD = CFG.get("fourc_ld", "")
ENV_PRE = f"LD_LIBRARY_PATH={LD} " if LD else ""

cmd_t = f"{ENV_PRE}stdbuf -oL -eL {BIN} deck_t.4C.yaml {OUT_T} > deck_t.log 2>&1"
res_t = subprocess.run(cmd_t, shell=True)
if res_t.returncode != 0:
    pass # Let the recovery logic handle it

cmd_u = f"{ENV_PRE}stdbuf -oL -eL {BIN} {DECK_U} {OUT_U} > deck_u.log 2>&1"
res_u = subprocess.run(cmd_u, shell=True)
if res_u.returncode != 0:
    pass # Let the recovery logic handle it

# ── RECOVERY FROM 4C's OWN OUTPUTS (served): boundary flux VTU, displacement VTU, reaction yaml ──
import meshio  # noqa: E402

# ── 4C's OWN CONSOLE, ECHOED (served): the coupling tool captures THIS script's stdout as the
#    level's run log, and a run log is credited to 4C only by 4C's own lines (its step and
#    time-integration output), never by the NDOF line alone. Measured: a cell whose run logs held
#    only the driver header and the NDOF line was graded as no per-code execution evidence.
for _lg in sorted(glob.glob("*.log")):
    if _lg.startswith("participant_output") or "level" in _lg:      # the coupling tool's own captures, never re-echoed
        continue
    try:
        _txt = Path(_lg).read_text(errors="ignore")
    except OSError:
        continue
    if "── 4C console" in _txt:                                      # an earlier echo, not a deck console
        continue
    if "4C" in _txt[:4000] or "PROC 0" in _txt or "Finalised step" in _txt:
        print(f"── 4C console {_lg} ──")
        print(_txt if len(_txt) < 60000 else _txt[-60000:])

# ── YOUR DECKS, CHECKED BEFORE ANYTHING IS READ (served): 4C drops a condition whose E id no
#    topology section defines and RUNS THE WRONG PROBLEM to 'finished normally' (measured on a
#    worker deck: every boundary condition and the source gone, rc 0). A run like that is refused here.
for _dk in sorted(glob.glob("*.4C.yaml")) or [p for p in sorted(glob.glob("*.yaml")) if "monitor_dbc" not in p]:
    _txt = Path(_dk).read_text(errors="ignore")
    _topo = set(re.findall(r"\b(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)", _txt))
    _lost = []
    for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
        _head = _b.split(":", 1)[0].strip()
        _kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", _head) if _head.endswith("CONDITIONS") else None
        if _kw:   # every condition family with a geometry word (SCATRA FLUX CALC LINE CONDITIONS too)
            _kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[_kw.group(1)]
            _lost += [f"{_head} E {x}" for x in re.findall(r"\bE:\s*(\d+)", _b) if (_kind, x) not in _topo]
    if _lost:
        raise SystemExit(f"DECK CHECK: {_dk} puts conditions on E ids that no *-NODE TOPOLOGY section defines "
                         f"({'; '.join(_lost[:6])}): 4C dropped them silently, so the run solved a different "
                         f"problem. Add the DNODE/DLINE/DSURF/DVOL-NODE TOPOLOGY entries for those ids.")
    # twisted or clockwise 2-D elements (zero/negative area from the deck's own coordinates) solve nothing
    _cxy = {int(n): (float(x), float(y)) for n, x, y in re.findall(r'"NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+\S+"', _txt)}
    _twist = []
    for _e, _k, _ids in re.findall(r'"\s*(\d+)\s+\w+\s+(QUAD4|TRI3)\s+((?:\d+\s+)+)', _txt):
        _nn = [int(i) for i in _ids.split()][:4 if _k == "QUAD4" else 3]
        if len(_nn) >= 3 and all(i in _cxy for i in _nn):
            _p = [_cxy[i] for i in _nn]
            if 0.5 * sum(_p[q][0] * _p[(q + 1) % len(_p)][1] - _p[(q + 1) % len(_p)][0] * _p[q][1] for q in range(len(_p))) <= 1e-14:
                _twist.append((_e, _nn))
    if _twist:
        raise SystemExit(f"DECK CHECK: {_dk} has {len(_twist)} element(s) with zero or negative area (first: element {_twist[0][0]} "
                         f"nodes {' '.join(map(str, _twist[0][1]))}). Every element's nodes must run counter-clockwise: for node "
                         f"id = i + 1 + (NX + 1) * j the quad of cell (i, j) is (id, id + 1, id + NX + 2, id + NX + 1).")


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

# ── WHAT YOUR SOLVE MUST LEAVE BEHIND ─────────────────────────
# The code above and below the elided block uses these names. Your
# block has to define every one of them, or the rest will not run:
#
#     DECK_U
#     OUT_T
#     OUT_U
#     TZ
#     interior
#     nodes
#
# That is the whole contract. Read the surviving lines to see the
# shape each one has to have -- they are already indexed, assembled
# or written out there. OASiS does not serve the solve itself, but
# it will not make you guess which variables the hole was filling.
