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

import meshio  # noqa: E402
from scipy.io import savemat # noqa: E402

# --- MESH GENERATION ---
DX = (X1 - X0) / NX
DY = (Y1 - Y0) / NY
nodes_2d = []
node_map = {} # (i, j) -> deck_id
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * DX
        y = Y0 + j * DY
        deck_id = len(nodes_2d) + 1
        nodes_2d.append((x, y))
        node_map[(i, j)] = deck_id

# Interface nodes (Right edge: i = NX)
# Endpoints: j=0 and j=NY. These are outer nodes (T=0, u=0).
# Interior interface nodes: j=1..NY-1.
interior_ids = []
for j in range(1, NY):
    interior_ids.append(node_map[(NX, j)])

# Slab for Run U (3D, Plane Strain)
# Two layers: z=0 and z=TZ.
TZ = 0.1 # Arbitrary thickness, traction will be normalized by it
nodes_3d = []
node_map_3d = {} # (i, j, layer) -> deck_id
# Layer 0 (z=0)
for j in range(NY + 1):
    for i in range(NX + 1):
        x, y = nodes_2d[node_map[(i, j)] - 1]
        deck_id = len(nodes_3d) + 1
        nodes_3d.append((x, y, 0.0))
        node_map_3d[(i, j, 0)] = deck_id
# Layer 1 (z=TZ)
for j in range(NY + 1):
    for i in range(NX + 1):
        x, y = nodes_2d[node_map[(i, j)] - 1]
        deck_id = len(nodes_3d) + 1
        nodes_3d.append((x, y, TZ))
        node_map_3d[(i, j, 1)] = deck_id

# --- DECK T (Scalar_Transport) ---
deck_T_file = "deck_T.4C.yaml"
with open(deck_T_file, "w") as f:
    f.write('TITLE:\n')
    f.write('  - "Run T: Scalar Transport"\n')
    f.write('PROBLEM SIZE:\n')
    f.write(f'  ELEMENTS: {NX * NY}\n')
    f.write(f'  NODES: {len(nodes_2d)}\n')
    f.write('PROBLEM TYPE:\n')
    f.write('  PROBLEMTYPE: "Scalar_Transport"\n')
    f.write('SCALAR TRANSPORT DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n')
    f.write('  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n')
    f.write('  TIMESTEP: 1.0\n')
    f.write('  MAXTIME: 1.0\n')
    f.write('  LINEAR_SOLVER: 1\n')
    f.write('SOLVER 1:\n')
    f.write('  SOLVER: "UMFPACK"\n')
    f.write('MATERIALS:\n')
    f.write(f'  - MAT: 1\n')
    f.write(f'    MAT_scatra:\n')
    f.write(f'      DIFFUSIVITY: {KV}\n')
    f.write(f'FUNCT1:\n')
    f.write(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\n')
    f.write(f'DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n')
    f.write(f'  - E: 1\n')
    f.write(f'    NUMDOF: 1\n')
    f.write(f'    ONOFF: [1]\n')
    f.write(f'    VAL: [1.0]\n')
    f.write(f'    FUNCT: [1]\n')
    f.write(f'DESIGN LINE DIRICH CONDITIONS:\n')
    # Outer BCs: Left (i=0), Bottom (j=0), Top (j=NY)
    # We need to define Lines. Let's assign E IDs.
    # Left Line: nodes (0, 0)..(0, NY)
    # Bottom Line: nodes (0, 0)..(NX, 0)
    # Top Line: nodes (0, NY)..(NX, NY)
    # Right Line (Interface): handled by POINT DIRICH
    f.write(f'  - E: 1\n') # Left
    f.write(f'    NUMDOF: 1\n')
    f.write(f'    ONOFF: [1]\n')
    f.write(f'    VAL: [0.0]\n')
    f.write(f'    FUNCT: [0]\n')
    f.write(f'  - E: 2\n') # Bottom
    f.write(f'    NUMDOF: 1\n')
    f.write(f'    ONOFF: [1]\n')
    f.write(f'    VAL: [0.0]\n')
    f.write(f'    FUNCT: [0]\n')
    f.write(f'  - E: 3\n') # Top
    f.write(f'    NUMDOF: 1\n')
    f.write(f'    ONOFF: [1]\n')
    f.write(f'    VAL: [0.0]\n')
    f.write(f'    FUNCT: [0]\n')
    # Interface Points
    f.write(f'DESIGN POINT DIRICH CONDITIONS:\n')
    for idx, nid in enumerate(interior_ids):
        val = partner_values(nodes_2d[nid - 1][1]) # y-coord
        f.write(f'  - E: {4 + idx}\n')
        f.write(f'    NUMDOF: 1\n')
        f.write(f'    ONOFF: [1]\n')
        f.write(f'    VAL: [{val[0]}]\n') # T
        f.write(f'    FUNCT: [0]\n')
    f.write(f'NODE COORDS:\n')
    for i, n in enumerate(nodes_2d):
        f.write(f'  - "NODE {i+1} COORD {n[0]} {n[1]} 0.0"\n')
    f.write(f'TRANSPORT ELEMENTS:\n')
    eid = 1
    for j in range(NY):
        for i in range(NX):
            n1 = node_map[(i, j)]
            n2 = node_map[(i+1, j)]
            n3 = node_map[(i+1, j+1)]
            n4 = node_map[(i, j+1)]
            f.write(f'  - "{eid} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n')
            eid += 1
    f.write(f'DLINE-NODE TOPOLOGY:\n')
    # Left (E:1)
    for j in range(NY + 1):
        f.write(f'  - "NODE {node_map[(0, j)]} DLINE 1"\n')
    # Bottom (E:2)
    for i in range(NX + 1):
        f.write(f'  - "NODE {node_map[(i, 0)]} DLINE 2"\n')
    # Top (E:3)
    for i in range(NX + 1):
        f.write(f'  - "NODE {node_map[(i, NY)]} DLINE 3"\n')
    # Volume (E:1 for Neumann)
    f.write(f'DSURF-NODE TOPOLOGY:\n')
    for i, n in enumerate(nodes_2d):
        f.write(f'  - "NODE {i+1} DSURFACE 1"\n')
    # Interface Points Topology
    f.write(f'DNODE-NODE TOPOLOGY:\n')
    for idx, nid in enumerate(interior_ids):
        f.write(f'  - "NODE {nid} DNODE {4 + idx}"\n')
    f.write(f'CALCFLUX_BOUNDARY: "diffusive"\n')
    f.write(f'SCATRA FLUX CALC LINE CONDITIONS:\n')
    f.write(f'  - E: 4\n') # Interface Line ID (virtual, for flux calc)
    # We need to define the interface line topology for flux calc?
    # The documentation says "SCATRA FLUX CALC LINE CONDITIONS: - E: <line id>"
    # But we didn't define a DLINE for the interface (only POINT DIRICH).
    # However, for flux calc on the interface, we need to tell 4C which line it is.
    # Let's add a DLINE 4 for the interface nodes.
    f.write(f'DLINE-NODE TOPOLOGY:\n') # Wait, DLINE defined twice?
    # Fix: Add to existing DLINE block.
    # Re-structure: I will write the deck in memory then dump.

# --- REWRITE DECK T CORRECTLY ---
# Collect all DLINE nodes first
dline_1 = [node_map[(0, j)] for j in range(NY + 1)]
dline_2 = [node_map[(i, 0)] for i in range(NX + 1)]
dline_3 = [node_map[(i, NY)] for i in range(NX + 1)]
dline_4 = [node_map[(NX, j)] for j in range(NY + 1)] # Interface Line

with open(deck_T_file, "w") as f:
    f.write('TITLE:\n  - "Run T: Scalar Transport"\n')
    f.write('PROBLEM SIZE:\n  ELEMENTS: {}\n  NODES: {}\n'.format(NX * NY, len(nodes_2d)))
    f.write('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\n')
    f.write('SCALAR TRANSPORT DYNAMIC:\n')
    f.write('  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n  TIMESTEP: 1.0\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n')
    f.write('SOLVER 1:\n  SOLVER: "UMFPACK"\n')
    f.write('MATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: {}\n'.format(KV))
    f.write('FUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{}"\n'.format(SRC_T))
    f.write('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n')
    f.write('  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\n')
    f.write('DESIGN LINE DIRICH CONDITIONS:\n')
    f.write('  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n') # Left
    f.write('  - E: 2\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n') # Bottom
    f.write('  - E: 3\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n') # Top
    f.write('DESIGN POINT DIRICH CONDITIONS:\n')
    for idx, nid in enumerate(interior_ids):
        val = partner_values(nodes_2d[nid - 1][1])
        f.write('  - E: {}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{}]\n    FUNCT: [0]\n'.format(4 + idx, val[0]))
    f.write('NODE COORDS:\n')
    for i, n in enumerate(nodes_2d):
        f.write('  - "NODE {} COORD {} {} 0.0"\n'.format(i+1, n[0], n[1]))
    f.write('TRANSPORT ELEMENTS:\n')
    eid = 1
    for j in range(NY):
        for i in range(NX):
            n1, n2, n3, n4 = node_map[(i, j)], node_map[(i+1, j)], node_map[(i+1, j+1)], node_map[(i, j+1)]
            f.write('  - "{} TRANSP QUAD4 {} {} {} {} MAT 1 TYPE Std"\n'.format(eid, n1, n2, n3, n4))
            eid += 1
    f.write('DLINE-NODE TOPOLOGY:\n')
    for nid in dline_1: f.write('  - "NODE {} DLINE 1"\n'.format(nid))
    for nid in dline_2: f.write('  - "NODE {} DLINE 2"\n'.format(nid))
    for nid in dline_3: f.write('  - "NODE {} DLINE 3"\n'.format(nid))
    for nid in dline_4: f.write('  - "NODE {} DLINE 4"\n'.format(nid))
    f.write('DSURF-NODE TOPOLOGY:\n')
    for i, n in enumerate(nodes_2d): f.write('  - "NODE {} DSURFACE 1"\n'.format(i+1))
    f.write('DNODE-NODE TOPOLOGY:\n')
    for idx, nid in enumerate(interior_ids): f.write('  - "NODE {} DNODE {}\n".format(nid, 4 + idx))
    f.write('CALCFLUX_BOUNDARY: "diffusive"\n')
    f.write('SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 4\n')

# --- DECK U (Thermo_Structure_Interaction) ---
deck_U_file = "deck_U.4C.yaml"
OUT_U = "out_U"
OUT_T = "out_T"

# Slab Nodes: Layer 0 (z=0), Layer 1 (z=TZ)
# DNODE IDs for Interface Points:
# Layer 0: 1..N_int
# Layer 1: N_int+1..2*N_int
N_int = len(interior_ids)
dnode_ids_layer0 = list(range(1, N_int + 1))
dnode_ids_layer1 = list(range(N_int + 1, 2 * N_int + 1))

# Map 3D nodes to interface DNODE IDs
# Interface nodes in 3D: (NX, j, 0) and (NX, j, 1) for j=1..NY-1
dnode_map = {}
for idx, j in enumerate(range(1, NY)):
    dnode_map[(NX, j, 0)] = dnode_ids_layer0[idx]
    dnode_map[(NX, j, 1)] = dnode_ids_layer1[idx]

# DLINE for Outer BCs (3D)
# Left (i=0), Bottom (j=0), Top (j=NY)
# We need to define lines for both layers? No, 3D elements are HEX8.
# Outer BCs are on Surfaces (DSURFACE) or Lines?
# For Structure, `DESIGN LINE DIRICH` applies to lines of the mesh.
# In 3D slab, the outer boundaries are surfaces.
# But 4C Structure DIRICH can be `DESIGN SURF STRUCTURE DIRICH CONDITIONS`?
# Documentation says: `DESIGN LINE DIRICH` for 2D, `DESIGN SURF DIRICH` for 3D?
# Let's check: "STRUCTURE runtime-VTK...".
# The contract snippet uses `DESIGN LINE DIRICH` for 2D.
# For 3D slab, we should use `DESIGN SURF STRUCTURE DIRICH CONDITIONS`.
# But wait, the slab is thin. The outer boundary is a surface (x=0 plane, y=0 plane, etc).
# Let's use `DESIGN SURF STRUCTURE DIRICH CONDITIONS` for Outer BCs.
# And `DESIGN POINT STRUCTURE DIRICH CONDITIONS` for Interface.

# Material IDs:
# MAT 1: Structure (ThermoStVenantK)
# MAT 2: Thermal (Fourier)
# CLONING MATERIAL MAP: MAT 1 <-> MAT 2

with open(deck_U_file, "w") as f:
    f.write('TITLE:\n  - "Run U: Thermo Structure Interaction"\n')
    f.write('PROBLEM SIZE:\n  ELEMENTS: {}\n  NODES: {}\n'.format(NX * NY * 2, len(nodes_3d)))
    f.write('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\n')
    f.write('COUPALGO: "tsi_oneway"\n')
    f.write('STRUCTURAL DYNAMIC:\n  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n  TIMESTEP: 1.0\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n')
    f.write('THERMAL DYNAMIC:\n  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n')
    f.write('  NUMSTEP: 1\n  TIMESTEP: 1.0\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n')
    f.write('SOLVER 1:\n  SOLVER: "UMFPACK"\n')
    f.write('MATERIALS:\n')
    f.write('  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n')
    f.write('      YOUNG_MODULUS: {}\n      POISSON_RATIO: {}\n'.format(E_MOD, NU))
    f.write('      LINEAR_EXPANSION_COEFF: {}\n      INITTEMP: 0.0\n'.format(ALPHA))
    f.write('  - MAT: 2\n    MAT_Fourier:\n      DIFFUSIVITY: {}\n'.format(KV))
    f.write('CLONING MATERIAL MAP:\n  - MAT: 1\n    CLONE_MAT: 2\n')
    f.write('COUPVARIABLE: "Temperature"\n')
    f.write('FUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{}"\n'.format(SRC_UX))
    f.write('FUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{}"\n'.format(SRC_UY))
    f.write('FUNCT3:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{}"\n'.format(SRC_T))
    f.write('DESIGN VOL STRUCTURE NEUMANN CONDITIONS:\n')
    f.write('  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [1.0, 1.0, 1.0]\n    FUNCT: [1, 2, 0]\n')
    f.write('DESIGN VOL THERMO NEUMANN CONDITIONS:\n')
    f.write('  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [3]\n')
    f.write('DESIGN SURF STRUCTURE DIRICH CONDITIONS:\n')
    f.write('  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0.0, 0.0, 0.0]\n    FUNCT: [0]\n') # Left
    f.write('  - E: 2\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0.0, 0.0, 0.0]\n    FUNCT: [0]\n') # Bottom
    f.write('  - E: 3\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0.0, 0.0, 0.0]\n    FUNCT: [0]\n') # Top
    f.write('DESIGN SURF THERMO DIRICH CONDITIONS:\n')
    f.write('  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n') # Left
    f.write('  - E: 2\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n') # Bottom
    f.write('  - E: 3\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\n') # Top
    f.write('DESIGN POINT STRUCTURE DIRICH CONDITIONS:\n')
    # Interface Points (ux, uy from import, uz=0)
    for idx, nid in enumerate(interior_ids):
        val = partner_values(nodes_2d[nid - 1][1])
        # Layer 0
        nid_3d_0 = node_map_3d[(NX, idx + 1, 0)]
        f.write('  - E: {}\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [{}, {}, 0.0]\n    FUNCT: [0]\n'.format(
            dnode_ids_layer0[idx], val[1], val[2]))
        # Layer 1
        nid_3d_1 = node_map_3d[(NX, idx + 1, 1)]
        f.write('  - E: {}\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [{}, {}, 0.0]\n    FUNCT: [0]\n'.format(
            dnode_ids_layer1[idx], val[1], val[2]))
    f.write('TAG: monitor_reaction\n') # Apply to all above
    f.write('DESIGN POINT THERMO DIRICH CONDITIONS:\n')
    # Interface Points (T from import)
    for idx, nid in enumerate(interior_ids):
        val = partner_values(nodes_2d[nid - 1][1])
        # Layer 0
        nid_3d_0 = node_map_3d[(NX, idx + 1, 0)]
        f.write('  - E: {}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{}]\n    FUNCT: [0]\n'.format(
            dnode_ids_layer0[idx], val[0]))
        # Layer 1
        nid_3d_1 = node_map_3d[(NX, idx + 1, 1)]
        f.write('  - E: {}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{}]\n    FUNCT: [0]\n'.format(
            dnode_ids_layer1[idx], val[0]))
    f.write('NODE COORDS:\n')
    for i, n in enumerate(nodes_3d):
        f.write('  - "NODE {} COORD {} {} {}\n".format(i+1, n[0], n[1], n[2]))
    f.write('SOLID ELEMENTS:\n')
    eid = 1
    for j in range(NY):
        for i in range(NX):
            # HEX8: Bottom layer (0), Top layer (1)
            # Order: 1,2,3,4 bottom; 5,6,7,8 top
            n1 = node_map_3d[(i, j, 0)]
            n2 = node_map_3d[(i+1, j, 0)]
            n3 = node_map_3d[(i+1, j+1, 0)]
            n4 = node_map_3d[(i, j+1, 0)]
            n5 = node_map_3d[(i, j, 1)]
            n6 = node_map_3d[(i+1, j, 1)]
            n7 = node_map_3d[(i+1, j+1, 1)]
            n8 = node_map_3d[(i, j+1, 1)]
            f.write('  - "{} SOLID HEX8 {} {} {} {} {} {} {} {} MAT 1 TYPE Std"\n'.format(
                eid, n1, n2, n3, n4, n5, n6, n7, n8))
            eid += 1
    f.write('DSURF-NODE TOPOLOGY:\n')
    # Left (E:1): i=0
    for j in range(NY + 1):
        for l in range(2):
            f.write('  - "NODE {} DSURFACE 1"\n'.format(node_map_3d[(0, j, l)]))
    # Bottom (E:2): j=0
    for i in range(NX + 1):
        for l in range(2):
            f.write('  - "NODE {} DSURFACE 2"\n'.format(node_map_3d[(i, 0, l)]))
    # Top (E:3): j=NY
    for i in range(NX + 1):
        for l in range(2):
            f.write('  - "NODE {} DSURFACE 3"\n'.format(node_map_3d[(i, NY, l)]))
    f.write('DVOL-NODE TOPOLOGY:\n')
    for i, n in enumerate(nodes_3d): f.write('  - "NODE {} DVOL 1"\n'.format(i+1))
    f.write('DNODE-NODE TOPOLOGY:\n')
    # Interface Points (both layers)
    for idx, nid in enumerate(interior_ids):
        for l in range(2):
            nid_3d = node_map_3d[(NX, idx + 1, l)]
            dnode_id = dnode_ids_layer0[idx] if l == 0 else dnode_ids_layer1[idx]
            f.write('  - "NODE {} DNODE {}\n".format(nid_3d, dnode_id))
    f.write('IO/MONITOR STRUCTURE DBC:\n')
    f.write('  INTERVAL_STEPS: 1\n  FILE_TYPE: yaml\n  WRITE_CONDITION_INFORMATION: true\n')
    f.write('IO:\n  VTK: true\n')

# --- RUN T ---
bin_path = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
ld_path = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")
env = os.environ.copy()
env["LD_LIBRARY_PATH"] = ld_path
cmd_T = ["stdbuf", "-oL", "-eL", bin_path, deck_T_file, OUT_T]
try:
    with open(f"{OUT_T}.log", "w") as logf:
        subprocess.run(cmd_T, env=env, stdout=logf, stderr=logf, check=True)
except subprocess.CalledProcessError:
    print(why_4c_did_not_finish("run T"), file=sys.stderr, flush=True)
    sys.exit(1)

# --- RUN U ---
cmd_U = ["stdbuf", "-oL", "-eL", bin_path, deck_U_file, OUT_U]
try:
    with open(f"{OUT_U}.log", "w") as logf:
        subprocess.run(cmd_U, env=env, stdout=logf, stderr=logf, check=True)
except subprocess.CalledProcessError:
    print(why_4c_did_not_finish("run U"), file=sys.stderr, flush=True)
    sys.exit(1)

# ── RECOVERY FROM 4C's OWN OUTPUTS (served): boundary flux VTU, displacement VTU, reaction yaml ──
import meshio  # noqa: E402


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
    out = np.zeros((len(nodes_2d), comps))
    for i, (x, y) in enumerate(nodes_2d):
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
q_n = [float(FB[n - 1, 0] * nrm[0] + FB[n - 1, 1] * nrm[1]) for n in interior_ids]

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
_deck_u_txt = Path(deck_U_file).read_text(errors="ignore")
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
_coord = [float(nodes_2d[n - 1][_ax]) for n in interior_ids]
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else (Y1 - Y0)
_share = _h * float(TZ)
q_t = []
for n in interior_ids:
    _xy = (round(float(nodes_2d[n - 1][0]), 9), round(float(nodes_2d[n - 1][1]), 9))
    if _xy not in _F:
        raise SystemExit(f"no reaction file for the interface node at {_xy}: its POINT DIRICH entries "
                         f"(both layers) need TAG: monitor_reaction")
    q_t.append([_F[_xy][0] / _share, _F[_xy][1] / _share])
co = [[float(nodes_2d[n - 1][0]), float(nodes_2d[n - 1][1])] for n in interior_ids]
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
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy\n")
    for (_px, _py), _t, (_ux, _uy) in zip(nodes_2d, T2d, U2d):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_t):.11e},{float(_ux):.11e},{float(_uy):.11e}\n")
with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for (_px, _py), n, (_qn, _qx, _qy) in zip(co, interior_ids, Q):
        _f.write(f"{_px:.11e},{_py:.11e},{float(T2d[n-1]):.11e},{float(U2d[n-1,0]):.11e},"
                 f"{float(U2d[n-1,1]):.11e},{_qn:.11e},{-_qx:.11e},{-_qy:.11e}\n")
print(f"NDOF = {3 * len(nodes_2d)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes_2d)}  "
      f"T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]  "
      f"scatra-vs-tsi T mismatch {_dT:.2e}")
