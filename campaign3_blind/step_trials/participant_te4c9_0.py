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

# ===== MESH GENERATION =====
# Build 2-D node layout: NX x NY elements -> (NX+1) x (NY+1) nodes
nodes = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes.append((x, y))

# Slab thickness for 3D TSI (one element thick)
TZ = (Y1 - Y0) / NY  # same as h_y

# Identify interface nodes (along the specified interface)
# For "right" interface: x = X1, nodes with i = NX
# For "left" interface: x = X0, nodes with i = 0
# For "top" interface: y = Y1, nodes with j = NY
# For "bottom" interface: y = Y0, nodes with j = 0
if IF == "right":
    iface_nodes = [(i, j) for j in range(NY + 1) for i in [NX]]
elif IF == "left":
    iface_nodes = [(i, j) for j in range(NY + 1) for i in [0]]
elif IF == "top":
    iface_nodes = [(i, j) for i in range(NX + 1) for j in [NY]]
else:  # bottom
    iface_nodes = [(i, j) for i in range(NX + 1) for j in [0]]

# Map (i,j) to 1-based deck node id
def node_id(i, j):
    return j * (NX + 1) + i + 1

# Interface nodes (1-based ids), excluding endpoints for Dirichlet imposition
# Endpoints keep the outer boundary condition (T=0, u=0)
iface_ids = [node_id(i, j) for i, j in iface_nodes]
# Exclude first and last (endpoints) for interior interface nodes
interior = iface_ids[1:-1] if len(iface_ids) > 2 else []

# ===== DECK T: Scalar_Transport (2-D) =====
OUT_T = "run_t"
deck_t = f"""TITLE:
  - "Thermo-elastic coupling side A - temperature (Scalar_Transport)"
PROBLEM SIZE:
  ELEMENTS: {NX * NY}
  NODES: {(NX + 1) * (NY + 1)}
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
"""

# Add outer boundary Dirichlet conditions (T=0)
# For right interface, outer boundaries are left, bottom, top
outer_surf_ids = []
if IF == "right":
    # Left: j=0..NY, i=0
    outer_surf_ids.extend([(0, j) for j in range(NY + 1)])
    # Bottom: i=0..NX, j=0
    outer_surf_ids.extend([(i, 0) for i in range(NX + 1)])
    # Top: i=0..NX, j=NY
    outer_surf_ids.extend([(i, NY) for i in range(NX + 1)])
elif IF == "left":
    # Right: j=0..NY, i=NX
    outer_surf_ids.extend([(NX, j) for j in range(NY + 1)])
    # Bottom: i=0..NX, j=0
    outer_surf_ids.extend([(i, 0) for i in range(NX + 1)])
    # Top: i=0..NX, j=NY
    outer_surf_ids.extend([(i, NY) for i in range(NX + 1)])
elif IF == "top":
    # Left: j=0..NY, i=0
    outer_surf_ids.extend([(0, j) for j in range(NY + 1)])
    # Right: j=0..NY, i=NX
    outer_surf_ids.extend([(NX, j) for j in range(NY + 1)])
    # Bottom: i=0..NX, j=0
    outer_surf_ids.extend([(i, 0) for i in range(NX + 1)])
else:  # bottom
    # Left: j=0..NY, i=0
    outer_surf_ids.extend([(0, j) for j in range(NY + 1)])
    # Right: j=0..NY, i=NX
    outer_surf_ids.extend([(NX, j) for j in range(NY + 1)])
    # Top: i=0..NX, j=NY
    outer_surf_ids.extend([(i, NY) for i in range(NX + 1)])

# Remove duplicates and sort
outer_surf_ids = sorted(set(outer_surf_ids))

deck_t += """DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [0.0]
    FUNCT: [0]
DESIGN LINE DIRICH CONDITIONS:
"""

# Interface Dirichlet conditions (per-node, from imports.json)
deck_t += f"""  - E: 2
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{partner_values(nodes[iface_ids[0]][0] if IF in ('left','right') else nodes[iface_ids[0]][1])}]
    FUNCT: [0]
"""
for nid in interior:
    xy = nodes[nid - 1]
    val = partner_values(xy[0] if IF in ('left','right') else xy[1])
    deck_t += f"""  - E: {2 + (len([n for n in interior if n <= nid]))}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{val[0]}]
    FUNCT: [0]
"""
deck_t += f"""  - E: {2 + len(interior) + 1}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{partner_values(nodes[iface_ids[-1]][0] if IF in ('left','right') else nodes[iface_ids[-1]][1])}]
    FUNCT: [0]
"""

# Volume source (SURF in 2D)
deck_t += f"""DESIGN SURF TRANSPORT NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [1]
NODE COORDS:
"""
for idx, (x, y) in enumerate(nodes, 1):
    deck_t += f'  - "NODE {idx} COORD {x:.11e} {y:.11e} 0.0"\n'

deck_t += """TRANSPORT ELEMENTS:
"""
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n1 = node_id(i, j)
        n2 = node_id(i + 1, j)
        n3 = node_id(i + 1, j + 1)
        n4 = node_id(i, j + 1)
        deck_t += f'  - "{elem_id} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\n'
        elem_id += 1

deck_t += """DSURF-NODE TOPOLOGY:
"""
for idx in range(1, len(nodes) + 1):
    deck_t += f'  - "NODE {idx} DSURFACE 1"\n'

deck_t += """DLINE-NODE TOPOLOGY:
"""
# Outer boundary (E=1)
for i, j in outer_surf_ids:
    nid = node_id(i, j)
    deck_t += f'  - "NODE {nid} DLINE 1"\n'

# Interface line (E=2)
for i, j in iface_nodes:
    nid = node_id(i, j)
    deck_t += f'  - "NODE {nid} DLINE 2"\n'

deck_t += """DNODE-NODE TOPOLOGY:
"""
# Interface nodes (E=3 for first endpoint, then interior, then last)
e_id = 3
for idx, nid in enumerate(iface_ids):
    deck_t += f'  - "NODE {nid} DNODE {e_id + idx}"\n'

deck_t += """SCATRA FLUX CALC LINE CONDITIONS:
  - E: 2
"""

Path(f"{OUT_T}.4C.yaml").write_text(deck_t)

# ===== DECK U: Thermo_Structure_Interaction (3D slab) =====
OUT_U = "run_u"
DECK_U = f"{OUT_U}.4C.yaml"

# 3D slab nodes: two layers at z=0 and z=TZ
slab_nodes = []
layer0_ids = []
layer1_ids = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        slab_nodes.append((x, y, 0.0))
        slab_nodes.append((x, y, TZ))

# Node IDs: layer 0 = 1..(NX+1)*(NY+1), layer 1 = (NX+1)*(NY+1)+1 .. 2*(NX+1)*(NY+1)
nlayer0 = (NX + 1) * (NY + 1)

def slab_node_id(i, j, layer):
    base = j * (NX + 1) + i + 1
    return base if layer == 0 else base + nlayer0

deck_u = f"""TITLE:
  - "Thermo-elastic coupling side A - displacement (TSI plane strain)"
PROBLEM SIZE:
  DIM: 3
  ELEMENTS: {NX * NY}
  NODES: {2 * (NX + 1) * (NY + 1)}
PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: Standard
  DYNAMICTYPE: "Statics"
  TIMESTEP: 1.0
  NUMSTEP: 1
  MAXTIME: 1.0
  TOLDISP: 1e-8
  TOLRES: 1e-8
  MAXITER: 20
  LINEAR_SOLVER: 2
  PREDICT: TangDis
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
      DENS: 7850.0
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
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
FUNCT2:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"
"""

# Body forces (structural) - need to be added via VOL THERMO NEUMANN for thermal
# and regular VOL NEUMANN for mechanical
# Actually for TSI, structural body force goes in DESIGN VOL NEUMANN CONDITIONS
# But we need to wire the source functions properly

# Let's add the thermal source and structural sources
deck_u += f"""DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 1
    ONOFF: [1]
    VAL: [1.0]
    FUNCT: [2]
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
"""

# Outer boundary conditions (u=0 on outer surfaces)
deck_u += """DESIGN SURF DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [1, 1, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
"""

# Plane strain: u_z = 0 on ALL nodes (both layers)
deck_u += """DESIGN VOL DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    ONOFF: [0, 0, 1]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
"""

# Interface Dirichlet conditions (per-node, both layers, with monitor_reaction)
# First endpoint (outer BC keeps u=0, so skip)
# Interior nodes get partner values
# Last endpoint (outer BC keeps u=0, so skip)

deck_u += """DESIGN POINT DIRICH CONDITIONS:
"""
e_id_struct = 1
e_id_thermo = 1
for idx, nid in enumerate(interior):
    xy = nodes[nid - 1]
    val = partner_values(xy[0] if IF in ('left','right') else xy[1])
    # Layer 0
    slab_nid = slab_node_id(*[(i for i,j in zip(range(NX+1), range(NY+1)) if node_id(i,j)==nid)[0]], 
                             *(j for i,j in zip(range(NX+1), range(NY+1)) if node_id(i,j)==nid)[0], 0)
    
    # Find (i,j) for this node
    for j in range(NY + 1):
        for i in range(NX + 1):
            if node_id(i, j) == nid:
                slab_nid_l0 = slab_node_id(i, j, 0)
                slab_nid_l1 = slab_node_id(i, j, 1)
                break
    
    deck_u += f"""  - E: {e_id_struct}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{val[1]}, {val[2]}, 0.0]
    FUNCT: [0, 0, 0]
    TAG: monitor_reaction
"""
    e_id_struct += 1
    
    deck_u += f"""  - E: {e_id_struct}
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [{val[1]}, {val[2]}, 0.0]
    FUNCT: [0, 0, 0]
    TAG: monitor_reaction
"""
    e_id_struct += 1
    
    # Thermal interface conditions
    deck_u += f"""DESIGN POINT THERMO DIRICH CONDITIONS:
  - E: {e_id_thermo}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{val[0]}]
    FUNCT: [0]
"""
    e_id_thermo += 1
    
    deck_u += f"""  - E: {e_id_thermo}
    NUMDOF: 1
    ONOFF: [1]
    VAL: [{val[0]}]
    FUNCT: [0]
"""
    e_id_thermo += 1

# IO for reaction monitoring
deck_u += """IO/MONITOR STRUCTURE DBC:
  INTERVAL_STEPS: 1
  FILE_TYPE: yaml
  WRITE_CONDITION_INFORMATION: true
IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: 1
IO/RUNTIME VTK OUTPUT/STRUCTURE:
  OUTPUT_STRUCTURE: true
  DISPLACEMENT: true
IO/RUNTIME VTK OUTPUT/THERMO:
  OUTPUT_THERMO: true
  TEMPERATURE: true
NODE COORDS:
"""
for idx, (x, y, z) in enumerate(slab_nodes, 1):
    deck_u += f'  - "NODE {idx} COORD {x:.11e} {y:.11e} {z:.11e}"\n'

deck_u += """STRUCTURE ELEMENTS:
"""
elem_id = 1
for j in range(NY):
    for i in range(NX):
        n1 = slab_node_id(i, j, 0)
        n2 = slab_node_id(i + 1, j, 0)
        n3 = slab_node_id(i + 1, j + 1, 0)
        n4 = slab_node_id(i, j + 1, 0)
        n5 = slab_node_id(i, j, 1)
        n6 = slab_node_id(i + 1, j, 1)
        n7 = slab_node_id(i + 1, j + 1, 1)
        n8 = slab_node_id(i, j + 1, 1)
        deck_u += f'  - "{elem_id} SOLIDSCATRA HEX8 {n1} {n2} {n3} {n4} {n5} {n6} {n7} {n8} MAT 1 KINEM linear TYPE Undefined"\n'
        elem_id += 1

# Topology sections
deck_u += """DVOL-NODE TOPOLOGY:
"""
for idx in range(1, len(slab_nodes) + 1):
    deck_u += f'  - "NODE {idx} DVOL 1"\n'

deck_u += """DSURF-NODE TOPOLOGY:
"""
# Outer surface nodes
for i, j in outer_surf_ids:
    nid_l0 = slab_node_id(i, j, 0)
    nid_l1 = slab_node_id(i, j, 1)
    deck_u += f'  - "NODE {nid_l0} DSURFACE 1"\n'
    deck_u += f'  - "NODE {nid_l1} DSURFACE 1"\n'

deck_u += """DNODE-NODE TOPOLOGY:
"""
# Interface nodes on both layers
e_id = 1
for idx, nid in enumerate(interior):
    for j in range(NY + 1):
        for i in range(NX + 1):
            if node_id(i, j) == nid:
                deck_u += f'  - "NODE {slab_node_id(i, j, 0)} DNODE {e_id}"\n'
                e_id += 1
                deck_u += f'  - "NODE {slab_node_id(i, j, 1)} DNODE {e_id}"\n'
                e_id += 1
                break

Path(DECK_U).write_text(deck_u)

# ===== RUN BOTH DECKS =====
fourc_bin = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
fourc_ld = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")

# Run deck T
cmd_t = [f"stdbuf", "-oL", "-eL", f"LD_LIBRARY_PATH={fourc_ld}", fourc_bin, f"{OUT_T}.4C.yaml", OUT_T]
result_t = subprocess.run(cmd_t, capture_output=True, text=True)
Path(f"{OUT_T}.log").write_text(result_t.stdout + result_t.stderr)

# Run deck U
cmd_u = [f"stdbuf", "-oL", "-eL", f"LD_LIBRARY_PATH={fourc_ld}", fourc_bin, f"{OUT_U}.4C.yaml", OUT_U]
result_u = subprocess.run(cmd_u, capture_output=True, text=True)
Path(f"{OUT_U}.log").write_text(result_u.stdout + result_u.stderr)

# ── RECOVERY FROM 4C's OWN OUTPUTS (served): boundary flux VTU, displacement VTU, reaction yaml ──
import meshio  # noqa: E402


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
