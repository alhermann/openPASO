"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity, plane strain).
    imports  values        = [T, ux, uy]   per interface point
    exports  normal_fluxes = [qn, qx, qy]  qn = -(k grad T).n_out, (qx, qy) = -(sigma_tot.n_out)
The traction uses the heat flux's sign rule (minus the flux through n_out): the two sides' exports
cancel and the Neumann partner applies them UNCHANGED. The task's OUTWARD traction is MINUS (qx, qy).
TWO 4C RUNS PER ITERATION (hole 2, both under a second): run T, a 2-D Scalar_Transport deck with
CALCFLUX_BOUNDARY "diffusive" (4C's consistent boundary flux); run U, a Thermo_Structure_Interaction
deck on a ONE-ELEMENT-THICK SOLIDSCATRA HEX8 slab with u_z pinned (exact plane strain), tsi_oneway,
COUPVARIABLE Temperature, MAT_Struct_ThermoStVenantK (alpha = beta/(3 lambda + 2 mu), INITTEMP 0),
`TAG: monitor_reaction` on the interface point conditions: 4C writes <out>-<id>_monitor_dbc.yaml per
condition (node gid ZERO-based, force f) and (f_layer0 + f_layer1)/(h*t_z) IS the exported traction
(measured order 2.0). SERVED: handshake, node classification, point conditions and deck tables,
finish diagnosis, recovery, self-checks, exports. YOURS: hole 1 (node layout), hole 2 (headers, runs).
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

# ── HOLE 1 (yours): the 2-D node layout of this subdomain. Build the NX x NY
#    structured grid on [X0, X1] x [Y0, Y1] and leave behind
#      nodes   the list of (x, y) tuples; 4C node id = list index + 1
#      quads   the list of 4-tuples of node ids, each quad counter-clockwise
#              (i,j), (i+1,j), (i+1,j+1), (i,j+1)
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin
hx, hy = (X1 - X0) / NX, (Y1 - Y0) / NY
nodes = [(X0 + i * hx, Y0 + j * hy) for j in range(NY + 1) for i in range(NX + 1)]
def _gid(i, j):
    return 1 + i + (NX + 1) * j
quads = [(_gid(i, j), _gid(i + 1, j), _gid(i + 1, j + 1), _gid(i, j + 1)) for j in range(NY) for i in range(NX)]
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end

# ── THE INTERFACE HANDSHAKE ONTO YOUR NODES (served): interface / interior / outer
#    by coordinate, the slab numbering, and the point conditions and tables of both decks ──
N2 = len(nodes)
_TOL = 1e-9 * max(X1 - X0, Y1 - Y0)
_AX, _VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]
def _on_edge(n):
    x, y = nodes[n - 1]
    return abs(x - X0) < _TOL or abs(x - X1) < _TOL or abs(y - Y0) < _TOL or abs(y - Y1) < _TOL
iface_all = sorted([n for n in range(1, N2 + 1) if abs(nodes[n - 1][_AX] - _VAL) < _TOL],
                   key=lambda n: nodes[n - 1][1 - _AX])
if len(iface_all) < 3:
    raise SystemExit(f"only {len(iface_all)} node(s) lie on the {IF} edge x/y = {_VAL}: `nodes` and config disagree")
interior = iface_all[1:-1]                      # the two ENDPOINTS keep the outer datum (they are outer nodes)
outer = [n for n in range(1, N2 + 1) if _on_edge(n) and n not in set(interior)]
TZ = min((X1 - X0) / NX, (Y1 - Y0) / NY)        # slab thickness: one well-shaped HEX8 layer
def gid3(n, layer):
    """4C node id of 2-D node n on slab layer 0 (z = 0) or 1 (z = TZ): the second layer follows the first."""
    return n + layer * N2
_g = {n: partner_values(nodes[n - 1][1 - _AX]) for n in interior}     # (T, ux, uy) per interior node

def scatra_point_conditions():
    """Deck T: `DESIGN POINT DIRICH CONDITIONS` entries imposing the partner's T on the interior interface
    nodes, and the matching `DNODE-NODE TOPOLOGY` entries (DNODE ids 1..len(interior))."""
    cond, topo = [], []
    for d, n in enumerate(interior, 1):
        cond.append(f"  - E: {d}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{_g[n][0]:.17g}]\n    FUNCT: [0]\n")
        topo.append(f'  - "NODE {n} DNODE {d}"\n')
    return "".join(cond), "".join(topo)

def slab_point_conditions():
    """Deck U: the partner's (ux, uy) as `DESIGN POINT DIRICH CONDITIONS` entries (NUMDOF 3, u_z = 0,
    TAG: monitor_reaction) and its T as `DESIGN POINT THERMO DIRICH CONDITIONS` entries, on BOTH slab
    layers of every interior interface node, with one DNODE per node-and-layer, ids continuous across
    the two families; plus the matching `DNODE-NODE TOPOLOGY` entries."""
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
    """Deck T: `DLINE-NODE TOPOLOGY` (DLINE 1 = the outer boundary nodes, DLINE 2 = every interface node)
    and `DSURF-NODE TOPOLOGY` (every node on DSURFACE 1, the surface the source load acts on)."""
    dline = "".join(f'  - "NODE {n} DLINE 1"\n' for n in outer) + "".join(f'  - "NODE {n} DLINE 2"\n' for n in iface_all)
    dsurf = "".join(f'  - "NODE {n} DSURFACE 1"\n' for n in range(1, N2 + 1))
    return dline, dsurf

def slab_topology():
    """Deck U: `DSURF-NODE TOPOLOGY` (the outer boundary nodes of BOTH layers on DSURFACE 1) and
    `DVOL-NODE TOPOLOGY` (every node of both layers on DVOL 1)."""
    dsurf = "".join(f'  - "NODE {gid3(n, layer)} DSURFACE 1"\n' for layer in (0, 1) for n in outer)
    dvol = "".join(f'  - "NODE {n} DVOL 1"\n' for n in range(1, 2 * N2 + 1))
    return dsurf, dvol

def deck_T_tables():
    """Deck T's TABLES, from YOUR nodes and quads and the partner's data: the interface point
    conditions, NODE COORDS, TRANSPORT ELEMENTS and the DLINE / DNODE / DSURF topology. Append this to
    YOUR deck-T header (the sections listed below; do not repeat any of these sections in the header)."""
    cond, dn = scatra_point_conditions()
    dline, dsurf = scatra_topology()
    out = "DESIGN POINT DIRICH CONDITIONS:\n" + cond
    out += "NODE COORDS:\n" + "".join(f'  - "NODE {n} COORD {x:.15f} {y:.15f} 0.0"\n' for n, (x, y) in enumerate(nodes, 1))
    out += "TRANSPORT ELEMENTS:\n" + "".join(f'  - "{e} TRANSP QUAD4 {a} {b} {c} {d} MAT 1 TYPE Std"\n'
                                             for e, (a, b, c, d) in enumerate(quads, 1))
    out += "DLINE-NODE TOPOLOGY:\n" + dline + "DNODE-NODE TOPOLOGY:\n" + dn + "DSURF-NODE TOPOLOGY:\n" + dsurf
    return out

def deck_U_tables():
    """Deck U's TABLES, from YOUR nodes and quads and the partner's data: both point-condition families,
    NODE COORDS of both slab layers, SOLIDSCATRA HEX8 elements and the DNODE / DSURF / DVOL topology.
    Append this to YOUR deck-U header (do not repeat any of these sections in the header)."""
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

# ── HOLE 2 (yours): the two deck HEADERS and the two runs. Each deck is YOUR header
#    string (physics, dynamics, solvers, material, sources, outer boundary sections --
#    the exact section list is in the door's text right after this block) followed by
#    the served tables: HEADER_T + deck_T_tables() and HEADER_U + deck_U_tables().
#    Run each with the binary at config fourc_bin (libraries on fourc_ld), line-buffered
#    (stdbuf -oL -eL <bin> <deck> <prefix> > <deck>.log 2>&1); on a non-zero exit FALL
#    THROUGH, the served check reads the log. LEAVE BEHIND: OUT_T and OUT_U (the two
#    output prefixes) and DECK_U (the FILE NAME you wrote deck U to, e.g. "deck_U.4C.yaml").
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin
# Write HEADER_T and HEADER_U (the sections above, as strings), write the decks
# as HEADER_T + deck_T_tables() and HEADER_U + deck_U_tables(), run both, and
# leave behind OUT_T, OUT_U, DECK_U.
OUT_T, OUT_U, DECK_U = "out_T", "out_U", "deck_U.4C.yaml"
raise SystemExit("the deck-and-run hole above the recovery is not filled")
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end

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
