"""4C as the NEUMANN side of a partitioned coupling (Scalar_Transport).

Subdomain B: (0,1) x (0.625, 1.5)
Equation: -div(k grad u) = f with k=2, no reaction
Interface at y=0.625: receives flux from partner, applies as Neumann BC, exports interface field values
Outer BC: u=0 on all outer boundaries (x=0, x=1, y=1.5)

Reads ./config.json {"level":k,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":diffusivity,"iface":"left|right|bottom|top","source_expr":"<f(x,y) or 0.0>",
"fourc_bin":..,"fourc_ld":..}.
Contract: reads ./imports.json (the partner's interface samples at its points),
maps them onto THIS side's interface nodes -- as the Neumann load -- runs YOUR 4C deck,
and exports its own consistent outward flux plus its interface TRACE as values.
"""
import json
import numpy as np
from pathlib import Path

CFG = json.loads(Path("config.json").read_text())
NX, NY = CFG["nx"], CFG["ny"]
X0, X1, Y0, Y1 = CFG["x0"], CFG["x1"], CFG["y0"], CFG["y1"]
KV = CFG["k"]; IF = CFG.get("iface", "bottom")
SIDE = CFG.get("side", "neumann")   # "neumann" | "dirichlet": the role the task gives this subdomain

# SOURCE f(x,y) AS A 4C EXPRESSION STRING (not a Python function): '^' for
# powers (never '**'), lowercase 'pi' ('PI' aborts: "Missing variables PI"),
# coordinates 'x','y', time 't'; "0.0" means no source. Wire it into YOUR deck
# as FUNCT1 + a DESIGN SURF NEUMANN VAL*FUNCT block; a Python src() that never
# reaches the deck is the classic 4C trap and does nothing.
SRC_EXPR = str(CFG.get("source_expr", CFG.get("source_const", "0.0")))
HAS_SRC = SRC_EXPR.strip() not in ("", "0", "0.", "0.0")

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())
def _partner(key, y_or_x):
    """One imported partner sample (key = "normal_fluxes" on the Neumann side,
    "values" on the Dirichlet side) interpolated onto one of THIS side's
    interface points. The driver does NOT interpolate between the two meshes --
    each participant maps the partner's samples onto its own points, here. Empty
    on iteration 1, so fall back to 0.0."""
    for _n, d in imp.items():
        co = d.get("coordinates") or []; q = d.get(key) or []
        if co and q and len(q) == len(co):
            ax = 1 if IF in ("left", "right") else 0
            pts = sorted(zip([c[ax] for c in co], q))
            xs = [p[0] for p in pts]; qs = [float(p[1]) for p in pts]
            t = min(max(y_or_x, xs[0]), xs[-1])
            for a, b, qa, qb in zip(xs, xs[1:], qs, qs[1:]):
                if a <= t <= b:
                    w = 0.0 if b == a else (t - a) / (b - a)
                    return qa + w * (qb - qa)
            return qs[-1]
    return 0.0
def partner_flux(y_or_x):
    """NEUMANN side: the partner's outward flux at one of your interface points --
    your inward load there. Build your Neumann loads from this in the solve."""
    return _partner("normal_fluxes", y_or_x)
def partner_value(y_or_x):
    """DIRICHLET side: the partner's field value at one of your interface points --
    the trace you impose there. Build your interface Dirichlet data from this."""
    return _partner("values", y_or_x)
# SIGN CONVENTION: the inward load on THIS side is the partner's OUTWARD flux --
# the two interface normals are opposite. Apply the partner's number UNCHANGED
# as your 4C Neumann VAL; there is no extra minus sign anywhere.

# ── DID 4C FINISH? (served: when the run leaves no output, name the cause) ─
import atexit, glob, sys
def why_4c_did_not_finish():
    _why = []
    try:                                   # 1. 4C's own message, builtins only
        for _lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
            try:
                with open(_lg, errors="ignore") as _fh:
                    _lines = _fh.read().splitlines()
            except OSError:
                continue
            for _i, _ln in enumerate(_lines):
                if "PROC 0 ERROR" in _ln:
                    _said = []
                    for l in _lines[_i + 1:_i + 14]:   # the message, then the offending input block
                        if l.startswith("---") or l.lstrip().startswith(("0#", "1#")) or "MPI_ABORT" in l:
                            break
                        if l.strip():
                            _said.append(l.strip())
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said))
                    break
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(log scan failed: {_e!r})")
    try:                                   # 2. the deck, linted against 4C's own grammar
        import os as _os, re as _re, subprocess as _sp
        _deck = globals().get("DECK") or next(iter(sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml"))), None)
        if _deck and _os.path.isfile(str(_deck)):
            with open(_deck, errors="ignore") as _fh:
                _txt = _fh.read()
            _secs = _re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, _re.M)
            _dup = sorted({s for s in _secs if _secs.count(s) > 1})
            if _dup:
                _why.append("section(s) written more than once, 4C reads each once: " + ", ".join(_dup))
            _bin = globals().get("FOURC_BIN") or CFG.get("fourc_bin") or _os.environ.get("FOURC_BIN")
            _valid = set()
            if _bin and _os.path.isfile(str(_bin)):
                _env = dict(_os.environ)
                _ld = CFG.get("fourc_ld") or _os.environ.get("FOURC_LD")
                if _ld:
                    _env["LD_LIBRARY_PATH"] = f"{_ld}:{_env.get('LD_LIBRARY_PATH', '')}"
                _dump = _sp.run([str(_bin), "-p"], capture_output=True, text=True, timeout=120, env=_env).stdout
                _valid = set(_re.findall(r"^    - name: (.+?)\s*$", _dump, _re.M)) | set(
                    _re.findall(r"^  - ([A-Z][A-Z0-9 _/.:-]*?)\s*$", _dump.split("legacy_string_sections:", 1)[-1], _re.M))
                _valid |= {"TITLE"}
            if len(_valid) > 100:
                _bad = [s for s in dict.fromkeys(_secs) if s not in _valid and not _re.fullmatch(r"FUNCT\d+", s)]
                if _bad:
                    _why.append("section name(s) the binary's own grammar (`4C -p`) does not know: " + ", ".join(_bad))
            if _re.search(r'PROBLEMTYPE:\s*"?Thermo"?\s*$', _txt, _re.M):
                _why.append("PROBLEMTYPE Thermo: this contract's recovery reads Scalar_Transport output "
                            "(phi_1 and flux_boundary_phi_1 in out-vtk-files/), which a Thermo problem never "
                            "writes, and Thermo does not know SOLVERTYPE/CALCFLUX_BOUNDARY -- use PROBLEMTYPE "
                            "Scalar_Transport with a SCALAR TRANSPORT DYNAMIC section, MAT_scatra and TRANSP elements")
            if "Scalar_Transport" in _txt and "THERMAL DYNAMIC:" in _txt and "SCALAR TRANSPORT DYNAMIC:" not in _txt:
                _why.append("PROBLEMTYPE Scalar_Transport needs its dynamics under `SCALAR TRANSPORT DYNAMIC`, "
                            "not `THERMAL DYNAMIC` (that is the Thermo problem type)")
            if "CALCFLUX_BOUNDARY" in _txt and "FLUX CALC" not in _txt:
                _why.append("CALCFLUX_BOUNDARY is set but no `SCATRA FLUX CALC LINE CONDITIONS` (SURF in 3-D) entry "
                            "names the interface, and 4C refuses flux output without one")
            _blocks = _re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=_re.M)
            _topo = set(_re.findall(r"D(?:NODE|LINE|SURF|VOL)\s+(\d+)", _txt))
            for _b in _blocks:
                _head = _b.split(":", 1)[0].strip()
                if not (_head.startswith("DESIGN") and _head.endswith("CONDITIONS")):
                    continue
                _entries = [e for e in _re.split(r"^\s*-\s", _b, flags=_re.M)[1:] if e.strip()]
                _noid = [e for e in _entries if not _re.search(r"\bE:\s*\d+|NODE_SET_NAME", e)]
                if _noid:
                    _why.append(f"{len(_noid)} entr{'y' if len(_noid) == 1 else 'ies'} in {_head} without `E: <id>` (or NODE_SET_NAME)")
                _ids = _re.findall(r"\bE:\s*(\d+)", _b)
                _missing = sorted({i for i in _ids if i not in _topo}, key=int)
                if _missing:
                    _why.append(f"{_head} names E id(s) {', '.join(_missing)} that no *-NODE TOPOLOGY section defines")
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(deck lint failed: {_e!r})")
    if not _why:
        _why.append("no VTU under out-vtk-files/ and no 4C error line in any *.log here -- run the binary line-buffered "
                    "(stdbuf -oL -eL) with its console captured to a log next to the deck, then read that log from the top")
    return "4C DID NOT FINISH -- " + "; ".join(_why)
def _diagnose_at_exit():
    if not Path("exports.json").is_file() and not glob.glob("out-vtk-files/*.vtu"):
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)
atexit.register(_diagnose_at_exit)

# ============================================================================
# ── SOLVE ─ THE 4C DECK AND RUN FOR SUBDOMAIN B (NEUMANN SIDE) ───────────────
# ============================================================================

# Subdomain B geometry
HX = (X1 - X0) / NX
HY = (Y1 - Y0) / NY

# Generate mesh nodes (structured grid, 1-based node IDs for 4C)
# Node ordering: row by row, left to right, bottom to top
# Node ID = j * (NX + 1) + i + 1 where i=0..NX, j=0..NY
nodes = []  # nodes[node_id - 1] = (x, y)
for j in range(NY + 1):
    y = Y0 + j * HY
    for i in range(NX + 1):
        x = X0 + i * HX
        nodes.append((x, y))

NUM_NODES = (NX + 1) * (NY + 1)
NUM_ELEMENTS = NX * NY

# Identify interface nodes (at y = Y0 = 0.625, the bottom edge of subdomain B)
# Interface is at y=Y0, so j=0 row
interface_node_ids = list(range(1, NX + 2))  # nodes 1, 2, ..., NX+1 along bottom edge
# Interior interface nodes (drop endpoints at corners)
interior_interface_ids = list(range(2, NX + 2))  # nodes 2, 3, ..., NX+1 (drop first corner)
# Actually drop both endpoints: nodes 2 to NX (inclusive)
interior_interface_ids = list(range(2, NX + 1))  # interior nodes only

# Identify boundary lines for outer Dirichlet conditions
# Left edge (x=X0): nodes 1, NY+2, 2*(NY+1)+1, ... -> column 0
# Right edge (x=X1): nodes NX+1, 2*NX+2, ... -> column NX
# Top edge (y=Y1): nodes (NY)*(NX+1)+1 to (NY+1)*(NX+1) -> row NY
# Bottom edge (y=Y0): nodes 1 to NX+1 -> row 0 (this is the interface, handled separately)

# Outer boundary lines (DLINE ids starting after interface DLINE)
# We'll assign:
#   DLINE 1: left edge (x=X0)
#   DLINE 2: right edge (x=X1)  
#   DLINE 3: top edge (y=Y1)
#   DLINE 4: bottom edge (interface, y=Y0) - for flux calc

# Build the 4C deck
deck_lines = []

# TITLE
deck_lines.append('TITLE:')
deck_lines.append('  - "Side B - Neumann side coupling"')

# PROBLEM SIZE
deck_lines.append('PROBLEM SIZE:')
deck_lines.append(f'  ELEMENTS: {NUM_ELEMENTS}')
deck_lines.append(f'  NODES: {NUM_NODES}')

# PROBLEM TYPE
deck_lines.append('PROBLEM TYPE:')
deck_lines.append('  PROBLEMTYPE: "Scalar_Transport"')

# SCALAR TRANSPORT DYNAMIC
deck_lines.append('SCALAR TRANSPORT DYNAMIC:')
deck_lines.append('  TIMEINTEGR: "Stationary"')
deck_lines.append('  SOLVERTYPE: "linear_full"')
deck_lines.append('  VELOCITYFIELD: "zero"')
deck_lines.append('  TIMESTEP: 1.0')
deck_lines.append('  NUMSTEP: 1')
deck_lines.append('  MAXTIME: 1.0')
deck_lines.append('  LINEAR_SOLVER: 1')
deck_lines.append('  CALCFLUX_BOUNDARY: "diffusive"')

# SOLVER 1
deck_lines.append('SOLVER 1:')
deck_lines.append('  SOLVER: "UMFPACK"')

# MATERIALS
deck_lines.append('MATERIALS:')
deck_lines.append('  - MAT: 1')
deck_lines.append('    MAT_scatra:')
deck_lines.append(f'      DIFFUSIVITY: {KV}')

# FUNCT1 for source term (convert ** to ^)
if HAS_SRC:
    # Convert Python expression to 4C expression
    fourc_src = SRC_EXPR.replace('**', '^').replace('pi', 'pi')
    deck_lines.append('FUNCT1:')
    deck_lines.append(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{fourc_src}"')

# DESIGN LINE DIRICH CONDITIONS for outer boundaries (u=0)
# Left edge (DLINE 1)
deck_lines.append('DESIGN LINE DIRICH CONDITIONS:')
deck_lines.append('  - E: 1')  # left
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('  - E: 2')  # right
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')
deck_lines.append('  - E: 3')  # top
deck_lines.append('    NUMDOF: 1')
deck_lines.append('    ONOFF: [1]')
deck_lines.append('    VAL: [0.0]')
deck_lines.append('    FUNCT: [0]')

# DESIGN POINT NEUMANN CONDITIONS for interface (imported flux)
# One per interior interface node
deck_lines.append('DESIGN POINT NEUMANN CONDITIONS:')
h_if = HX  # spacing along interface (x-direction)
for idx, node_id in enumerate(interior_interface_ids):
    x_coord = nodes[node_id - 1][0]
    # Compute nodal load using Simpson's rule approximation
    # VAL = (h/6) * (q(x-h) + 4*q(x) + q(x+h))
    q_left = partner_flux(x_coord - h_if) if x_coord - h_if >= X0 else partner_flux(x_coord)
    q_center = partner_flux(x_coord)
    q_right = partner_flux(x_coord + h_if) if x_coord + h_if <= X1 else partner_flux(x_coord)
    nodal_load = (h_if / 6.0) * (q_left + 4.0 * q_center + q_right)
    
    # DNODE id starts at 1 for first interior interface node
    dnode_id = idx + 1
    deck_lines.append(f'  - E: {dnode_id}')
    deck_lines.append('    NUMDOF: 1')
    deck_lines.append('    ONOFF: [1]')
    deck_lines.append(f'    VAL: [{nodal_load}]')
    deck_lines.append('    FUNCT: [0]')

# DESIGN SURF TRANSPORT NEUMANN CONDITIONS for volume source
if HAS_SRC:
    deck_lines.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
    deck_lines.append('  - E: 1')
    deck_lines.append('    NUMDOF: 1')
    deck_lines.append('    ONOFF: [1]')
    deck_lines.append('    VAL: [1.0]')
    deck_lines.append('    FUNCT: [1]')

# NODE COORDS
deck_lines.append('NODE COORDS:')
for node_id, (x, y) in enumerate(nodes, start=1):
    deck_lines.append(f'  - "NODE {node_id} COORD {x:.10f} {y:.10f} 0.0"')

# TRANSPORT ELEMENTS (QUAD4 elements)
deck_lines.append('TRANSPORT ELEMENTS:')
elem_id = 1
for j in range(NY):
    for i in range(NX):
        # Quad element: bottom-left, bottom-right, top-right, top-left (counter-clockwise)
        n_bl = j * (NX + 1) + i + 1
        n_br = j * (NX + 1) + i + 2
        n_tr = (j + 1) * (NX + 1) + i + 2
        n_tl = (j + 1) * (NX + 1) + i + 1
        deck_lines.append(f'  - "{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
        elem_id += 1

# DLINE-NODE TOPOLOGY for outer boundaries
deck_lines.append('DLINE-NODE TOPOLOGY:')
# Left edge (DLINE 1): nodes at i=0, j=0..NY
for j in range(NY + 1):
    node_id = j * (NX + 1) + 1
    deck_lines.append(f'  - "NODE {node_id} DLINE 1"')
# Right edge (DLINE 2): nodes at i=NX, j=0..NY
for j in range(NY + 1):
    node_id = j * (NX + 1) + (NX + 1)
    deck_lines.append(f'  - "NODE {node_id} DLINE 2"')
# Top edge (DLINE 3): nodes at j=NY, i=0..NX
for i in range(NX + 1):
    node_id = NY * (NX + 1) + i + 1
    deck_lines.append(f'  - "NODE {node_id} DLINE 3"')
# Bottom edge/interface (DLINE 4): nodes at j=0, i=0..NX
for i in range(NX + 1):
    node_id = i + 1
    deck_lines.append(f'  - "NODE {node_id} DLINE 4"')

# DNODE-NODE TOPOLOGY for interface point conditions
deck_lines.append('DNODE-NODE TOPOLOGY:')
for idx, node_id in enumerate(interior_interface_ids):
    dnode_id = idx + 1
    deck_lines.append(f'  - "NODE {node_id} DNODE {dnode_id}"')

# DSURF-NODE TOPOLOGY for volume source (all nodes)
if HAS_SRC:
    deck_lines.append('DSURF-NODE TOPOLOGY:')
    for node_id in range(1, NUM_NODES + 1):
        deck_lines.append(f'  - "NODE {node_id} DSURFACE 1"')

# SCATRA FLUX CALC LINE CONDITIONS for interface flux recovery
deck_lines.append('SCATRA FLUX CALC LINE CONDITIONS:')
deck_lines.append('  - E: 4')  # interface line

# Write deck to file
DECK = "side_B_deck.4C.yaml"
with open(DECK, 'w') as f:
    f.write('\n'.join(deck_lines))

# Run 4C binary
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
FOURC_LD = CFG.get("fourc_ld", "")

import subprocess
import os

env = os.environ.copy()
if FOURC_LD:
    env["LD_LIBRARY_PATH"] = f"{FOURC_LD}:{env.get('LD_LIBRARY_PATH', '')}"

# Run with stdbuf for line buffering
cmd = ["stdbuf", "-oL", "-eL", FOURC_BIN, DECK, "out"]
result = subprocess.run(cmd, capture_output=True, text=True, env=env)

# Log output
with open("run.log", "w") as f:
    f.write(result.stdout)
    f.write(result.stderr)

print(f"4C exit code: {result.returncode}")
if result.stdout:
    print("STDOUT:", result.stdout[:500])
if result.stderr:
    print("STDERR:", result.stderr[:500])

# ============================================================================
# ── CONSISTENT OUTWARD FLUX + EXPORTS -- the served recovery route ─────────
# ============================================================================
import glob
import meshio

# ── DID 4C FINISH? ── the check defined above the hole, run first here ─────
_vtus = sorted(glob.glob("out-vtk-files/*.vtu"))
if not _vtus:
    raise SystemExit(why_4c_did_not_finish())

# Take the last VTU file (step 1 for stationary)
vtu = sorted(glob.glob("out-vtk-files/*.vtu"))[-1]
_m = meshio.read(vtu)
vpts = [(float(p[0]), float(p[1])) for p in _m.points]
phi = [float(x) for x in _m.point_data["phi_1"].ravel()]   # the scalar is 'phi_1'

# Collapse duplicated points by coordinate (4C QUAD4 VTU repeats each node once per element)
val = {}
for (x, y), _u in zip(vpts, phi):
    val[(round(x, 12), round(y, 12))] = _u

u = [val[(round(x, 12), round(y, 12))] for (x, y) in nodes]

# Get flux from CALCFLUX_BOUNDARY output
fbname = next((da for da in _m.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("no flux_boundary field in the VTU -- set CALCFLUX_BOUNDARY "
                     "'diffusive' and add a SCATRA FLUX CALC condition on the interface")

fb = _m.point_data[fbname]
# For bottom interface, outward normal is (0, -1)
nrm = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1), "top": (0, 1)}[IF]

fbmap = {}
for (x, y), vec in zip(vpts, fb):
    fbmap[(round(x, 10), round(y, 10))] = float(vec[0] * nrm[0] + vec[1] * nrm[1])

q_own = [fbmap[(round(nodes[n-1][0], 10), round(nodes[n-1][1], 10))] for n in interior_interface_ids]
co = [list(nodes[n-1]) for n in interior_interface_ids]
vals = [u[n-1] for n in interior_interface_ids]

# ── EXPORT SELF-CHECK ─
_chk_vals = np.asarray(vals, float).ravel()
_chk_flux = np.asarray(q_own, float).ravel()
if not (np.isfinite(_chk_vals).all() and np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was exported")

_chk_imp = (json.loads(Path("imports.json").read_text() or "{}")
            if Path("imports.json").is_file() else {})
_chk_qin = (np.concatenate([np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                             for _d in _chk_imp.values()])
            if _chk_imp else np.zeros(0))

if SIDE == "neumann" and _chk_qin.size and np.abs(_chk_qin).max() > 0 and (
        np.abs(_chk_flux).max() < 1e-9 * np.abs(_chk_qin).max()):
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 against a "
                     "nonzero imported flux: the imported load never entered the "
                     "assembled system (the condition that integrates it is missing). "
                     "Fix the application; do not couple on")

if SIDE == "dirichlet" and _chk_qin.shape == _chk_flux.shape and _chk_flux.size and (
        np.array_equal(_chk_flux, -_chk_qin)):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array "
                     "negated, bit for bit: a copy, not a recovery from this side's "
                     "own assembled system")

# NEUMANN LOAD CONSISTENCY
if SIDE == "neumann" and q_own:
    _ax = 1 if IF in ("left", "right") else 0
    _q_applied = [partner_flux(float(nodes[n - 1][_ax])) for n in interior_interface_ids]
    _scale = max(abs(x) for x in _q_applied) if _q_applied else 0.0
    if _scale > 0:
        _mis = max(abs(a + b) for a, b in zip(q_own, _q_applied)) / _scale
        if _mis > 0.3:
            raise SystemExit(f"EXPORT SELF-CHECK: the recovered interface flux does not match the load you "
                             f"applied: max|q_own + q_imported| / max|q_imported| = {_mis:.2f}, while a correct "
                             f"Neumann side reproduces its load to a few percent.")

# Write exports.json LAST
json.dump({"field_name": "u", "coordinates": co, "values": (vals if SIDE == "neumann" else []),
           "normal_fluxes": q_own, "n_points": len(co)},
          open("exports.json", "w"))

# PER-LEVEL PERSISTENCE
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(nodes, u):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\n")

with open(f"interface_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u,qn\n")
    for (_px, _py), _n, _q in zip(co, interior_interface_ids, q_own):
        _f.write(f"{_px:.11e},{_py:.11e},{float(u[_n - 1]):.11e},{float(_q):.11e}\n")

print(f"NDOF = {len(nodes)}")
print(f"4C Neumann participant: NDOF = {len(nodes)}  "
      f"max|u| = {max(abs(t) for t in vals) if vals else 0:.6e}")
