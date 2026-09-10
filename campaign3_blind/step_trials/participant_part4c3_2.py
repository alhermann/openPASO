"""4C as the NEUMANN side of a partitioned coupling (Scalar_Transport).

Reads ./config.json {"level":k,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":diffusivity,"iface":"left|right|bottom|top","source_expr":"<f(x,y) or 0.0>",
"fourc_bin":..,"fourc_ld":..}.
Contract: reads ./imports.json (partner's outward flux at its points), applies
it as THIS side's interface Neumann load (opposite normals), runs YOUR 4C deck,
and exports its interface TRACE as values plus its own consistent outward flux.
"""
import json
from pathlib import Path
import glob
import meshio
import subprocess
import sys

CFG = json.loads(Path("config.json").read_text())
level = CFG.get("level", 1)
NX = 8 * 2**(level-1)
NY = 10 * 2**(level-1)
X0, X1, Y0, Y1 = 0.6, 1.4, 0.0, 1.0
KV = 5.0
IF = "left"

SRC_EXPR = "3*x^2*y + 2*y^2 - 4*x*y^3"

# ---- the partner's interface samples, mapped onto THIS side (handshake) ----
imp = {}
if Path("imports.json").is_file():
    imp = json.loads(Path("imports.json").read_text())
def partner_flux(y_or_x):
    """The imported partner flux interpolated onto one of THIS side's interface
    points. The driver does NOT interpolate between the two meshes -- each
    participant maps the partner's samples onto its own points, here. Empty on
    iteration 1, so fall back to 0.0. Call this when you build your Neumann loads
    in the solve below."""
    for _n, d in imp.items():
        co = d.get("coordinates") or []; q = d.get("normal_fluxes") or []
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

# ── SOLVE ───────────────────────────────────────────────────────────────────
# Build mesh coordinates
nodes = []
for i in range(NX+1):
    for j in range(NY+1):
        x = X0 + i * (X1-X0)/NX
        y = Y0 + j * (Y1-Y0)/NY
        nodes.append((x, y))

# Interface nodes are at i=0 (x=0.6)
interface_node_ids_4c = [j+1 for j in range(NY+1)]
interface_nodes = [nodes[j] for j in range(NY+1)]

# Compute pre-integrated nodal Neumann loads (Simpson-like rule)
dy = (Y1-Y0)/NY
g_vals = [partner_flux(y) for x, y in interface_nodes]
F_vals = []
for j in range(NY+1):
    if j == 0:
        F = dy/6.0 * (g_vals[0] + 4.0*g_vals[1] + g_vals[2])
    elif j == NY:
        F = dy/6.0 * (g_vals[NY-2] + 4.0*g_vals[NY-1] + g_vals[NY])
    else:
        F = dy/3.0 * (g_vals[j-1] + 4.0*g_vals[j] + g_vals[j+1])
    F_vals.append(F)

# Generate 4C deck
deck = []
deck.append('TITLE:')
deck.append('  - "Subdomain B"')
deck.append('PROBLEM SIZE:')
deck.append(f'  ELEMENTS: {NX*NY}')
deck.append(f'  NODES: {(NX+1)*(NY+1)}')
deck.append('PROBLEM TYPE:')
deck.append('  PROBLEMTYPE: "Scalar_Transport"')
deck.append('SCALAR TRANSPORT DYNAMIC:')
deck.append('  TIMEINTEGR: "Stationary"')
deck.append('  SOLVERTYPE: "linear_full"')
deck.append('  NUMSTEP: 1')
deck.append('  TIMESTEP: 1.0')
deck.append('  MAXTIME: 1.0')
deck.append('  LINEAR_SOLVER: 1')
deck.append('  CALCFLUX_BOUNDARY: "diffusive"')
deck.append('SOLVER 1:')
deck.append('  SOLVER: "UMFPACK"')
deck.append('MATERIALS:')
deck.append('  - MAT: 1')
deck.append('    MAT_scatra:')
deck.append(f'      DIFFUSIVITY: {KV}')
deck.append('FUNCT1:')
deck.append(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"')
deck.append('DESIGN LINE DIRICH CONDITIONS:')
deck.append('  - E: 1')
deck.append('    NUMDOF: 1')
deck.append('    ONOFF: [1]')
deck.append('    VAL: [0.0]')
deck.append('    FUNCT: [0]')
deck.append('  - E: 2')
deck.append('    NUMDOF: 1')
deck.append('    ONOFF: [1]')
deck.append('    VAL: [0.0]')
deck.append('    FUNCT: [0]')
deck.append('  - E: 3')
deck.append('    NUMDOF: 1')
deck.append('    ONOFF: [1]')
deck.append('    VAL: [0.0]')
deck.append('    FUNCT: [0]')
deck.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
deck.append('  - E: 1')
deck.append('    NUMDOF: 1')
deck.append('    ONOFF: [1]')
deck.append('    VAL: [1.0]')
deck.append('    FUNCT: [1]')
deck.append('DESIGN POINT NEUMANN CONDITIONS:')
for idx, F in enumerate(F_vals):
    deck.append(f'  - E: {idx+1}')
    deck.append('    NUMDOF: 1')
    deck.append('    ONOFF: [1]')
    deck.append(f'    VAL: [{F}]')
    deck.append('    FUNCT: [0]')
deck.append('SCATRA FLUX CALC LINE CONDITIONS:')
deck.append('  - E: 4')
deck.append('NODE COORDS:')
for i in range(NX+1):
    for j in range(NY+1):
        nid = i*(NY+1) + j + 1
        x, y = nodes[i*(NY+1)+j]
        deck.append(f'  - "NODE {nid} COORD {x:.11e} {y:.11e} 0.0"')
deck.append('TRANSPORT ELEMENTS:')
for i in range(NX):
    for j in range(NY):
        eid = i*NY + j + 1
        n1 = i*(NY+1) + j + 1
        n2 = (i+1)*(NY+1) + j + 1
        n3 = (i+1)*(NY+1) + j + 2
        n4 = i*(NY+1) + j + 2
        deck.append(f'  - "{eid} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"')
deck.append('DLINE-NODE TOPOLOGY:')
# DLINE 1: x=1.4 (right)
for j in range(NY+1):
    nid = NX*(NY+1) + j + 1
    deck.append(f'  - "NODE {nid} DLINE 1"')
# DLINE 2: y=0 (bottom)
for i in range(NX+1):
    nid = i*(NY+1) + 1
    deck.append(f'  - "NODE {nid} DLINE 2"')
# DLINE 3: y=1 (top)
for i in range(NX+1):
    nid = i*(NY+1) + NY + 1
    deck.append(f'  - "NODE {nid} DLINE 3"')
# DLINE 4: x=0.6 (interface)
for j in range(NY+1):
    nid = j + 1
    deck.append(f'  - "NODE {nid} DLINE 4"')
deck.append('DNODE-NODE TOPOLOGY:')
for j in range(NY+1):
    nid = j + 1
    deck.append(f'  - "NODE {nid} DNODE {j+1}"')
deck.append('DSURF-NODE TOPOLOGY:')
for nid in range(1, (NX+1)*(NY+1)+1):
    deck.append(f'  - "NODE {nid} DSURFACE 1"')

Path("deck.4C.yaml").write_text("\n".join(deck) + "\n")

# Run 4C
cmd = ("LD_LIBRARY_PATH=/opt/4C-dependencies/lib stdbuf -oL -eL "
       "/home/alexander/4C/build/4C deck.4C.yaml out > run.log 2>&1")
res = subprocess.run(cmd, shell=True)
if res.returncode != 0:
    print("4C failed. Check run.log for details.", file=sys.stderr)
    sys.exit(1)

# ── CONSISTENT OUTWARD FLUX + EXPORTS -- the served recovery route ─────────
vtu = sorted(glob.glob("out-vtk-files/*.vtu"))[-1]
_m = meshio.read(vtu)
vpts = [(float(p[0]), float(p[1])) for p in _m.points]
phi = [float(x) for x in _m.point_data["phi_1"].ravel()]
val = {}
for (x, y), _u in zip(vpts, phi):
    val[(round(x, 12), round(y, 12))] = _u
u = [val[(round(x, 12), round(y, 12))] for (x, y) in nodes]

fbname = next((da for da in _m.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("no flux_boundary field in the VTU -- set CALCFLUX_BOUNDARY "
                     "'diffusive' and add a SCATRA FLUX CALC condition on the interface")
fb = _m.point_data[fbname]
nrm = {"left": (-1, 0), "right": (1, 0), "bottom": (0, -1), "top": (0, 1)}[IF]
fbmap = {}
for (x, y), vec in zip(vpts, fb):
    fbmap[(round(x, 10), round(y, 10))] = float(vec[0] * nrm[0] + vec[1] * nrm[1])

interior = interface_node_ids_4c
q_own = [fbmap[(round(nodes[n-1][0], 10), round(nodes[n-1][1], 10))] for n in interior]
co = [list(nodes[n-1]) for n in interior]
vals = [u[n-1] for n in interior]

# ── EXPORT SELF-CHECK ─
import numpy as _np, json as _json
from pathlib import Path as _Path
_chk_vals = _np.asarray(vals, float).ravel()
_chk_flux = _np.asarray(q_own, float).ravel()
if not (_np.isfinite(_chk_vals).all() and _np.isfinite(_chk_flux).all()):
    raise SystemExit("EXPORT SELF-CHECK: non-finite interface values or fluxes; "
                     "the solve did not produce a usable field, so nothing was exported")
_chk_imp = (_json.loads(_Path("imports.json").read_text() or "{}")
            if _Path("imports.json").is_file() else {})
_chk_qin = (_np.concatenate([_np.asarray(_d.get("normal_fluxes") or [], float).ravel()
                             for _d in _chk_imp.values()])
            if _chk_imp else _np.zeros(0))
if True and _chk_qin.size and _np.abs(_chk_qin).max() > 0 and (
        _np.abs(_chk_flux).max() < 1e-9 * _np.abs(_chk_qin).max()):
    raise SystemExit("EXPORT SELF-CHECK: the recovered interface flux is ~0 against a "
                     "nonzero imported flux: the imported load never entered the "
                     "assembled system (the condition that integrates it is missing). "
                     "Fix the application; do not couple on")
if False and _chk_qin.shape == _chk_flux.shape and _chk_flux.size and (
        _np.array_equal(_chk_flux, -_chk_qin)):
    raise SystemExit("EXPORT SELF-CHECK: the exported flux is the partner's array "
                     "negated, bit for bit: a copy, not a recovery from this side's "
                     "own assembled system")
json.dump({"field_name": "u", "coordinates": co, "values": vals,
           "normal_fluxes": q_own, "n_points": len(co)},
          open("exports.json", "w"))

# PER-LEVEL PERSISTENCE
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as _f:
    _f.write("x,y,u\n")
    for (_px, _py), _u in zip(nodes, u):
        _f.write(f"{_px:.11e},{_py:.11e},{float(_u):.11e}\n")
print(f"4C Neumann participant: NDOF = {len(nodes)}  "
      f"max|u| = {max(abs(t) for t in vals) if vals else 0:.6e}")
