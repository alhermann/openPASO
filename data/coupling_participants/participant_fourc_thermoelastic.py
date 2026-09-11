"""4C as the DIRICHLET side of a THERMO-ELASTIC coupling (steady thermoelasticity,
plane strain): the partner's temperature AND displacement come in together at
the interface, this side's heat flux AND traction go out together.

WHAT IS EXCHANGED (one interface state with THREE components per point):
    imports  values        = [T, ux, uy]     the partner's interface field
    exports  normal_fluxes = [qn, qx, qy]    this side's consistent outward flux:
             qn = -(k grad T).n_out           (Fourier, outward normal)
             (qx, qy) = -(sigma_tot . n_out)  the TRACTION IN THE FLUX CONVENTION
The traction is exported with the SAME sign rule as the heat flux (q = -flux
of the conserved quantity through n_out), so the two sides' exports cancel
componentwise and the Neumann partner applies them UNCHANGED as its natural
datum. The task's "outward traction sigma_tot . n_out" is the NEGATIVE of the
exported pair: write -qx, -qy into your interface CSV, qn as it is.

THE 4C ROUTE, MEASURED ON THIS BINARY (2026-09-11). 4C has no 2-D thermo-
elastic element, and the thermal reaction of a Dirichlet node is not written
by any output. So the Dirichlet side is TWO 4C runs per coupling iteration,
each a deck YOU write (the hole below), both cheap (well under a second):
  run T  PROBLEMTYPE Scalar_Transport, 2-D TRANSP QUAD4, the heat equation
         alone (it does not depend on u): the partner's T imposed per interior
         interface node, the heat source as a FUNCT, CALCFLUX_BOUNDARY
         "diffusive" + a SCATRA FLUX CALC LINE condition on the interface so 4C
         writes its own CONSISTENT boundary flux (flux_boundary_phi_1) into the
         VTU -- the same reaction recovery every other backend here uses;
  run U  PROBLEMTYPE Thermo_Structure_Interaction on a ONE-ELEMENT-THICK
         SOLIDSCATRA HEX8 slab with u_z = 0 pinned on every node: that is EXACT
         plane strain, not an approximation. COUPALGO tsi_oneway with
         COUPVARIABLE Temperature (thermal strain loads the structure), Statics
         in both fields, MAT_Struct_ThermoStVenantK whose stress is
         C:(eps - alpha (T - T0) I) = sigma_el - beta T I with
         alpha = beta / (3 lambda + 2 mu) and INITTEMP 0; the heat source as a
         DESIGN VOL THERMO NEUMANN FUNCT, the body force as a DESIGN VOL NEUMANN
         FUNCT, the partner's (T, ux, uy) per interior interface node on BOTH
         z-layers as DESIGN POINT THERMO DIRICH + DESIGN POINT DIRICH, and
         `TAG: monitor_reaction` on every interface POINT DIRICH entry plus an
         IO/MONITOR STRUCTURE DBC section (FILE_TYPE yaml,
         WRITE_CONDITION_INFORMATION true): 4C then writes one
         <out>-<id>_monitor_dbc.yaml per condition carrying the node gid
         (ZERO-based: gid 17 is the deck's NODE 18) and the reaction force f. The exported (qx, qy) is
             (f_layer0 + f_layer1) / (h * t_z)
         at each interior interface node (h = node spacing along the interface,
         t_z = slab thickness) -- measured against a manufactured thermo-elastic
         solution: relative error 1.45e-2, 3.6e-3, 9.0e-4 at h = 1/10, 1/20,
         1/40 (order 2.0), with that sign. The thermo field of run U solves the
         same heat problem as run T (its load is evaluated at Gauss points, run
         T's at nodes, so the two agree to O(h^2), measured 1.4e-2 at h = 1/8);
         the recovery below cross-checks them and stops on a mismatch.

WHAT IS SERVED HERE is the handshake (config + imports, three components mapped
per component), the sign convention, the finish diagnosis, the recovery from 4C's
OWN outputs (boundary flux VTU, displacement VTU, reaction yaml), the self-checks
and the exports schema. THE TWO DECKS AND THE RUNS ARE YOURS -- see the hole.

Reads ./config.json {"level":k,"nx":..,"ny":..,"x0":..,"x1":..,"y0":..,"y1":..,
"k":conductivity,"lam":lambda,"mu":mu,"beta":beta,"iface":"left|right|bottom|top",
"source_T":"<4C expression or 0.0>","source_ux":"..","source_uy":"..",
"fourc_bin":..,"fourc_ld":..}. Expressions are 4C FUNCT strings: '^' for powers
(never '**'), lowercase 'pi', coordinates 'x', 'y'.
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
    _why = []
    try:                                   # 1. 4C's own message
        for _lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
            try:
                with open(_lg, errors="ignore") as _fh:
                    _lines = _fh.read().splitlines()
            except OSError:
                continue
            for _i, _ln in enumerate(_lines):
                if "PROC 0 ERROR" in _ln:
                    _said = []
                    for l in _lines[_i + 1:_i + 14]:
                        if l.startswith("---") or l.lstrip().startswith(("0#", "1#")) or "MPI_ABORT" in l:
                            break
                        if l.strip():
                            _said.append(l.strip())
                    _why.append(f"4C said ({_lg}): " + " | ".join(_said))
                    break
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(log scan failed: {_e!r})")
    try:                                   # 2. the decks, linted for the measured defects
        for _deck in sorted(glob.glob("*.4C.yaml")) or sorted(glob.glob("*.yaml")):
            with open(_deck, errors="ignore") as _fh:
                _txt = _fh.read()
            _secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", _txt, re.M)
            _dup = sorted({s for s in _secs if _secs.count(s) > 1})
            if _dup:
                _why.append(f"{_deck}: section(s) written more than once, 4C reads each once: " + ", ".join(_dup))
            if "Thermo_Structure_Interaction" in _txt:
                if "CLONING MATERIAL MAP" not in _txt:
                    _why.append(f"{_deck}: TSI needs a CLONING MATERIAL MAP pairing the structure material with the MAT_Fourier thermal material")
                if "COUPVARIABLE" not in _txt or "Temperature" not in _txt.split("COUPVARIABLE", 1)[-1][:40]:
                    _why.append(f"{_deck}: TSI DYNAMIC/PARTITIONED needs COUPVARIABLE \"Temperature\" -- the default (Displacement) runs to rc=0 with ZERO thermal strain")
                if re.search(r"\b(WALL|SOLID) QUAD4\b|\bTRI3\b", _txt):
                    _why.append(f"{_deck}: 4C has no 2-D thermo-elastic element; use a one-element-thick SOLIDSCATRA HEX8 slab with u_z pinned")
                if "monitor_reaction" in _txt and "IO/MONITOR STRUCTURE DBC" not in _txt:
                    _why.append(f"{_deck}: TAG: monitor_reaction writes nothing without an IO/MONITOR STRUCTURE DBC section")
                if "DESIGN VOL THERMO DIRICH" in _txt:
                    _why.append(f"{_deck}: DESIGN VOL THERMO DIRICH imposes the temperature on the whole volume -- the heat equation is then not solved; the interface T goes in as POINT THERMO DIRICH, the outer T as SURF THERMO DIRICH")
            if "Scalar_Transport" in _txt:
                if "THERMAL DYNAMIC:" in _txt and "SCALAR TRANSPORT DYNAMIC:" not in _txt:
                    _why.append(f"{_deck}: PROBLEMTYPE Scalar_Transport needs its dynamics under `SCALAR TRANSPORT DYNAMIC`, not `THERMAL DYNAMIC`")
                if "CALCFLUX_BOUNDARY" in _txt and "FLUX CALC" not in _txt:
                    _why.append(f"{_deck}: CALCFLUX_BOUNDARY is set but no `SCATRA FLUX CALC LINE CONDITIONS` entry names the interface")
                if "CALCFLUX_BOUNDARY" not in _txt:
                    _why.append(f"{_deck}: no CALCFLUX_BOUNDARY \"diffusive\" -- 4C then writes no flux_boundary_phi_1 and the flux recovery has nothing to read")
            if re.search(r"^IO:\s*$", _txt, re.M) and "Scalar_Transport" in _txt:
                _why.append(f"{_deck}: an `IO:` section in a Scalar_Transport deck aborted every trial read (\"Could not match this input\"); the VTU appears without it")
            _topo = set(re.findall(r"D(?:NODE|LINE|SURF|VOL)\s+(\d+)", _txt))
            for _b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", _txt, flags=re.M):
                _head = _b.split(":", 1)[0].strip()
                if not (_head.startswith("DESIGN") and _head.endswith("CONDITIONS")):
                    continue
                _ids = re.findall(r"\bE:\s*(\d+)", _b)
                _missing = sorted({i for i in _ids if i not in _topo}, key=int)
                if _missing:
                    _why.append(f"{_deck}: {_head} names E id(s) {', '.join(_missing[:6])} that no *-NODE TOPOLOGY section defines")
    except Exception as _e:                # noqa: BLE001
        _why.append(f"(deck lint failed: {_e!r})")
    if not _why:
        _why.append("no output from the run and no 4C error line in any *.log here -- run the binary "
                    "line-buffered (stdbuf -oL -eL) with its console captured to a log next to the deck")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


atexit.register(_diagnose_at_exit)

# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin
# THE TWO DECKS AND THE RUNS ARE YOURS. Build the 2-D node layout of this
# subdomain (NX x NY quads on [X0,X1] x [Y0,Y1]), find the interface nodes,
# write and run BOTH decks, and leave behind the names listed at the end.
#
# RUN T -- the heat equation, 2-D, PROBLEMTYPE "Scalar_Transport":
#   SCALAR TRANSPORT DYNAMIC: TIMEINTEGR "Stationary", SOLVERTYPE "linear_full",
#     VELOCITYFIELD "zero", TIMESTEP/NUMSTEP/MAXTIME 1, LINEAR_SOLVER 1,
#     CALCFLUX_BOUNDARY "diffusive"
#   IO/RUNTIME VTK OUTPUT: INTERVAL_STEPS 1 (no other IO section)
#   SOLVER 1: SOLVER "UMFPACK";  MATERIALS: MAT 1 MAT_scatra DIFFUSIVITY KV
#   FUNCT1: SYMBOLIC_FUNCTION_OF_SPACE_TIME "<SRC_T>" and a DESIGN SURF NEUMANN
#     entry (E 1, NUMDOF 1, ONOFF [1], VAL [1.0], FUNCT [1]) so the source
#     enters the assembled RHS
#   DESIGN LINE DIRICH CONDITIONS on the OUTER boundary line (VAL [0.0] or the
#     task's outer T), DESIGN POINT DIRICH CONDITIONS one per INTERIOR interface
#     node with VAL [partner_values(y)[0]] (the interface ENDPOINTS keep the
#     outer datum: they are outer-boundary nodes on both sides)
#   SCATRA FLUX CALC LINE CONDITIONS: - E: <the interface DLINE id>
#   NODE COORDS ("NODE i COORD x y 0.0"), TRANSPORT ELEMENTS ("e TRANSP QUAD4
#     n1 n2 n3 n4 MAT 1 TYPE Std"), DLINE-NODE TOPOLOGY (outer line AND the
#     interface line), DNODE-NODE TOPOLOGY (one DNODE per interface point
#     condition, ids continuous across ALL condition families), DSURF-NODE
#     TOPOLOGY (every node, DSURFACE 1)
#
# RUN U -- the displacement, PROBLEMTYPE "Thermo_Structure_Interaction" on the
# one-element-thick slab (DIM 3): every 2-D node twice, at z = 0 and z = TZ
# (TZ = min(hx, hy) keeps the hexes well shaped), each quad as ONE
# "e SOLIDSCATRA HEX8 n1 n2 n3 n4 n5 n6 n7 n8 MAT 1 KINEM linear TYPE Undefined"
# (bottom four counter-clockwise, then the same four on the top layer):
#   IO: STRUCT_STRESS "No", STRUCT_STRAIN "No"
#   IO/MONITOR STRUCTURE DBC: INTERVAL_STEPS 1, FILE_TYPE yaml,
#     WRITE_CONDITION_INFORMATION true
#   STRUCTURAL DYNAMIC: DYNAMICTYPE "Statics", TIMESTEP/NUMSTEP/MAXTIME 1,
#     LINEAR_SOLVER 2;  THERMAL DYNAMIC: DYNAMICTYPE Statics, same steps,
#     LINEAR_SOLVER 1;  TSI DYNAMIC: COUPALGO "tsi_oneway", NUMSTEP/MAXTIME/
#     TIMESTEP 1, ITEMAX 1;  TSI DYNAMIC/PARTITIONED: COUPVARIABLE "Temperature"
#   IO/RUNTIME VTK OUTPUT: INTERVAL_STEPS 1;  IO/RUNTIME VTK OUTPUT/STRUCTURE:
#     OUTPUT_STRUCTURE true, DISPLACEMENT true;  THERMAL DYNAMIC/RUNTIME VTK
#     OUTPUT: OUTPUT_THERMO true, TEMPERATURE true
#   SOLVER 1 and SOLVER 2: SOLVER "UMFPACK"
#   MATERIALS: MAT 1 MAT_Struct_ThermoStVenantK {YOUNGNUM 1, YOUNG [E_MOD],
#     NUE NU, DENS 1, THEXPANS ALPHA, INITTEMP 0, THERMOMAT 2};
#     MAT 2 MAT_Fourier {CAPA 1, CONDUCT: constant: [KV]}
#   CLONING MATERIAL MAP: - SRC_FIELD "structure", SRC_MAT 1, TAR_FIELD
#     "thermo", TAR_MAT 2
#   FUNCT1 "<SRC_UX>", FUNCT2 "<SRC_UY>", FUNCT3 "<SRC_T>"
#   DESIGN VOL NEUMANN CONDITIONS: E 1, NUMDOF 3, ONOFF [1,1,0], VAL
#     [1.0,1.0,0.0], FUNCT [1,2,0]           (the body force)
#   DESIGN VOL THERMO NEUMANN CONDITIONS: E 1, NUMDOF 1, ONOFF [1], VAL [1.0],
#     FUNCT [3]                              (the heat source)
#   DESIGN VOL DIRICH CONDITIONS: E 1, NUMDOF 3, ONOFF [0,0,1], VAL [0,0,0],
#     FUNCT [0,0,0]                          (u_z = 0 everywhere: plane strain)
#   DESIGN SURF DIRICH CONDITIONS (E 1: all outer-boundary nodes of BOTH
#     layers, NUMDOF 3, ONOFF [1,1,1], VAL [0,0,0]) and DESIGN SURF THERMO
#     DIRICH CONDITIONS (E 1, NUMDOF 1, ONOFF [1], VAL [0]) -- or the task's
#     outer values
#   DESIGN POINT DIRICH CONDITIONS: one entry per INTERIOR interface node AND
#     per layer (two entries per 2-D node), NUMDOF 3, ONOFF [1,1,1],
#     VAL [ux, uy, 0.0] from partner_values(y), FUNCT [0,0,0],
#     TAG: monitor_reaction
#   DESIGN POINT THERMO DIRICH CONDITIONS: the same DNODE ids, NUMDOF 1,
#     ONOFF [1], VAL [T] from partner_values(y), FUNCT [0]
#   NODE COORDS, STRUCTURE ELEMENTS, DNODE-NODE TOPOLOGY (one DNODE per point
#     condition), DSURF-NODE TOPOLOGY (the outer nodes, both layers),
#     DVOL-NODE TOPOLOGY (every node)
#
# Run each deck with the binary at config `fourc_bin` and its libraries on
# `fourc_ld` as LD_LIBRARY_PATH, line-buffered, console captured:
#     stdbuf -oL -eL <bin> <deck> <out_prefix> > <deck>.log 2>&1
# with two different output prefixes, e.g. out_T and out_U. When a run exits
# non-zero DO NOT raise: fall through, the served check reads the log and the
# decks and names the cause. OASiS does not run the solver for you.
#
# WHAT YOUR SOLVE MUST LEAVE BEHIND (the recovery below uses these names):
#     nodes      the 2-D node list [(x, y), ...] indexed by 4C node id - 1
#                (the same ids in run T; run U's 3-D ids may differ, the
#                recovery maps its reactions by coordinate)
#     interior   the INTERIOR interface node ids (1-based, run-T numbering),
#                ordered along the interface; the two endpoints are dropped
#     TZ         the slab thickness used in run U
#     OUT_T      run T's output prefix (its VTU is <OUT_T>-vtk-files/scatra-*.vtu)
#     OUT_U      run U's output prefix (structure-*.vtu, thermo-*.vtu and
#                <OUT_U>-*_monitor_dbc.yaml)
#     DECK_U     the path of the run-U deck (its NODE COORDS map node gids to
#                coordinates for the reaction files)
# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end

# ── RECOVERY FROM 4C's OWN OUTPUTS (served) ────────────────────────────────
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
# The two runs solve the same discrete heat problem up to the load rule
# (nodal vs Gauss-point evaluation of the source, O(h^2)); a larger gap means
# one deck is not the problem it claims to be -- checked, not assumed.
_dT = float(np.max(np.abs(Ttsi - T2d)) / max(float(np.max(np.abs(T2d))), 1e-300))
if _dT > 0.05:
    raise SystemExit(f"the TSI deck's temperature differs from the scatra deck's by {_dT:.3f} relative: "
                     f"the two decks do not solve the same heat problem (check FUNCT3 = the heat source, "
                     f"the POINT THERMO DIRICH values and the outer SURF THERMO DIRICH of run U)")

# REACTIONS -> TRACTION. One yaml per monitored condition, with the node gid(s);
# the run-U deck's NODE COORDS map gids to coordinates.
_gid_xy = {}
for _ln in Path(DECK_U).read_text(errors="ignore").splitlines():
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
    # MEASURED: the yaml's node gids are ZERO-based while the deck's NODE ids
    # are one-based (gid 17 in the file is "NODE 18" in the deck)
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
