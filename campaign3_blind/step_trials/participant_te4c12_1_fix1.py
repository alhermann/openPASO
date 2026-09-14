"""Participant A: 4C Thermo-Elastic Dirichlet Side (Steady, Plane Strain)
Subdomain A: [0, 0.8] x [0, 1]. Interface at x = 0.8.
Imports: T, ux, uy from partner B. Exports: qn, qx, qy.
Two 4C runs per iteration: 
  1. Scalar_Transport (T field + heat flux recovery)
  2. Thermo_Structure_Interaction (u field + traction recovery via reactions)
"""
import atexit
import os
import sys
import json
import re
import subprocess
import glob
from pathlib import Path

import numpy as np
import meshio

# ==============================================================================
# CONFIGURATION & MESH GENERATION
# ==============================================================================

def load_config():
    cfg = json.loads(Path("config.json").read_text())
    # Override with environment if present (OASiS multi-level pattern)
    env_cfg = os.environ.get("OASIS_CONFIG_JSON")
    if env_cfg:
        cfg.update(json.loads(env_cfg))
    return cfg

CFG = load_config()
NX, NY = int(CFG["nx"]), int(CFG["ny"])
X0, X1, Y0, Y1 = float(CFG["x0"]), float(CFG["x1"]), float(CFG["y0"]), float(CFG["y1"])
K_VAL = float(CFG["k"])
LAM, MU, BETA = float(CFG["lam"]), float(CFG["mu"]), float(CFG["beta"])
IFACE = CFG.get("iface", "right")
SIDE = CFG.get("side", "dirichlet")
SRC_T_STR = str(CFG.get("source_T", "0.0"))
SRC_UX_STR = str(CFG.get("source_ux", "0.0"))
SRC_UY_STR = str(CFG.get("source_uy", "0.0"))
FOURC_BIN = CFG.get("fourc_bin", "/home/alexander/4C/build/4C")
FOURC_LD = CFG.get("fourc_ld", "/opt/4C-dependencies/lib")

# Material conversions for 4C
E_MOD = MU * (3.0 * LAM + 2.0 * MU) / (LAM + MU)
NU = LAM / (2.0 * (LAM + MU))
ALPHA = BETA / (3.0 * LAM + 2.0 * MU)
INITTEMP = 0.0  # Reference temperature for zero strain

# Slab thickness for plane strain approximation
TZ = 0.1

# Build 2D node layout
nodes_2d = []
for j in range(NY + 1):
    for i in range(NX + 1):
        x = X0 + i * (X1 - X0) / NX
        y = Y0 + j * (Y1 - Y0) / NY
        nodes_2d.append((x, y))

N_NODES_2D = len(nodes_2d)

# Identify interface nodes (interior of the interface line)
# Interface is x = 0.8 (Right side)
interface_nodes_indices = []
for idx, (x, y) in enumerate(nodes_2d):
    if abs(x - X1) < 1e-9:
        interface_nodes_indices.append(idx)

# Interior interface nodes exclude endpoints (corners)
corner_x = X1
corner_y_bottom = Y0
corner_y_top = Y1
interior_indices = []
for idx, (x, y) in enumerate(nodes_2d):
    if abs(x - corner_x) < 1e-9:
        if abs(y - corner_y_bottom) > 1e-5 and abs(y - corner_y_top) > 1e-5:
            interior_indices.append(idx)

# Global names required by the contract snippet
nodes = nodes_2d
interior = [idx + 1 for idx in interior_indices]  # 1-based IDs
OUT_T = "run_t"
OUT_U = "run_u"
DECK_U = f"{OUT_U}.4C.yaml"

# Convert Python source strings to 4C syntax
def py_to_4c_src(s):
    s = s.replace("**", "^").replace("PI", "pi").replace("Pi", "pi")
    return s

FUNCT_SRC_T = py_to_4c_src(SRC_T_STR)
FUNCT_SRC_UX = py_to_4c_src(SRC_UX_STR)
FUNCT_SRC_UY = py_to_4c_src(SRC_UY_STR)

# ==============================================================================
# IMPORT HANDSHAKE
# ==============================================================================

imp = {}
if Path("imports.json").is_file():
    try:
        imp = json.loads(Path("imports.json").read_text() or "{}")
    except Exception:
        pass

def get_partner_values(coord):
    """Interpolate partner's (T, ux, uy) at a specific coordinate along the interface."""
    if not imp:
        return (0.0, 0.0, 0.0)
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

        # Axis for interpolation (y for vertical interface)
        ax = 1 if IFACE in ("left", "right") else 0
        s = np.asarray([c[ax] for c in co], float)
        o = np.argsort(s)
        t = min(max(float(coord), s[o][0]), s[o][-1])
        return tuple(float(np.interp(t, s[o], va[o, c])) for c in range(3))
    return (0.0, 0.0, 0.0)

# ==============================================================================
# DECK GENERATION
# ==============================================================================

def build_deck_scalar_transport():
    lines = []
    lines.append('TITLE:')
    lines.append('  - "Side A Thermal Solve (Scalar Transport)"')
    lines.append('PROBLEM SIZE:')
    lines.append(f'  ELEMENTS: {NX*NY}')
    lines.append(f'  NODES: {N_NODES_2D}')
    lines.append('PROBLEM TYPE:')
    lines.append('  PROBLEMTYPE: "Scalar_Transport"')
    lines.append('SCALAR TRANSPORT DYNAMIC:')
    lines.append('  TIMEINTEGR: "Stationary"')
    lines.append('  SOLVERTYPE: "linear_full"')
    lines.append('  NUMSTEP: 1')
    lines.append('  TIMESTEP: 1.0')
    lines.append('  MAXTIME: 1.0')
    lines.append('  LINEAR_SOLVER: 1')
    lines.append('  CALCFLUX_BOUNDARY: "diffusive"')
    lines.append('SOLVER 1:')
    lines.append('  SOLVER: "UMFPACK"')
    lines.append('MATERIALS:')
    lines.append('  - MAT: 1')
    lines.append('    MAT_scatra:')
    lines.append(f'      DIFFUSIVITY: {K_VAL}')
    lines.append('FUNCT1:')
    lines.append(f'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_T}"')

    # Volume source
    lines.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
    lines.append('  - E: 1')
    lines.append('    NUMDOF: 1')
    lines.append('    ONOFF: [1]')
    lines.append('    VAL: [1.0]')
    lines.append('    FUNCT: [1]')

    # Outer Dirichlets (T=0)
    lines.append('DESIGN LINE TRANSPORT DIRICH CONDITIONS:')
    # Left (x=0): E:1, Bottom (y=0): E:2, Top (y=1): E:3
    for eid in [1, 2, 3]:
        lines.append(f'  - E: {eid}')
        lines.append('    NUMDOF: 1')
        lines.append('    ONOFF: [1]')
        lines.append('    VAL: [0.0]')
        lines.append('    FUNCT: [0]')

    # Interface Point Dirichlets
    lines.append('DESIGN POINT TRANSPORT DIRICH CONDITIONS:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        T_val, _, _ = get_partner_values(nodes_2d[idx][1])
        lines.append(f'  - E: {dnode_id}')
        lines.append('    NUMDOF: 1')
        lines.append('    ONOFF: [1]')
        lines.append(f'    VAL: [{T_val:.15e}]')
        lines.append('    FUNCT: [0]')
        dnode_id += 1

    # Flux calc line
    lines.append('SCATRA FLUX CALC LINE CONDITIONS:')
    lines.append('  - E: 4')  # Interface ID

    # IO
    lines.append('IO/RUNTIME VTK OUTPUT:')
    lines.append('  INTERVAL_STEPS: 1')
    lines.append('  OUTPUT_DATA_FORMAT: ascii')

    # NODE COORDS
    lines.append('NODE COORDS:')
    for i, (x, y) in enumerate(nodes_2d):
        nid = i + 1
        lines.append(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} 0.0"')

    # TRANSPORT ELEMENTS
    lines.append('TRANSPORT ELEMENTS:')
    elem_id = 1
    for j in range(NY):
        for i in range(NX):
            n_bl = 1 + j * (NX + 1) + i
            n_br = 1 + j * (NX + 1) + i + 1
            n_tr = 1 + (j + 1) * (NX + 1) + i + 1
            n_tl = 1 + (j + 1) * (NX + 1) + i
            lines.append(f'  - "{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
            elem_id += 1

    # DSURF-NODE TOPOLOGY (Volume source)
    lines.append('DSURF-NODE TOPOLOGY:')
    for i in range(N_NODES_2D):
        lines.append(f'  - "NODE {i+1} DSURFACE 1"')

    # DLINE-NODE TOPOLOGY (Outer boundaries)
    lines.append('DLINE-NODE TOPOLOGY:')
    for i, (x, y) in enumerate(nodes_2d):
        nid = i + 1
        if abs(x - X0) < 1e-9:
            lines.append(f'  - "NODE {nid} DLINE 1"')
        elif abs(y - Y0) < 1e-9:
            lines.append(f'  - "NODE {nid} DLINE 2"')
        elif abs(y - Y1) < 1e-9:
            lines.append(f'  - "NODE {nid} DLINE 3"')
        elif abs(x - X1) < 1e-9:
            lines.append(f'  - "NODE {nid} DLINE 4"')

    # DNODE-NODE TOPOLOGY (Interface points)
    lines.append('DNODE-NODE TOPOLOGY:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        nid = idx + 1
        lines.append(f'  - "NODE {nid} DNODE {dnode_id}"')
        dnode_id += 1

    return '\n'.join(lines)


def build_deck_tsi():
    lines = []
    lines.append('TITLE:')
    lines.append('  - "Side A Structural Solve (Thermo Structure Interaction)"')
    lines.append('PROBLEM SIZE:')
    lines.append('  DIM: 3')
    lines.append(f'  ELEMENTS: {NX*NY}')
    lines.append(f'  NODES: {N_NODES_2D*2}')
    lines.append('PROBLEM TYPE:')
    lines.append('  PROBLEMTYPE: "Thermo_Structure_Interaction"')

    lines.append('STRUCTURAL DYNAMIC:')
    lines.append('  INT_STRATEGY: Standard')
    lines.append('  DYNAMICTYPE: "Statics"')
    lines.append('  TIMESTEP: 1.0')
    lines.append('  NUMSTEP: 1')
    lines.append('  MAXTIME: 1.0')
    lines.append('  TOLDISP: 1e-8')
    lines.append('  TOLRES: 1e-8')
    lines.append('  LINEAR_SOLVER: 2')

    lines.append('THERMAL DYNAMIC:')
    lines.append('  INITIALFIELD: "field_by_function"')
    lines.append('  INITFUNCNO: 1')
    lines.append('  TIMESTEP: 1.0')
    lines.append('  MAXTIME: 1.0')
    lines.append('  TOLTEMP: 1e-8')
    lines.append('  TOLRES: 1e-8')
    lines.append('  LINEAR_SOLVER: 1')

    lines.append('TSI DYNAMIC:')
    lines.append('  COUPALGO: "tsi_oneway"')
    lines.append('  MAXTIME: 1.0')
    lines.append('  TIMESTEP: 1.0')
    lines.append('  ITEMAX: 1')

    lines.append('TSI DYNAMIC/PARTITIONED:')
    lines.append('  COUPVARIABLE: "Temperature"')

    lines.append('SOLVER 1:')
    lines.append('  SOLVER: "UMFPACK"')
    lines.append('SOLVER 2:')
    lines.append('  SOLVER: "UMFPACK"')

    lines.append('MATERIALS:')
    lines.append('  - MAT: 1')
    lines.append('    MAT_Struct_ThermoStVenantK:')
    lines.append('      YOUNGNUM: 1')
    lines.append(f'      YOUNG: [{E_MOD:.6e}]')
    lines.append(f'      NUE: {NU}')
    lines.append('      DENS: 1.0')
    lines.append(f'      THEXPANS: {ALPHA}')
    lines.append(f'      INITTEMP: {INITTEMP}')
    lines.append('      THERMOMAT: 2')
    lines.append('  - MAT: 2')
    lines.append('    MAT_Fourier:')
    lines.append('      CAPA: 0.0')
    lines.append('      CONDUCT:')
    lines.append(f'        constant: [{K_VAL}]')

    lines.append('CLONING MATERIAL MAP:')
    lines.append('  - SRC_FIELD: "structure"')
    lines.append('    SRC_MAT: 1')
    lines.append('    TAR_FIELD: "thermo"')
    lines.append('    TAR_MAT: 2')

    # Sources
    lines.append('FUNCT1:')
    lines.append(f'  - COMPONENT: 0')
    lines.append(f'    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_T}"')
    lines.append('FUNCT2:')
    lines.append(f'  - COMPONENT: 0')
    lines.append(f'    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_UX}"')
    lines.append('FUNCT3:')
    lines.append(f'  - COMPONENT: 0')
    lines.append(f'    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_UY}"')

    # Thermal source
    lines.append('DESIGN VOL THERMO NEUMANN CONDITIONS:')
    lines.append('  - E: 1')
    lines.append('    NUMDOF: 1')
    lines.append('    ONOFF: [1]')
    lines.append('    VAL: [1.0]')
    lines.append('    FUNCT: [1]')

    # Mechanical source
    lines.append('DESIGN VOL NEUMANN CONDITIONS:')
    lines.append('  - E: 1')
    lines.append('    NUMDOF: 3')
    lines.append('    ONOFF: [1, 1, 1]')
    lines.append('    VAL: [1.0, 1.0, 0.0]')
    lines.append('    FUNCT: [2, 3, 0]')

    # Outer Dirichlets (Structure)
    lines.append('DESIGN SURF DIRICH CONDITIONS:')
    # Left (x=0): E:2, Bottom (y=0): E:3, Top (y=1): E:4
    for eid in [2, 3, 4]:
        lines.append(f'  - E: {eid}')
        lines.append('    NUMDOF: 3')
        lines.append('    ONOFF: [1, 1, 1]')
        lines.append('    VAL: [0.0, 0.0, 0.0]')
        lines.append('    FUNCT: [0, 0, 0]')

    # Outer Thermal Dirichlets (T=0)
    lines.append('DESIGN SURF THERMO DIRICH CONDITIONS:')
    for eid in [2, 3, 4]:
        lines.append(f'  - E: {eid}')
        lines.append('    NUMDOF: 1')
        lines.append('    ONOFF: [1]')
        lines.append('    VAL: [0.0]')
        lines.append('    FUNCT: [0]')

    # Interface Point Dirichlets (Structure)
    lines.append('DESIGN POINT DIRICH CONDITIONS:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        _, ux, uy = get_partner_values(nodes_2d[idx][1])
        nid0, nid1 = idx + 1, N_NODES_2D + idx + 1
        # Layer 0
        lines.append(f'  - E: {dnode_id}')
        lines.append('    TAG: monitor_reaction')
        lines.append('    NUMDOF: 3')
        lines.append('    ONOFF: [1, 1, 1]')
        lines.append(f'    VAL: [{ux:.15e}, {uy:.15e}, 0.0]')
        lines.append('    FUNCT: [0, 0, 0]')
        lines.append(f'  - "NODE {nid0} DNODE {dnode_id}"')
        dnode_id += 1
        # Layer 1
        lines.append(f'  - E: {dnode_id}')
        lines.append('    TAG: monitor_reaction')
        lines.append('    NUMDOF: 3')
        lines.append('    ONOFF: [1, 1, 1]')
        lines.append(f'    VAL: [{ux:.15e}, {uy:.15e}, 0.0]')
        lines.append('    FUNCT: [0, 0, 0]')
        lines.append(f'  - "NODE {nid1} DNODE {dnode_id}"')
        dnode_id += 1

    # Interface Thermal Dirichlets (T)
    lines.append('DESIGN POINT THERMO DIRICH CONDITIONS:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        T_val, _, _ = get_partner_values(nodes_2d[idx][1])
        nid0, nid1 = idx + 1, N_NODES_2D + idx + 1
        # Layer 0
        lines.append(f'  - E: {dnode_id}')
        lines.append('    NUMDOF: 1')
        lines.append('    ONOFF: [1]')
        lines.append(f'    VAL: [{T_val:.15e}]')
        lines.append('    FUNCT: [0]')
        lines.append(f'  - "NODE {nid0} DNODE {dnode_id}"')
        dnode_id += 1
        # Layer 1
        lines.append(f'  - E: {dnode_id}')
        lines.append('    NUMDOF: 1')
        lines.append('    ONOFF: [1]')
        lines.append(f'    VAL: [{T_val:.15e}]')
        lines.append('    FUNCT: [0]')
        lines.append(f'  - "NODE {nid1} DNODE {dnode_id}"')
        dnode_id += 1

    # Pin u_z everywhere (Plane Strain)
    lines.append('DESIGN VOL DIRICH CONDITIONS:')
    lines.append('  - E: 1')
    lines.append('    NUMDOF: 3')
    lines.append('    ONOFF: [0, 0, 1]')
    lines.append('    VAL: [0.0, 0.0, 0.0]')
    lines.append('    FUNCT: [0, 0, 0]')

    # DVOL-NODE TOPOLOGY
    lines.append('DVOL-NODE TOPOLOGY:')
    for i in range(N_NODES_2D):
        lines.append(f'  - "NODE {i+1} DVOL 1"')
        lines.append(f'  - "NODE {N_NODES_2D+i+1} DVOL 1"')

    # DSURF-NODE TOPOLOGY (Outer boundaries)
    lines.append('DSURF-NODE TOPOLOGY:')
    for i, (x, y) in enumerate(nodes_2d):
        nid0, nid1 = i + 1, N_NODES_2D + i + 1
        if abs(x - X0) < 1e-9:
            lines.append(f'  - "NODE {nid0} DSURFACE 2"')
            lines.append(f'  - "NODE {nid1} DSURFACE 2"')
        elif abs(y - Y0) < 1e-9:
            lines.append(f'  - "NODE {nid0} DSURFACE 3"')
            lines.append(f'  - "NODE {nid1} DSURFACE 3"')
        elif abs(y - Y1) < 1e-9:
            lines.append(f'  - "NODE {nid0} DSURFACE 4"')
            lines.append(f'  - "NODE {nid1} DSURFACE 4"')

    # IO
    lines.append('IO/MONITOR STRUCTURE DBC:')
    lines.append('  INTERVAL_STEPS: 1')
    lines.append('  FILE_TYPE: yaml')
    lines.append('  WRITE_CONDITION_INFORMATION: true')
    lines.append('IO/RUNTIME VTK OUTPUT:')
    lines.append('  INTERVAL_STEPS: 1')
    lines.append('IO/RUNTIME VTK OUTPUT/STRUCTURE:')
    lines.append('  OUTPUT_STRUCTURE: true')
    lines.append('  DISPLACEMENT: true')
    lines.append('THERMAL DYNAMIC/RUNTIME VTK OUTPUT:')
    lines.append('  OUTPUT_THERMO: true')
    lines.append('  TEMPERATURE: true')

    # NODE COORDS
    lines.append('NODE COORDS:')
    for i, (x, y) in enumerate(nodes_2d):
        nid = i + 1
        lines.append(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} 0.0"')
    for i, (x, y) in enumerate(nodes_2d):
        nid = N_NODES_2D + i + 1
        lines.append(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} {TZ:.10f}"')

    # STRUCTURE ELEMENTS
    lines.append('STRUCTURE ELEMENTS:')
    elem_id = 1
    for j in range(NY):
        for i in range(NX):
            n_bl = 1 + j * (NX + 1) + i
            n_br = 1 + j * (NX + 1) + i + 1
            n_tr = 1 + (j + 1) * (NX + 1) + i + 1
            n_tl = 1 + (j + 1) * (NX + 1) + i
            n_bl_1 = n_bl + N_NODES_2D
            n_br_1 = n_br + N_NODES_2D
            n_tr_1 = n_tr + N_NODES_2D
            n_tl_1 = n_tl + N_NODES_2D
            lines.append(f'  - "{elem_id} SOLIDSCATRA HEX8 {n_bl} {n_br} {n_tr} {n_tl} {n_bl_1} {n_br_1} {n_tr_1} {n_tl_1} MAT 1 KINEM linear TYPE Undefined"')
            elem_id += 1

    # DNODE-NODE TOPOLOGY (Interface points - already embedded in conditions)
    # Actually we need a separate section for DNODE
    lines.append('DNODE-NODE TOPOLOGY:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        nid0, nid1 = idx + 1, N_NODES_2D + idx + 1
        lines.append(f'  - "NODE {nid0} DNODE {dnode_id}"')
        dnode_id += 1
        lines.append(f'  - "NODE {nid1} DNODE {dnode_id}"')
        dnode_id += 1

    return '\n'.join(lines)


# ==============================================================================
# RUN & DIAGNOSIS
# ==============================================================================

def run_4c(deck_content, prefix, log_name):
    deck_file = f"{prefix}.4C.yaml"
    Path(deck_file).write_text(deck_content)

    cmd = [FOURC_BIN, deck_file, prefix]
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = FOURC_LD

    # Line buffered capture
    full_cmd = f"stdbuf -oL -eL {' '.join(cmd)}"
    try:
        proc = subprocess.run(full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        log_path = Path(log_name)
        log_path.write_text(proc.stdout + proc.stderr)
        return proc.returncode
    except Exception as e:
        Path(log_name).write_text(str(e))
        return 1


def why_4c_did_not_finish(tag=""):
    _why = []
    for lg in sorted(glob.glob("*.log")) + sorted(glob.glob("*.txt")):
        try:
            lines = open(lg, errors="ignore").read().splitlines()
        except OSError:
            continue
        hit = False
        for i, ln in enumerate(lines):
            if "PROC 0 ERROR" in ln:
                said = [l.strip() for l in lines[i+1:i+12] if l.strip() and "MPI_ABORT" not in l]
                _why.append(f"4C said ({lg}): " + " | ".join(said[:8]))
                hit = True
                break
        if not hit:
            for i, ln in enumerate(lines):
                if "*** Process received signal ***" in ln:
                    sig = next((l.split("Signal:", 1)[1].strip() for l in lines[i:i+4] if "Signal:" in l), "signal")
                    bef = [l.strip() for l in lines[max(0, i-12):i] if l.strip()]
                    _why.append(f"4C died on {sig} ({lg}); last print: " + " | ".join(bef[-2:]))
                    break
    for dk in sorted(glob.glob("*.4C.yaml")):
        txt = Path(dk).read_text(errors="ignore")
        secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", txt, re.M)
        dup = sorted({x for x in secs if secs.count(x) > 1})
        if dup:
            _why.append(f"{dk}: section(s) written twice: " + ", ".join(dup))
    if not _why:
        _why.append("no 4C error line in any *.log here -- run line-buffered")
    return f"4C DID NOT FINISH{(' (' + tag + ')') if tag else ''} -- " + "; ".join(_why)


@atexit.register
def _diagnose_at_exit():
    if not Path("exports.json").is_file():
        print(why_4c_did_not_finish(), file=sys.stderr, flush=True)


# ==============================================================================
# RECOVERY
# ==============================================================================

def _latest(pattern):
    vs = sorted(glob.glob(pattern))
    if not vs:
        return None

    def step(p):
        m = re.match(r".*-(\d+)-\d+\.vtu$", p)
        return int(m.group(1)) if m else -1
    return max(vs, key=step)


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
            raise SystemExit(f"4C VTU field {field!r} has no node at {(x, y)}")
        out[i] = val[rr, :comps].mean(axis=0)
    return out


# Run T
Path(f"{OUT_T}.4C.yaml").write_text(build_deck_scalar_transport())
rc_t = run_4c(None, OUT_T, f"{OUT_T}.log")
vtu_T = _latest(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None:
    raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None:
    raise SystemExit("run T wrote no flux_boundary field")
FB = _nodal(mT, fbname, 2)
nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IFACE]
q_n = [float(FB[n-1, 0] * nrm[0] + FB[n-1, 1] * nrm[1]) for n in interior]

# Run U
Path(DECK_U).write_text(build_deck_tsi())
rc_u = run_4c(None, OUT_U, f"{OUT_U}.log")
vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None:
    raise SystemExit(why_4c_did_not_finish("run U"))
U2d = _nodal(meshio.read(vtu_U), "displacement", 2, zlayer=0.0)
Ttsi = _nodal(meshio.read(vtu_UT), "temperature", 1, zlayer=0.0)[:, 0]

# Reactions -> Traction
_gid_xy = {}
deck_txt = Path(DECK_U).read_text()
for ln in deck_txt.splitlines():
    m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', ln)
    if m:
        _gid_xy[int(m.group(1))] = (round(float(m.group(2)), 9), round(float(m.group(3)), 9))

_F = {}
for yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    txt = Path(yf).read_text(errors="ignore")
    gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", txt.split("dbc monitor condition data", 1)[0], re.M)]
    fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt, re.M)
    if not (gids and fs):
        continue
    fx, fy = float(fs[-1][0]), float(fs[-1][1])
    xy = _gid_xy.get(gids[0] + 1)
    if xy is None:
        continue
    _acc = _F.setdefault(xy, [0.0, 0.0])
    _acc[0] += fx
    _acc[1] += fy

if not _F:
    raise SystemExit("run U wrote no *_monitor_dbc.yaml")

_ax = 1 if IFACE in ("left", "right") else 0
_coord = [float(nodes[n-1][_ax]) for n in interior]
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else (Y1 - Y0)
_share = _h * float(TZ)

q_t = []
for n in interior:
    xy = (round(float(nodes[n-1][0]), 9), round(float(nodes[n-1][1]), 9))
    if xy not in _F:
        raise SystemExit(f"no reaction file for node at {xy}")
    q_t.append([_F[xy][0] / _share, _F[xy][1] / _share])

co = [[float(nodes[n-1][0]), float(nodes[n-1][1])] for n in interior]
Q = [[qn, qt[0], qt[1]] for qn, qt in zip(q_n, q_t)]

# Export
json.dump({"field_name": "thermoelastic", "coordinates": co, "values": [],
           "normal_fluxes": Q, "n_points": len(co)}, open("exports.json", "w"))

# Level persistence
_LVL = CFG.get("level", "X")
with open(f"field_level{_LVL}.csv", "w") as f:
    f.write("x,y,T,ux,uy\n")
    for (px, py), t, (ux, uy) in zip(nodes, T2d, U2d):
        f.write(f"{px:.11e},{py:.11e},{t:.11e},{ux:.11e},{uy:.11e}\n")

with open(f"interface_level{_LVL}.csv", "w") as f:
    f.write("x,y,T,ux,uy,qn,tx,ty\n")
    for (px, py), n, (qn, qx, qy) in zip(co, interior, Q):
        f.write(f"{px:.11e},{py:.11e},{T2d[n-1]:.11e},{U2d[n-1,0]:.11e},{U2d[n-1,1]:.11e},{qn:.11e},{-qx:.11e},{-qy:.11e}\n")

print(f"NDOF = {3 * len(nodes)}")
print(f"4C thermo-elastic Dirichlet participant: NDOF = {3 * len(nodes)} T=[{T2d.min():.6g},{T2d.max():.6g}] u=[{U2d.min():.6g},{U2d.max():.6g}]")
