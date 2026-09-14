"""Participant A: 4C Thermo-Elastic Dirichlet Side (Steady, Plane Strain)
Subdomain A: [0, 0.8] x [0, 1]. Interface at x = 0.8.
Imports: T, ux, uy from partner B. Exports: qn, qx, qy.
Two 4C runs per iteration: 
  1. Scalar_Transport (T field + heat flux recovery)
  2. Thermo_Structure_Interaction (u field + traction recovery via reactions)
"""
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
INITTEMP = 0.0 # Reference temperature for zero strain

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
# Interface is x = 0.8 (Right side). Endpoints are corners.
interface_nodes_indices = []
interface_node_ids_deck = [] # 1-based
for idx, (x, y) in enumerate(nodes_2d):
    if abs(x - X1) < 1e-9:
        # Exclude corners if they are constrained by outer BCs? 
        # Prompt says "Outer Boundary (x=0, y=0, y=1)". Interface is x=0.8.
        # Corners (0.8, 0) and (0.8, 1) are part of y=0 and y=1 boundaries.
        # Usually corners belong to the intersection. We will treat them as interface 
        # but ensure outer BCs are applied there too if needed. 
        # For simplicity in partitioning, we apply partner values on ALL interface nodes.
        interface_nodes_indices.append(idx)
        interface_node_ids_deck.append(idx + 1)

# Interior interface nodes exclude endpoints (corners) for reaction calculation logic sometimes, 
# but for Dirichlet imposition we impose on all. The served contract uses `interior` list for flux export.
# Let's define `interior` as non-corner interface nodes.
interior_indices = []
corner_x = X1
corner_y_bottom = Y0
corner_y_top = Y1
for idx, (x, y) in enumerate(nodes_2d):
    if abs(x - corner_x) < 1e-9:
        if abs(y - corner_y_bottom) > 1e-5 and abs(y - corner_y_top) > 1e-5:
            interior_indices.append(idx)

# Global names required by the contract snippet
nodes = nodes_2d
interior = [idx + 1 for idx in interior_indices] # 1-based IDs
OUT_T = "run_t"
OUT_U = "run_u"
DECK_U = f"{OUT_U}.4C.yaml" # We will write this file name

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
        if va.ndim == 1: va = va.reshape(-1, 1)
        if va.shape[1] < 3: continue
        
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

def generate_mesh_sections(deck_name, include_z=True):
    lines = []
    # NODE COORDS
    # Layer 0 (z=0)
    for i, (x, y) in enumerate(nodes_2d):
        nid = i + 1
        lines.append(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} 0.0"')
    # Layer 1 (z=TZ)
    if include_z:
        for i, (x, y) in enumerate(nodes_2d):
            nid = N_NODES_2D + i + 1
            lines.append(f'  - "NODE {nid} COORD {x:.10f} {y:.10f} {TZ:.10f}"')
    
    # ELEMENTS
    # 2D Quads -> 3D Hexes
    elem_id = 1
    for j in range(NY):
        for i in range(NX):
            # 2D Quad nodes (CCW)
            n_bl = 1 + j*(NX+1) + i
            n_br = 1 + j*(NX+1) + i + 1
            n_tr = 1 + (j+1)*(NX+1) + i + 1
            n_tl = 1 + (j+1)*(NX+1) + i
            
            if include_z:
                # Hex nodes: bottom layer then top layer
                n_bl_1 = n_bl + N_NODES_2D
                n_br_1 = n_br + N_NODES_2D
                n_tr_1 = n_tr + N_NODES_2D
                n_tl_1 = n_tl + N_NODES_2D
                
                if deck_name == "tsi":
                    # SOLIDSCATRA HEX8
                    lines.append(f'  - "{elem_id} SOLIDSCATRA HEX8 {n_bl} {n_br} {n_tr} {n_tl} {n_bl_1} {n_br_1} {n_tr_1} {n_tl_1} MAT 1 KINEM linear TYPE Undefined"')
                else:
                    # TRANSP QUAD4 (Scalar Transport, 2D only effectively)
                    # Note: For Scalar Transport 2D, we use WALL or TRANSP?
                    # Documentation says: "P1 triangles... spelled ... TRANSP TRI3". "TRANSPORT ELEMENTS ... TRANSP QUAD4".
                    lines.append(f'  - "{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
                elem_id += 1
            else:
                # Only for Scalar Transport 2D
                lines.append(f'  - "{elem_id} TRANSP QUAD4 {n_bl} {n_br} {n_tr} {n_tl} MAT 1 TYPE Std"')
                elem_id += 1
    
    return "\n".join(lines)

def generate_topology(deck_name):
    """Generate DLINE, DSURF, DNODE topology sections."""
    topo_lines = []
    
    # Outer Boundaries (Lines)
    # Left (x=0): E:1, Bottom (y=0): E:2, Top (y=1): E:3, Right (Interface): E:4
    # We need separate IDs for conditions.
    
    eid_outer_left = 1
    eid_outer_bottom = 2
    eid_outer_top = 3
    eid_interface = 4
    
    # Map nodes to Design Entities
    dline_entries = []
    dsurf_entries = [] # For volume sources in 2D
    
    for i, (x, y) in enumerate(nodes_2d):
        nid = i + 1
        # Volume Source (All nodes)
        dsurf_entries.append(f'  - "NODE {nid} DSURFACE 1"')
        
        # Interface Line
        if abs(x - X1) < 1e-9:
            dline_entries.append(f'  - "NODE {nid} DLINE {eid_interface}"')
        # Outer Left
        elif abs(x - X0) < 1e-9:
            dline_entries.append(f'  - "NODE {nid} DLINE {eid_outer_left}"')
        # Outer Bottom
        elif abs(y - Y0) < 1e-9:
            dline_entries.append(f'  - "NODE {nid} DLINE {eid_outer_bottom}"')
        # Outer Top
        elif abs(y - Y1) < 1e-9:
            dline_entries.append(f'  - "NODE {nid} DLINE {eid_outer_top}"')
            
    # DNODE for Interface Point Conditions (TSI needs point-wise)
    dnode_entries = []
    dnode_id = 1
    for idx in interface_nodes_indices:
        nid = idx + 1
        dnode_entries.append(f'  - "NODE {nid} DNODE {dnode_id}"')
        dnode_id += 1
        
    # Slab Layer 1 Topology (TSI)
    # If TSI, duplicate for z=TZ layer
    slab_dlines = []
    slab_dsrf = []
    slab_dnods = []
    
    return dline_entries, dsurf_entries, dnode_entries, slab_dlines, slab_dsrf, slab_dnods

def build_deck_scalar_transport():
    yaml = []
    yaml.append('TITLE:\n  - "Side A Thermal Solve (Scalar Transport)"')
    yaml.append('PROBLEM SIZE:\n  ELEMENTS: {}\n  NODES: {}'.format(NX*NY, N_NODES_2D))
    yaml.append('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"')
    yaml.append('SCALAR TRANSPORT DYNAMIC:\n  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n  NUMSTEP: 1\n  TIMESTEP: 1.0\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n  CALCFLUX_BOUNDARY: "diffusive"')
    yaml.append('SOLVER 1:\n  SOLVER: "UMFPACK"')
    yaml.append('MATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: {}'.format(K_VAL))
    yaml.append('FUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{}"'.format(FUNCT_SRC_T))
    
    yaml.append('DESIGN SURF TRANSPORT NEUMANN CONDITIONS:')
    yaml.append('  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]')
    
    yaml.append('DSURF-NODE TOPOLOGY:')
    yaml.extend(generate_topology("scalar")[1])
    
    yaml.append('DESIGN LINE TRANSPORT DIRICH CONDITIONS:')
    # Outer: T=0
    for eid, name in [(1, "left"), (2, "bottom"), (3, "top")]:
        yaml.append(f'  - E: {eid}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]')
    
    # Interface: Point-wise imported T
    yaml.append('DESIGN POINT TRANSPORT DIRICH CONDITIONS:')
    dnode_cnt = 1
    for idx in interface_nodes_indices:
        T_val, _, _ = get_partner_values(nodes_2d[idx][1])
        yaml.append(f'  - E: {dnode_cnt}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{T_val:.15e}]\n    FUNCT: [0]')
        dnode_cnt += 1
        
    yaml.append('DNODE-NODE TOPOLOGY:')
    dnode_entries = generate_topology("scalar")[2]
    yaml.extend(dnode_entries)
    
    yaml.append('SCATRA FLUX CALC LINE CONDITIONS:')
    yaml.append('  - E: 4') # Interface ID
    
    yaml.append('IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  OUTPUT_DATA_FORMAT: ascii')
    
    yaml.append('NODE COORDS:')
    yaml.extend(generate_mesh_sections("scalar", include_z=False).split('\n'))
    
    yaml.append('TRANSPORT ELEMENTS:')
    yaml.extend(generate_mesh_sections("scalar", include_z=False).split('\n')[len(generate_mesh_sections("scalar", include_z=False).split('\n'))//2:]) # Skip coords
    # Fix: generate_mesh returns both. Split manually.
    # Re-implement simple split above.
    lines = generate_mesh_sections("scalar", include_z=False).split('\n')
    yaml[-1] = "" # Remove empty last line
    yaml.extend(lines[len([l for l in lines if 'NODE' in l]):]) # Approximate split
    
    return '\n'.join(yaml)

def build_deck_tsi():
    yaml = []
    yaml.append('TITLE:\n  - "Side A Structural Solve (Thermo Structure Interaction)"')
    yaml.append('PROBLEM SIZE:\n  DIM: 3\n  ELEMENTS: {}\n  NODES: {}'.format(NX*NY, N_NODES_2D*2))
    yaml.append('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"')
    
    yaml.append('STRUCTURAL DYNAMIC:\n  INT_STRATEGY: Standard\n  DYNAMICTYPE: "Statics"\n  TIMESTEP: 1.0\n  NUMSTEP: 1\n  MAXTIME: 1.0\n  TOLDISP: 1e-8\n  TOLRES: 1e-8\n  LINEAR_SOLVER: 2')
    yaml.append('THERMAL DYNAMIC:\n  INITIALFIELD: "field_by_function"\n  INITFUNCNO: 1\n  TIMESTEP: 1.0\n  MAXTIME: 1.0\n  TOLTEMP: 1e-8\n  TOLRES: 1e-8\n  LINEAR_SOLVER: 1')
    yaml.append('TSI DYNAMIC:\n  COUPALGO: "tsi_oneway"\n  MAXTIME: 1.0\n  TIMESTEP: 1.0\n  ITEMAX: 1')
    yaml.append('TSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"')
    
    yaml.append('SOLVER 1:\n  SOLVER: "UMFPACK"')
    yaml.append('SOLVER 2:\n  SOLVER: "UMFPACK"')
    
    yaml.append('MATERIALS:')
    yaml.append('  - MAT: 1')
    yaml.append('    MAT_Struct_ThermoStVenantK:')
    yaml.append(f'      YOUNGNUM: 1')
    yaml.append(f'      YOUNG: [{E_MOD:.6e}]')
    yaml.append(f'      NUE: {NU}')
    yaml.append(f'      DENS: 1.0')
    yaml.append(f'      THEXPANS: {ALPHA}')
    yaml.append(f'      INITTEMP: {INITTEMP}')
    yaml.append(f'      THERMOMAT: 2')
    yaml.append('  - MAT: 2')
    yaml.append('    MAT_Fourier:')
    yaml.append('      CAPA: 0.0')
    yaml.append(f'      CONDUCT:')
    yaml.append(f'        constant: [{K_VAL}]')
    
    yaml.append('CLONING MATERIAL MAP:')
    yaml.append('  - SRC_FIELD: "structure"')
    yaml.append('    SRC_MAT: 1')
    yaml.append('    TAR_FIELD: "thermo"')
    yaml.append('    TAR_MAT: 2')
    
    # Sources
    yaml.append('FUNCT1:')
    yaml.append(f'  - COMPONENT: 0\n    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_T}"')
    yaml.append('FUNCT2:')
    yaml.append(f'  - COMPONENT: 0\n    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_UX}"')
    yaml.append('FUNCT3:')
    yaml.append(f'  - COMPONENT: 0\n    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{FUNCT_SRC_UY}"')
    
    # Thermal Source
    yaml.append('DESIGN VOL THERMO NEUMANN CONDITIONS:')
    yaml.append('  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]')
    
    # Mechanical Source
    yaml.append('DESIGN VOL NEUMANN CONDITIONS:')
    yaml.append('  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [1.0, 1.0, 0.0]\n    FUNCT: [2, 3, 0]')
    
    # DVOL for Volume Source
    yaml.append('DVOL-NODE TOPOLOGY:')
    for i, (x, y) in enumerate(nodes_2d):
        yaml.append(f'  - "NODE {i+1} DVOL 1"')
        yaml.append(f'  - "NODE {N_NODES_2D+i+1} DVOL 1"')
        
    # Outer Dirichlets (Structure)
    # x=0, y=0, y=1 -> u=0
    yaml.append('DESIGN SURF DIRICH CONDITIONS:')
    # We map outer boundary nodes to DSURF entities 2, 3, 4 (Left, Bottom, Top)
    # Need to construct these mappings explicitly
    surf_entries = []
    eid_left, eid_bottom, eid_top = 2, 3, 4
    for i, (x, y) in enumerate(nodes_2d):
        nid0, nid1 = i+1, N_NODES_2D+i+1
        if abs(x-X0)<1e-9:
            surf_entries.append(f'  - "NODE {nid0} DSURFACE {eid_left}"')
            surf_entries.append(f'  - "NODE {nid1} DSURFACE {eid_left}"')
        elif abs(y-Y0)<1e-9:
            surf_entries.append(f'  - "NODE {nid0} DSURFACE {eid_bottom}"')
            surf_entries.append(f'  - "NODE {nid1} DSURFACE {eid_bottom}"')
        elif abs(y-Y1)<1e-9:
            surf_entries.append(f'  - "NODE {nid0} DSURFACE {eid_top}"')
            surf_entries.append(f'  - "NODE {nid1} DSURFACE {eid_top}"')
    yaml.extend(surf_entries)
    for eid in [eid_left, eid_bottom, eid_top]:
        yaml.append(f'  - E: {eid}\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0.0, 0.0, 0.0]\n    FUNCT: [0, 0, 0]')

    # Outer Thermal Dirichlets (T=0)
    yaml.append('DESIGN SURF THERMO DIRICH CONDITIONS:')
    # Reuse same topology entries? No, 4C requires consistent topology blocks.
    # We already wrote DSURFACE lines above.
    for eid in [eid_left, eid_bottom, eid_top]:
        yaml.append(f'  - E: {eid}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]')

    # Interface Point Dirichlets (Structure)
    yaml.append('DESIGN POINT DIRICH CONDITIONS:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        _, ux, uy = get_partner_values(nodes_2d[idx][1])
        nid0, nid1 = idx+1, N_NODES_2D+idx+1
        # Layer 0
        yaml.append(f'  - E: {dnode_id}\n    TAG: monitor_reaction\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [{ux:.15e}, {uy:.15e}, 0.0]\n    FUNCT: [0, 0, 0]')
        yaml.append(f'  - "NODE {nid0} DNODE {dnode_id}"')
        dnode_id += 1
        # Layer 1
        yaml.append(f'  - E: {dnode_id}\n    TAG: monitor_reaction\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [{ux:.15e}, {uy:.15e}, 0.0]\n    FUNCT: [0, 0, 0]')
        yaml.append(f'  - "NODE {nid1} DNODE {dnode_id}"')
        dnode_id += 1
        
    # Interface Thermal Dirichlets (T)
    yaml.append('DESIGN POINT THERMO DIRICH CONDITIONS:')
    dnode_id = 1
    for idx in interface_nodes_indices:
        T_val, _, _ = get_partner_values(nodes_2d[idx][1])
        nid0, nid1 = idx+1, N_NODES_2D+idx+1
        # Layer 0
        yaml.append(f'  - E: {dnode_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{T_val:.15e}]\n    FUNCT: [0]')
        yaml.append(f'  - "NODE {nid0} DNODE {dnode_id}"')
        dnode_id += 1
        # Layer 1
        yaml.append(f'  - E: {dnode_id}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [{T_val:.15e}]\n    FUNCT: [0]')
        yaml.append(f'  - "NODE {nid1} DNODE {dnode_id}"')
        dnode_id += 1
        
    # Pin u_z everywhere (Plane Strain)
    yaml.append('DESIGN VOL DIRICH CONDITIONS:')
    yaml.append('  - E: 1\n    NUMDOF: 3\n    ONOFF: [0, 0, 1]\n    VAL: [0.0, 0.0, 0.0]\n    FUNCT: [0, 0, 0]')
    # DVOL 1 was defined earlier.

    # IO
    yaml.append('IO/MONITOR STRUCTURE DBC:\n  INTERVAL_STEPS: 1\n  FILE_TYPE: yaml\n  WRITE_CONDITION_INFORMATION: true')
    yaml.append('IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1')
    yaml.append('IO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true')
    yaml.append('THERMAL DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_THERMO: true\n  TEMPERATURE: true')
    
    yaml.append('NODE COORDS:')
    yaml.extend(generate_mesh_sections("tsi", include_z=True).split('\n'))
    yaml.append('STRUCTURE ELEMENTS:')
    # Extract elements from previous function call result
    lines = generate_mesh_sections("tsi", include_z=True).split('\n')
    # Find where elements start
    elem_start = next(i for i, l in enumerate(lines) if 'HEX8' in l)
    yaml.extend(lines[elem_start:])
    
    return '\n'.join(yaml)

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
        except OSError: continue
        hit = False
        for i, ln in enumerate(lines):
            if "PROC 0 ERROR" in ln:
                said = [l.strip() for l in lines[i+1:i+12] if l.strip() and "MPI_ABORT" not in l]
                _why.append(f"4C said ({lg}): " + " | ".join(said[:8]))
                hit = True; break
        if not hit:
            for i, ln in enumerate(lines):
                if "*** Process received signal ***" in ln:
                    sig = next((l.split("Signal:", 1)[1].strip() for l in lines[i:i+4] if "Signal:" in l), "signal")
                    bef = [l.strip() for l in lines[max(0,i-12):i] if l.strip()]
                    _why.append(f"4C died on {sig} ({lg}); last print: " + " | ".join(bef[-2:]))
                    break
    for dk in sorted(glob.glob("*.4C.yaml")):
        txt = Path(dk).read_text(errors="ignore")
        secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", txt, re.M)
        dup = sorted({x for x in secs if secs.count(x) > 1})
        if dup: _why.append(f"{dk}: section(s) written twice: " + ", ".join(dup))
    if not _why: _why.append("no 4C error line in any *.log here -- run line-buffered")
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
    if not vs: return None
    def step(p): m=re.match(r".*-(\d+)-\d+\.vtu$", p); return int(m.group(1)) if m else -1
    return max(vs, key=step)

def _nodal(m, field, comps, zlayer=None):
    pts = np.asarray(m.points)
    val = np.asarray(m.point_data[field])
    if val.ndim == 1: val = val.reshape(-1, 1)
    rows = np.arange(len(pts)) if zlayer is None else np.where(np.abs(pts[:, 2] - zlayer) < 1e-9)[0]
    key = {}
    for r in rows:
        key.setdefault((round(float(pts[r, 0]), 9), round(float(pts[r, 1]), 9)), []).append(r)
    out = np.zeros((len(nodes), comps))
    for i, (x, y) in enumerate(nodes):
        rr = key.get((round(float(x), 9), round(float(y), 9)))
        if not rr: raise SystemExit(f"4C VTU field {field!r} has no node at {(x, y)}")
        out[i] = val[rr, :comps].mean(axis=0)
    return out

# Run T
Path(f"{OUT_T}.4C.yaml").write_text(build_deck_scalar_transport())
rc_t = run_4c(None, OUT_T, f"{OUT_T}.log")
vtu_T = _latest(f"{OUT_T}-vtk-files/scatra-*.vtu")
if vtu_T is None: raise SystemExit(why_4c_did_not_finish("run T"))
mT = meshio.read(vtu_T)
T2d = _nodal(mT, "phi_1", 1)[:, 0]
fbname = next((da for da in mT.point_data if "flux_boundary" in da), None)
if fbname is None: raise SystemExit("run T wrote no flux_boundary field")
FB = _nodal(mT, fbname, 2)
nrm = {"left": (-1.0, 0.0), "right": (1.0, 0.0), "bottom": (0.0, -1.0), "top": (0.0, 1.0)}[IFACE]
q_n = [float(FB[n-1, 0] * nrm[0] + FB[n-1, 1] * nrm[1]) for n in interior]

# Run U
Path(DECK_U).write_text(build_deck_tsi())
rc_u = run_4c(None, OUT_U, f"{OUT_U}.log")
vtu_U = _latest(f"{OUT_U}-vtk-files/structure-*.vtu")
vtu_UT = _latest(f"{OUT_U}-vtk-files/thermo-*.vtu")
if vtu_U is None or vtu_UT is None: raise SystemExit(why_4c_did_not_finish("run U"))
U2d = _nodal(meshio.read(vtu_U), "displacement", 2, zlayer=0.0)
Ttsi = _nodal(meshio.read(vtu_UT), "temperature", 1, zlayer=0.0)[:, 0]

# Reactions -> Traction
_gid_xy = {}
deck_txt = Path(DECK_U).read_text()
for ln in deck_txt.splitlines():
    m = re.match(r'\s*-\s*"?NODE\s+(\d+)\s+COORD\s+(\S+)\s+(\S+)\s+(\S+)"?', ln)
    if m: _gid_xy[int(m.group(1))] = (round(float(m.group(2)), 9), round(float(m.group(3)), 9))
_F = {}
for yf in glob.glob(f"{OUT_U}-*_monitor_dbc.yaml"):
    txt = Path(yf).read_text(errors="ignore")
    gids = [int(g) for g in re.findall(r"^\s*-\s*(\d+)\s*$", txt.split("dbc monitor condition data", 1)[0], re.M)]
    fs = re.findall(r"^\s*f:\s*\n\s*-\s*(\S+)\s*\n\s*-\s*(\S+)", txt, re.M)
    if not (gids and fs): continue
    fx, fy = float(fs[-1][0]), float(fs[-1][1])
    xy = _gid_xy.get(gids[0] + 1)
    if xy is None: continue
    _acc = _F.setdefault(xy, [0.0, 0.0])
    _acc[0] += fx; _acc[1] += fy
if not _F: raise SystemExit("run U wrote no *_monitor_dbc.yaml")
_ax = 1 if IFACE in ("left", "right") else 0
_coord = [float(nodes[n-1][_ax]) for n in interior]
_h = float(np.median(np.diff(sorted(_coord)))) if len(_coord) > 1 else (Y1 - Y0)
_share = _h * float(TZ)
q_t = []
for n in interior:
    xy = (round(float(nodes[n-1][0]), 9), round(float(nodes[n-1][1]), 9))
    if xy not in _F: raise SystemExit(f"no reaction file for node at {xy}")
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
