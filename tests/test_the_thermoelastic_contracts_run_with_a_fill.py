"""The served THERMO-ELASTIC contracts run, with fills openPASO never serves.

Option B elides the solve; what openPASO serves for a temperature-plus-
displacement exchange (three-component handshake, sign convention, 4C's
two-run recovery from its boundary-flux VTU and reaction monitor, FEniCSx's
reaction recovery of both fields, self-checks, exports) must be executable
and RIGHT. Each test fills the contract's holes with a minimal solve kept
here, runs it on a manufactured thermo-elastic solution, and checks the
exported flux/traction (Dirichlet role) or trace (Neumann role) against the
exact ones. The fills are test fixtures: they never reach an agent.

Manufactured solution (k = 2, lambda = 500, mu = 300, beta = 1) on
A = (0, 0.8) x (0, 1), interface x = 0.8:
    T = x sin(pi y),  ux = 0.02 x sin(pi y),  uy = 0.01 x^2 sin(pi y)
and its sources from the strong form. Measured on the shipped files
(2026-09-11): 4C flux/traction 3.7e-3 / 1.45e-2 / 1.43e-2 at h = 1/10,
second order to h = 1/40; FEniCSx Neumann trace 1.8e-2 / 1.2e-2 / 8e-3.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PART = ROOT / "data" / "coupling_participants"
BEGIN = "# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ begin"
END = "# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─ end"
FOURC = Path("/home/user/4C/build/4C")
FENICS_PY = Path("/home/user/miniconda3/envs/fenics/bin/python")
LX, LY = 0.8, 1.0
KV, LAM, MU, BETA = 2.0, 500.0, 300.0, 1.0


def _manufactured():
    sp = pytest.importorskip("sympy")
    x, y = sp.symbols("x y")
    T = x * sp.sin(sp.pi * y)
    ux = sp.Rational(2, 100) * x * sp.sin(sp.pi * y)
    uy = sp.Rational(1, 100) * x**2 * sp.sin(sp.pi * y)
    exx, eyy = sp.diff(ux, x), sp.diff(uy, y)
    exy = (sp.diff(ux, y) + sp.diff(uy, x)) / 2
    sxx = 2 * MU * exx + LAM * (exx + eyy) - BETA * T
    syy = 2 * MU * eyy + LAM * (exx + eyy) - BETA * T
    sxy = 2 * MU * exy
    fT = sp.simplify(-KV * (sp.diff(T, x, 2) + sp.diff(T, y, 2)))
    fx = sp.simplify(-(sp.diff(sxx, x) + sp.diff(sxy, y)))
    fy = sp.simplify(-(sp.diff(sxy, x) + sp.diff(syy, y)))
    lam = lambda e: sp.lambdify((x, y), e, "numpy")
    return dict(T=lam(T), ux=lam(ux), uy=lam(uy), qn=lam(-KV * sp.diff(T, x)), qx=lam(-sxx), qy=lam(-sxy),
                fT=str(fT), fx=str(fx), fy=str(fy),
                npy=lambda e: str(e).replace("sin", "np.sin").replace("cos", "np.cos").replace("pi", "np.pi"),
                sym=(fT, fx, fy))


def _imports(m, n=20):
    ys = [i / n for i in range(n + 1)]
    return {"P": {"field_name": "thermoelastic", "n_points": len(ys), "coordinates": [[LX, yy] for yy in ys],
                  "values": [[float(m["T"](LX, yy)), float(m["ux"](LX, yy)), float(m["uy"](LX, yy))] for yy in ys],
                  # the partner's export in the flux convention = this side's natural datum for the Neumann role
                  "normal_fluxes": [[float(-m["qn"](LX, yy)), float(-m["qx"](LX, yy)), float(-m["qy"](LX, yy))] for yy in ys]}}


FILL_4C_H1 = '# ---- MY FILL, the hole (validation only, never served): the 2-D node layout ----\nhx, hy = (X1 - X0) / NX, (Y1 - Y0) / NY\nnodes = [(X0 + i * hx, Y0 + j * hy) for j in range(NY + 1) for i in range(NX + 1)]\ndef gid(i, j): return 1 + i + (NX + 1) * j\nquads = [(gid(i, j), gid(i + 1, j), gid(i + 1, j + 1), gid(i, j + 1)) for j in range(NY) for i in range(NX)]\n'
FILL_4C_H2 = '# ---- MY FILL, hole 2 (validation only, never served): the two deck headers and runs ----\nHEADER_T = f\'\'\'TITLE:\n  - "run T"\nPROBLEM SIZE:\n  DIM: 2\nPROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n  VELOCITYFIELD: "zero"\n  TIMESTEP: 1.0\n  NUMSTEP: 1\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n  CALCFLUX_BOUNDARY: "diffusive"\nIO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  OUTPUT_DATA_FORMAT: "ascii"\nSOLVER 1:\n  SOLVER: "UMFPACK"\n  NAME: "direct"\nMATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: {KV}\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\nDESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\nDESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\nSCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\n\'\'\'\nDECK_T = "deck_T.4C.yaml"; OUT_T = "out_T"\nPath(DECK_T).write_text(HEADER_T + deck_T_tables())\nBIN = CFG.get("fourc_bin", os.environ.get("FOURC_BIN", "4C")); LD = CFG.get("fourc_ld", "")\nenv = dict(os.environ); env["LD_LIBRARY_PATH"] = LD + ":" + env.get("LD_LIBRARY_PATH", "")\nwith open("run_T.log", "w") as lg:\n    subprocess.run(["stdbuf", "-oL", "-eL", BIN, DECK_T, OUT_T], stdout=lg, stderr=subprocess.STDOUT, env=env, timeout=1200)\nHEADER_U = f\'\'\'TITLE:\n  - "run U"\nPROBLEM SIZE:\n  DIM: 3\nIO:\n  STRUCT_STRESS: "No"\n  STRUCT_STRAIN: "No"\nIO/MONITOR STRUCTURE DBC:\n  INTERVAL_STEPS: 1\n  FILE_TYPE: yaml\n  WRITE_CONDITION_INFORMATION: true\nPROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nSTRUCTURAL DYNAMIC:\n  DYNAMICTYPE: "Statics"\n  TIMESTEP: 1\n  NUMSTEP: 1\n  MAXTIME: 1\n  LINEAR_SOLVER: 2\nTHERMAL DYNAMIC:\n  DYNAMICTYPE: Statics\n  TIMESTEP: 1\n  NUMSTEP: 1\n  MAXTIME: 1\n  LINEAR_SOLVER: 1\nTSI DYNAMIC:\n  COUPALGO: "tsi_oneway"\n  NUMSTEP: 1\n  MAXTIME: 1\n  TIMESTEP: 1\n  ITEMAX: 1\nTSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\nIO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  OUTPUT_DATA_FORMAT: "ascii"\nIO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true\nTHERMAL DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_THERMO: true\n  TEMPERATURE: true\nSOLVER 1:\n  SOLVER: "UMFPACK"\n  NAME: "Thermal_Solver"\nSOLVER 2:\n  SOLVER: "UMFPACK"\n  NAME: "Structure_Solver"\nMATERIALS:\n  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n      YOUNGNUM: 1\n      YOUNG: [{E_MOD:.17g}]\n      NUE: {NU:.17g}\n      DENS: 1\n      THEXPANS: {ALPHA:.17g}\n      INITTEMP: 0\n      THERMOMAT: 2\n  - MAT: 2\n    MAT_Fourier:\n      CAPA: 1\n      CONDUCT:\n        constant: [{KV}]\nCLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\n    SRC_MAT: 1\n    TAR_FIELD: "thermo"\n    TAR_MAT: 2\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"\nFUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"\nFUNCT3:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\nDESIGN VOL NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 0]\n    VAL: [1.0, 1.0, 0.0]\n    FUNCT: [1, 2, 0]\nDESIGN VOL THERMO NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [3]\nDESIGN VOL DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [0, 0, 1]\n    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]\nDESIGN SURF DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]\nDESIGN SURF THERMO DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0]\n    FUNCT: [0]\n\'\'\'\nDECK_U = "deck_U.4C.yaml"; OUT_U = "out_U"\nPath(DECK_U).write_text(HEADER_U + deck_U_tables())\nwith open("run_U.log", "w") as lg:\n    subprocess.run(["stdbuf", "-oL", "-eL", BIN, DECK_U, OUT_U], stdout=lg, stderr=subprocess.STDOUT, env=env, timeout=1200)\n'
# The deck TABLES (node classification, slab numbering, point conditions, topology): part of
# the fill since 2026-09-11, when they stopped being served (the deck is the agent's).
FILL_4C_TABLES = 'N2 = len(nodes)\n_TOL = 1e-9 * max(X1 - X0, Y1 - Y0)\n_AX, _VAL = {"left": (0, X0), "right": (0, X1), "bottom": (1, Y0), "top": (1, Y1)}[IF]\ndef _on_edge(n):\n    x, y = nodes[n - 1]\n    return abs(x - X0) < _TOL or abs(x - X1) < _TOL or abs(y - Y0) < _TOL or abs(y - Y1) < _TOL\niface_all = sorted([n for n in range(1, N2 + 1) if abs(nodes[n - 1][_AX] - _VAL) < _TOL],\n                   key=lambda n: nodes[n - 1][1 - _AX])\nif len(iface_all) < 3:\n    raise SystemExit(f"only {len(iface_all)} node(s) lie on the {IF} edge x/y = {_VAL}: `nodes` and config disagree")\ninterior = iface_all[1:-1]                      # the two ENDPOINTS keep the outer datum (they are outer nodes)\nouter = [n for n in range(1, N2 + 1) if _on_edge(n) and n not in set(interior)]\nTZ = min((X1 - X0) / NX, (Y1 - Y0) / NY)        # slab thickness: one well-shaped HEX8 layer\ndef gid3(n, layer):\n    """4C node id of 2-D node n on slab layer 0 (z = 0) or 1 (z = TZ): the second layer follows the first."""\n    return n + layer * N2\n_g = {n: partner_values(nodes[n - 1][1 - _AX]) for n in interior}     # (T, ux, uy) per interior node\n\ndef scatra_point_conditions():\n    """Deck T: `DESIGN POINT DIRICH CONDITIONS` entries imposing the partner\'s T on the interior interface\n    nodes, and the matching `DNODE-NODE TOPOLOGY` entries (DNODE ids 1..len(interior))."""\n    cond, topo = [], []\n    for d, n in enumerate(interior, 1):\n        cond.append(f"  - E: {d}\\n    NUMDOF: 1\\n    ONOFF: [1]\\n    VAL: [{_g[n][0]:.17g}]\\n    FUNCT: [0]\\n")\n        topo.append(f\'  - "NODE {n} DNODE {d}"\\n\')\n    return "".join(cond), "".join(topo)\n\ndef slab_point_conditions():\n    """Deck U: the partner\'s (ux, uy) as `DESIGN POINT DIRICH CONDITIONS` entries (NUMDOF 3, u_z = 0,\n    TAG: monitor_reaction) and its T as `DESIGN POINT THERMO DIRICH CONDITIONS` entries, on BOTH slab\n    layers of every interior interface node, with one DNODE per node-and-layer, ids continuous across\n    the two families; plus the matching `DNODE-NODE TOPOLOGY` entries."""\n    struct, thermo, topo = [], [], []\n    d = 0\n    for n in interior:\n        T, ux, uy = _g[n]\n        for layer in (0, 1):\n            d += 1\n            struct.append(f"  - E: {d}\\n    NUMDOF: 3\\n    ONOFF: [1, 1, 1]\\n    VAL: [{ux:.17g}, {uy:.17g}, 0.0]\\n"\n                          f"    FUNCT: [0, 0, 0]\\n    TAG: monitor_reaction\\n")\n            thermo.append(f"  - E: {d}\\n    NUMDOF: 1\\n    ONOFF: [1]\\n    VAL: [{T:.17g}]\\n    FUNCT: [0]\\n")\n            topo.append(f\'  - "NODE {gid3(n, layer)} DNODE {d}"\\n\')\n    return "".join(struct), "".join(thermo), "".join(topo)\n\ndef scatra_topology():\n    """Deck T: `DLINE-NODE TOPOLOGY` (DLINE 1 = the outer boundary nodes, DLINE 2 = every interface node)\n    and `DSURF-NODE TOPOLOGY` (every node on DSURFACE 1, the surface the source load acts on)."""\n    dline = "".join(f\'  - "NODE {n} DLINE 1"\\n\' for n in outer) + "".join(f\'  - "NODE {n} DLINE 2"\\n\' for n in iface_all)\n    dsurf = "".join(f\'  - "NODE {n} DSURFACE 1"\\n\' for n in range(1, N2 + 1))\n    return dline, dsurf\n\ndef slab_topology():\n    """Deck U: `DSURF-NODE TOPOLOGY` (the outer boundary nodes of BOTH layers on DSURFACE 1) and\n    `DVOL-NODE TOPOLOGY` (every node of both layers on DVOL 1)."""\n    dsurf = "".join(f\'  - "NODE {gid3(n, layer)} DSURFACE 1"\\n\' for layer in (0, 1) for n in outer)\n    dvol = "".join(f\'  - "NODE {n} DVOL 1"\\n\' for n in range(1, 2 * N2 + 1))\n    return dsurf, dvol\n\ndef deck_T_tables():\n    """Deck T\'s TABLES, from YOUR nodes and quads and the partner\'s data: the interface point\n    conditions, NODE COORDS, TRANSPORT ELEMENTS and the DLINE / DNODE / DSURF topology. Append this to\n    YOUR deck-T header (the sections listed below; do not repeat any of these sections in the header)."""\n    cond, dn = scatra_point_conditions()\n    dline, dsurf = scatra_topology()\n    out = "DESIGN POINT DIRICH CONDITIONS:\\n" + cond\n    out += "NODE COORDS:\\n" + "".join(f\'  - "NODE {n} COORD {x:.15f} {y:.15f} 0.0"\\n\' for n, (x, y) in enumerate(nodes, 1))\n    out += "TRANSPORT ELEMENTS:\\n" + "".join(f\'  - "{e} TRANSP QUAD4 {a} {b} {c} {d} MAT 1 TYPE Std"\\n\'\n                                             for e, (a, b, c, d) in enumerate(quads, 1))\n    out += "DLINE-NODE TOPOLOGY:\\n" + dline + "DNODE-NODE TOPOLOGY:\\n" + dn + "DSURF-NODE TOPOLOGY:\\n" + dsurf\n    return out\n\ndef deck_U_tables():\n    """Deck U\'s TABLES, from YOUR nodes and quads and the partner\'s data: both point-condition families,\n    NODE COORDS of both slab layers, SOLIDSCATRA HEX8 elements and the DNODE / DSURF / DVOL topology.\n    Append this to YOUR deck-U header (do not repeat any of these sections in the header)."""\n    struct, thermo, dn = slab_point_conditions()\n    dsurf, dvol = slab_topology()\n    out = "DESIGN POINT DIRICH CONDITIONS:\\n" + struct + "DESIGN POINT THERMO DIRICH CONDITIONS:\\n" + thermo\n    out += "NODE COORDS:\\n" + "".join(f\'  - "NODE {gid3(n, 0)} COORD {x:.15f} {y:.15f} 0.0"\\n\' for n, (x, y) in enumerate(nodes, 1))\n    out += "".join(f\'  - "NODE {gid3(n, 1)} COORD {x:.15f} {y:.15f} {TZ:.15f}"\\n\' for n, (x, y) in enumerate(nodes, 1))\n    out += "STRUCTURE ELEMENTS:\\n" + "".join(\n        f\'  - "{e} SOLIDSCATRA HEX8 {a} {b} {c} {d} {gid3(a, 1)} {gid3(b, 1)} {gid3(c, 1)} {gid3(d, 1)} \'\n        f\'MAT 1 KINEM linear TYPE Undefined"\\n\' for e, (a, b, c, d) in enumerate(quads, 1))\n    out += "DNODE-NODE TOPOLOGY:\\n" + dn + "DSURF-NODE TOPOLOGY:\\n" + dsurf + "DVOL-NODE TOPOLOGY:\\n" + dvol\n    return out\n'


@pytest.mark.skipif(not (FOURC.is_file() and FENICS_PY.is_file()), reason="4C or the dolfinx python not on this host")
@pytest.mark.parametrize("deck_u_as", ["path", "text"])
def test_the_4c_thermoelastic_contract_recovers_flux_and_traction(tmp_path, deck_u_as):
    """deck_u_as='text': the worker leaves the assembled deck TEXT in DECK_U instead of its
    file name (measured on a trial script that was otherwise exact); the recovery takes both."""
    m = _manufactured()
    src = (PART / "participant_fourc_thermoelastic.py").read_text()
    a1 = src.index(BEGIN); b1 = src.index(END, a1) + len(END)
    assert src.find(BEGIN, b1) < 0, "the 4C thermo-elastic contract has ONE hole: mesh, decks, runs"
    fill2 = FILL_4C_H2
    if deck_u_as == "text":
        fill2 = fill2 + "DECK_U = HEADER_U + deck_U_tables()\n"
        assert "DECK_U = HEADER_U" in fill2
    (tmp_path / "participant_A.py").write_text(src[:a1] + FILL_4C_H1 + FILL_4C_TABLES + fill2 + src[b1:])
    f4 = lambda e: str(e).replace("**", "^")
    (tmp_path / "config.json").write_text(json.dumps({
        "level": 1, "nx": 8, "ny": 10, "x0": 0.0, "x1": LX, "y0": 0.0, "y1": LY, "k": KV, "lam": LAM, "mu": MU,
        "beta": BETA, "iface": "right", "side": "dirichlet", "source_T": f4(m["fT"]), "source_ux": f4(m["fx"]),
        "source_uy": f4(m["fy"]), "fourc_bin": str(FOURC), "fourc_ld": "/opt/4C-dependencies/lib"}))
    imp = _imports(m, 10)
    (tmp_path / "imports.json").write_text(json.dumps({"B": imp["P"]}))
    # the coupling tool's own capture of an EARLIER run lies in the same directory; it must never be
    # re-echoed (measured: six three-level couplings lost when each run's stdout began with the previous
    # level's capture and the grader read the stale first NDOF line)
    (tmp_path / "participant_output.log").write_text("iteration: 9\n--- stdout ---\n── 4C console stale.log ──\n4C banner\nNDOF = 999\n")
    r = subprocess.run([str(FENICS_PY), "participant_A.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "\nNDOF = 297\n" in "\n" + r.stdout
    # the participant's stdout carries 4C's OWN console lines (the coupling tool captures stdout as the
    # level's run log; a run log is credited to 4C by these lines, never by the NDOF line alone)
    from blind_eval.evidence import PER_CODE_SIGNATURES
    import re as _re
    assert any(_re.search(p, r.stdout, _re.M) for p in PER_CODE_SIGNATURES["4C"]), r.stdout[-1500:]
    assert "NDOF = 999" not in r.stdout and r.stdout.count("── 4C console") == 2, r.stdout.count("── 4C console")
    assert _re.search(r"^\s*NDOF\s*=\s*(\d+)\s*$", r.stdout, _re.M).group(1) == "297"           # the FIRST canonical line is this run's
    e = json.loads((tmp_path / "exports.json").read_text())
    assert e["values"] == []
    C = np.asarray(e["coordinates"], float); Q = np.asarray(e["normal_fluxes"], float)
    assert Q.shape == (9, 3)
    for c, key, tol in ((0, "qn", 0.02), (1, "qx", 0.05), (2, "qy", 0.05)):
        ex = m[key](C[:, 0], C[:, 1])
        err = np.abs(Q[:, c] - ex).max() / np.abs(ex).max()
        assert err < tol, f"{key}: relative error {err:.3e} against the manufactured solution"
    assert (tmp_path / "interface_level1.csv").read_text().startswith("x,y,T,ux,uy,qn,tx,ty")


def _fenics_program(side, m, nx=8, ny=10):
    src = (PART / "participant_fenics_thermoelastic.py").read_text()
    a = src.index("# ── EDIT THIS BLOCK")
    b = src.index("# ─────────────────────────────────────────────────────────────────────────\n", a)
    fT, fx, fy = m["sym"]
    block = (f'# ── EDIT THIS BLOCK (fill)\nSIDE = "{side}"\nPARTNER = "P"\nX0, X1 = 0.0, {LX}\nY0, Y1 = 0.0, {LY}\n'
             f'IFACE_X = {LX}\nK = {KV}\nLAM, MU = {LAM}, {MU}\nBETA = {BETA}\n\n\n'
             f'def F_T(x, y):\n    return {m["npy"](fT)} + 0.0 * x\n\n\n'
             f'def F_U(x, y):\n    return ({m["npy"](fx)} + 0.0 * x, {m["npy"](fy)} + 0.0 * x)\n'
             f'T_OUTER = 0.0\nUX_OUTER, UY_OUTER = 0.0, 0.0\nNX, NY = {nx}, {ny}\n'
             f'T_INIT, UX_INIT, UY_INIT = 0.0, 0.0, 0.0\nQ_INIT = (0.0, 0.0, 0.0)\n')
    return src[:a] + block + src[b:]


@pytest.mark.skipif(not FENICS_PY.is_file(), reason="the dolfinx python is not on this host")
@pytest.mark.parametrize("side", ["neumann", "dirichlet"])
def test_the_fenics_thermoelastic_contract_in_both_roles(tmp_path, side):
    m = _manufactured()
    (tmp_path / "participant.py").write_text(_fenics_program(side, m))
    (tmp_path / "imports.json").write_text(json.dumps(_imports(m)))
    r = subprocess.run([str(FENICS_PY), "participant.py"], cwd=tmp_path, capture_output=True, text=True, timeout=600,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "\nNDOF = 297\n" in "\n" + r.stdout
    e = json.loads((tmp_path / "exports.json").read_text())
    C = np.asarray(e["coordinates"], float); V = np.asarray(e["values"], float); Q = np.asarray(e["normal_fluxes"], float)
    assert V.shape == (11, 3) and Q.shape == (11, 3)
    inner = (C[:, 1] > 1e-9) & (C[:, 1] < LY - 1e-9)
    if side == "neumann":
        for c, key, tol in ((0, "T", 0.03), (1, "ux", 0.03), (2, "uy", 0.03)):
            ex = m[key](C[:, 0], C[:, 1])
            err = np.abs(V[inner, c] - ex[inner]).max() / np.abs(ex).max()
            assert err < tol, f"{key}: trace error {err:.3e}"
    else:
        for c, key, tol in ((0, "qn", 0.05), (1, "qx", 0.02), (2, "qy", 0.03)):
            ex = m[key](C[:, 0], C[:, 1])
            err = np.abs(Q[inner, c] - ex[inner]).max() / np.abs(ex).max()
            assert err < tol, f"{key}: flux error {err:.3e}"
