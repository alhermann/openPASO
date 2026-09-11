"""The served THERMO-ELASTIC contracts run, with fills OASiS never serves.

Option B elides the solve; what OASiS serves for a temperature-plus-
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
BEGIN = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin"
END = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end"
FOURC = Path("/home/alexander/4C/build/4C")
FENICS_PY = Path("/home/alexander/miniconda3/envs/fenics/bin/python")
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


FILL_4C = '# ---- MY FILL (validation only, never served): two decks on the subdomain ----\nhx, hy = (X1 - X0) / NX, (Y1 - Y0) / NY\nnodes = [(X0 + i * hx, Y0 + j * hy) for j in range(NY + 1) for i in range(NX + 1)]\ndef gid(i, j): return 1 + i + (NX + 1) * j\non_if = {"left": lambda x, y: abs(x - X0) < 1e-9, "right": lambda x, y: abs(x - X1) < 1e-9,\n         "bottom": lambda x, y: abs(y - Y0) < 1e-9, "top": lambda x, y: abs(y - Y1) < 1e-9}[IF]\ndef on_outer(x, y):\n    faces = [abs(x - X0) < 1e-9, abs(x - X1) < 1e-9, abs(y - Y0) < 1e-9, abs(y - Y1) < 1e-9]\n    faces[{"left": 0, "right": 1, "bottom": 2, "top": 3}[IF]] = False\n    return any(faces)\niface_all = sorted([gid(i, j) for j in range(NY + 1) for i in range(NX + 1) if on_if(*nodes[gid(i, j) - 1])],\n                   key=lambda n: nodes[n - 1][1 if IF in ("left", "right") else 0])\ninterior = [n for n in iface_all if not on_outer(*nodes[n - 1])]\nouter = [gid(i, j) for j in range(NY + 1) for i in range(NX + 1) if on_outer(*nodes[gid(i, j) - 1])]\nax_ = 1 if IF in ("left", "right") else 0\ng = {n: partner_values(nodes[n - 1][ax_]) for n in interior}\n# ---- run T: scatra 2-D\npd, dn = [], []\nfor d, n in enumerate(interior, 1):\n    pd.append(f"  - E: {d}\\n    NUMDOF: 1\\n    ONOFF: [1]\\n    VAL: [{g[n][0]:.17g}]\\n    FUNCT: [0]\\n")\n    dn.append(f\'  - "NODE {n} DNODE {d}"\\n\')\ndeckT = f\'\'\'TITLE:\n  - "run T"\nPROBLEM SIZE:\n  DIM: 2\nPROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n  VELOCITYFIELD: "zero"\n  TIMESTEP: 1.0\n  NUMSTEP: 1\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n  CALCFLUX_BOUNDARY: "diffusive"\nIO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  OUTPUT_DATA_FORMAT: "ascii"\nSOLVER 1:\n  SOLVER: "UMFPACK"\n  NAME: "direct"\nMATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: {KV}\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\nDESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.0]\n    FUNCT: [0]\nDESIGN POINT DIRICH CONDITIONS:\n{"".join(pd)}DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [1]\nSCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\nNODE COORDS:\n\'\'\'\ndeckT += "".join(f\'  - "NODE {n} COORD {x:.15f} {y:.15f} 0.0"\\n\' for n, (x, y) in enumerate(nodes, 1))\ndeckT += "TRANSPORT ELEMENTS:\\n" + "".join(\n    f\'  - "{e} TRANSP QUAD4 {gid(i, j)} {gid(i + 1, j)} {gid(i + 1, j + 1)} {gid(i, j + 1)} MAT 1 TYPE Std"\\n\'\n    for e, (j, i) in enumerate(((j, i) for j in range(NY) for i in range(NX)), 1))\ndeckT += "DLINE-NODE TOPOLOGY:\\n" + "".join(f\'  - "NODE {n} DLINE 1"\\n\' for n in outer) + "".join(f\'  - "NODE {n} DLINE 2"\\n\' for n in iface_all)\ndeckT += "DNODE-NODE TOPOLOGY:\\n" + "".join(dn)\ndeckT += "DSURF-NODE TOPOLOGY:\\n" + "".join(f\'  - "NODE {n} DSURFACE 1"\\n\' for n in range(1, len(nodes) + 1))\nDECK_T = "deck_T.4C.yaml"; OUT_T = "out_T"\nPath(DECK_T).write_text(deckT)\nBIN = CFG.get("fourc_bin", os.environ.get("FOURC_BIN", "4C")); LD = CFG.get("fourc_ld", "")\nenv = dict(os.environ); env["LD_LIBRARY_PATH"] = LD + ":" + env.get("LD_LIBRARY_PATH", "")\nwith open("run_T.log", "w") as lg:\n    subprocess.run(["stdbuf", "-oL", "-eL", BIN, DECK_T, OUT_T], stdout=lg, stderr=subprocess.STDOUT, env=env, timeout=1200)\n# ---- run U: TSI slab\nTZ = min(hx, hy)\nN2 = len(nodes)\ndef g3(n, k): return n + k * N2\npds, pdt, dn3 = [], [], []\nd = 1\nfor n in interior:\n    for k in (0, 1):\n        pds.append(f"  - E: {d}\\n    NUMDOF: 3\\n    ONOFF: [1, 1, 1]\\n    VAL: [{g[n][1]:.17g}, {g[n][2]:.17g}, 0.0]\\n    FUNCT: [0, 0, 0]\\n    TAG: monitor_reaction\\n")\n        pdt.append(f"  - E: {d}\\n    NUMDOF: 1\\n    ONOFF: [1]\\n    VAL: [{g[n][0]:.17g}]\\n    FUNCT: [0]\\n")\n        dn3.append(f\'  - "NODE {g3(n, k)} DNODE {d}"\\n\'); d += 1\ndeckU = f\'\'\'TITLE:\n  - "run U"\nPROBLEM SIZE:\n  DIM: 3\nIO:\n  STRUCT_STRESS: "No"\n  STRUCT_STRAIN: "No"\nIO/MONITOR STRUCTURE DBC:\n  INTERVAL_STEPS: 1\n  FILE_TYPE: yaml\n  WRITE_CONDITION_INFORMATION: true\nPROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nSTRUCTURAL DYNAMIC:\n  DYNAMICTYPE: "Statics"\n  TIMESTEP: 1\n  NUMSTEP: 1\n  MAXTIME: 1\n  LINEAR_SOLVER: 2\nTHERMAL DYNAMIC:\n  DYNAMICTYPE: Statics\n  TIMESTEP: 1\n  NUMSTEP: 1\n  MAXTIME: 1\n  LINEAR_SOLVER: 1\nTSI DYNAMIC:\n  COUPALGO: "tsi_oneway"\n  NUMSTEP: 1\n  MAXTIME: 1\n  TIMESTEP: 1\n  ITEMAX: 1\nTSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\nIO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  OUTPUT_DATA_FORMAT: "ascii"\nIO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true\nTHERMAL DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_THERMO: true\n  TEMPERATURE: true\nSOLVER 1:\n  SOLVER: "UMFPACK"\n  NAME: "Thermal_Solver"\nSOLVER 2:\n  SOLVER: "UMFPACK"\n  NAME: "Structure_Solver"\nMATERIALS:\n  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n      YOUNGNUM: 1\n      YOUNG: [{E_MOD:.17g}]\n      NUE: {NU:.17g}\n      DENS: 1\n      THEXPANS: {ALPHA:.17g}\n      INITTEMP: 0\n      THERMOMAT: 2\n  - MAT: 2\n    MAT_Fourier:\n      CAPA: 1\n      CONDUCT:\n        constant: [{KV}]\nCLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\n    SRC_MAT: 1\n    TAR_FIELD: "thermo"\n    TAR_MAT: 2\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UX}"\nFUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_UY}"\nFUNCT3:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_T}"\nDESIGN VOL NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 0]\n    VAL: [1.0, 1.0, 0.0]\n    FUNCT: [1, 2, 0]\nDESIGN VOL THERMO NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [1.0]\n    FUNCT: [3]\nDESIGN VOL DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [0, 0, 1]\n    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]\nDESIGN SURF DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]\nDESIGN SURF THERMO DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0]\n    FUNCT: [0]\nDESIGN POINT DIRICH CONDITIONS:\n{"".join(pds)}DESIGN POINT THERMO DIRICH CONDITIONS:\n{"".join(pdt)}NODE COORDS:\n\'\'\'\ndeckU += "".join(f\'  - "NODE {g3(n, k)} COORD {x:.15f} {y:.15f} {k * TZ:.15f}"\\n\' for k in (0, 1) for n, (x, y) in enumerate(nodes, 1))\ndeckU += "STRUCTURE ELEMENTS:\\n" + "".join(\n    f\'  - "{e} SOLIDSCATRA HEX8 {gid(i, j)} {gid(i + 1, j)} {gid(i + 1, j + 1)} {gid(i, j + 1)} \'\n    f\'{g3(gid(i, j), 1)} {g3(gid(i + 1, j), 1)} {g3(gid(i + 1, j + 1), 1)} {g3(gid(i, j + 1), 1)} MAT 1 KINEM linear TYPE Undefined"\\n\'\n    for e, (j, i) in enumerate(((j, i) for j in range(NY) for i in range(NX)), 1))\ndeckU += "DNODE-NODE TOPOLOGY:\\n" + "".join(dn3)\ndeckU += "DSURF-NODE TOPOLOGY:\\n" + "".join(f\'  - "NODE {g3(n, k)} DSURFACE 1"\\n\' for k in (0, 1) for n in outer)\ndeckU += "DVOL-NODE TOPOLOGY:\\n" + "".join(f\'  - "NODE {n} DVOL 1"\\n\' for n in range(1, 2 * N2 + 1))\nDECK_U = "deck_U.4C.yaml"; OUT_U = "out_U"\nPath(DECK_U).write_text(deckU)\nwith open("run_U.log", "w") as lg:\n    subprocess.run(["stdbuf", "-oL", "-eL", BIN, DECK_U, OUT_U], stdout=lg, stderr=subprocess.STDOUT, env=env, timeout=1200)\n'


@pytest.mark.skipif(not (FOURC.is_file() and FENICS_PY.is_file()), reason="4C or the dolfinx python not on this host")
def test_the_4c_thermoelastic_contract_recovers_flux_and_traction(tmp_path):
    m = _manufactured()
    src = (PART / "participant_fourc_thermoelastic.py").read_text()
    a, b = src.index(BEGIN), src.index(END) + len(END)
    (tmp_path / "participant_A.py").write_text(src[:a] + FILL_4C + src[b:])
    f4 = lambda e: str(e).replace("**", "^")
    (tmp_path / "config.json").write_text(json.dumps({
        "level": 1, "nx": 8, "ny": 10, "x0": 0.0, "x1": LX, "y0": 0.0, "y1": LY, "k": KV, "lam": LAM, "mu": MU,
        "beta": BETA, "iface": "right", "side": "dirichlet", "source_T": f4(m["fT"]), "source_ux": f4(m["fx"]),
        "source_uy": f4(m["fy"]), "fourc_bin": str(FOURC), "fourc_ld": "/opt/4C-dependencies/lib"}))
    imp = _imports(m, 10)
    (tmp_path / "imports.json").write_text(json.dumps({"B": imp["P"]}))
    r = subprocess.run([str(FENICS_PY), "participant_A.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900,
                       env=dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg"))
    assert r.returncode == 0, r.stderr[-2000:]
    assert "\nNDOF = 297\n" in "\n" + r.stdout
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
