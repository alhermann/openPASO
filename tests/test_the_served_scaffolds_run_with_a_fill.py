"""The served coupling SCAFFOLDS run, with a fill openPASO never serves.

Option B elides the solve; what openPASO does serve (handshake, mapped trace,
UFL constants, mesh access, flux recovery, finish diagnosis, self-checks,
exports) must be executable and RIGHT. This test fills each scaffold's holes
with a minimal solve kept here, runs it on a manufactured solution, and
checks the exported interface flux against the exact one. These fills are
test fixtures: they never reach an agent, and the tree scan of the custody
preflight skips this repository.

Manufactured solution on B = (0.6, 1.4) x (0, 1), k = 5:
    u = (1.4 - x) sin(pi y),  f = 5 pi^2 (1.4 - x) sin(pi y)
    outward flux at x = 0.6 (normal -x):  q = -5 sin(pi y)
DUNE on A = (0, 0.6) x (0, 1), k = 1:  u = x sin(pi y), f = pi^2 x sin(pi y),
    outward flux at x = 0.6 (normal +x):  q = -sin(pi y).
"""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
import backend_probe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
BAR = "# ── SOLVE ─ openPASO DOES NOT SERVE THIS ─────────────────────────────────────\n"
FOURC = backend_probe.fourc_binary()
FOURC_PY = backend_probe.fenics_python()
DUNE_PY = Path("/home/user/miniconda3/envs/dune-py313/bin/python")


def _scaffold(solver: str) -> str:
    from tools.consolidated import _coupling_participant_script
    t = _coupling_participant_script(solver)
    t = t[t.index('"""'):]
    return "\n".join(l for l in t.splitlines() if not l.strip().startswith("```")) + "\n"


FILL_4C = '# ---- MY FILL (validation only, never served): Scalar_Transport deck on the subdomain ----\nimport os, subprocess\nnodes = []\nfor j in range(NY + 1):\n    y = Y0 + j * (Y1 - Y0) / NY\n    for i in range(NX + 1):\n        x = X0 + i * (X1 - X0) / NX\n        nodes.append((x, y))\nIFX = X0 if IF == "left" else X1\niface_ids = [k + 1 for k, (x, y) in enumerate(nodes) if abs(x - IFX) < 1e-9]\nouter_ids = [k + 1 for k, (x, y) in enumerate(nodes) if abs(x - (X1 if IF == "left" else X0)) < 1e-9 or abs(y - Y0) < 1e-9 or abs(y - Y1) < 1e-9]\ninterior = [n for n in iface_ids if not (abs(nodes[n-1][1] - Y0) < 1e-9 or abs(nodes[n-1][1] - Y1) < 1e-9)]\nh = (Y1 - Y0) / NY\nL = ["TITLE:", \'  - "validation fill"\', "PROBLEM SIZE:", f"  ELEMENTS: {NX*NY}", f"  NODES: {len(nodes)}",\n     "PROBLEM TYPE:", \'  PROBLEMTYPE: "Scalar_Transport"\',\n     "SCALAR TRANSPORT DYNAMIC:", \'  TIMEINTEGR: "Stationary"\', \'  SOLVERTYPE: "linear_full"\', "  NUMSTEP: 1", "  TIMESTEP: 1.0", "  MAXTIME: 1.0", "  LINEAR_SOLVER: 1", \'  CALCFLUX_BOUNDARY: "diffusive"\',\n     "SOLVER 1:", \'  SOLVER: "UMFPACK"\', "MATERIALS:", "  - MAT: 1", "    MAT_scatra:", f"      DIFFUSIVITY: {KV}",\n     "FUNCT1:", f\'  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "{SRC_EXPR}"\',\n     "DESIGN SURF TRANSPORT NEUMANN CONDITIONS:", "  - E: 1", "    NUMDOF: 1", "    ONOFF: [1]", "    VAL: [1.0]", "    FUNCT: [1]",\n     "DESIGN LINE DIRICH CONDITIONS:", "  - E: 2", "    NUMDOF: 1", "    ONOFF: [1]", "    VAL: [0.0]", "    FUNCT: [0]"]\ndn = {}\nif SIDE == "neumann":\n    L.append("DESIGN POINT NEUMANN CONDITIONS:")\n    for idx, n in enumerate(interior):\n        dn[n] = 3 + idx; y = nodes[n-1][1]\n        g = partner_flux(y); gp = partner_flux(y - h); gn = partner_flux(y + h)\n        L += [f"  - E: {dn[n]}", "    NUMDOF: 1", "    ONOFF: [1]", f"    VAL: [{(h/6.0)*(gp + 4*g + gn)}]", "    FUNCT: [0]"]\nelse:\n    L.append("DESIGN POINT DIRICH CONDITIONS:")\n    for idx, n in enumerate(interior):\n        dn[n] = 3 + idx; y = nodes[n-1][1]\n        L += [f"  - E: {dn[n]}", "    NUMDOF: 1", "    ONOFF: [1]", f"    VAL: [{partner_value(y)}]", "    FUNCT: [0]"]\nL.append("NODE COORDS:")\nL += [f\'  - "NODE {k+1} COORD {x:.11e} {y:.11e} 0.0"\' for k, (x, y) in enumerate(nodes)]\nL.append("TRANSPORT ELEMENTS:")\ne = 1\nfor j in range(NY):\n    for i in range(NX):\n        n1 = j*(NX+1)+i+1; n2 = n1+1; n3 = (j+1)*(NX+1)+i+2; n4 = n3-1\n        L.append(f\'  - "{e} TRANSP QUAD4 {n1} {n2} {n3} {n4} MAT 1 TYPE Std"\'); e += 1\nL.append("DLINE-NODE TOPOLOGY:")\nL += [f\'  - "NODE {n} DLINE 2"\' for n in outer_ids] + [f\'  - "NODE {n} DLINE 5"\' for n in iface_ids]\nL.append("DSURF-NODE TOPOLOGY:")\nL += [f\'  - "NODE {k+1} DSURFACE 1"\' for k in range(len(nodes))]\nL.append("DNODE-NODE TOPOLOGY:")\nL += [f\'  - "NODE {n} DNODE {d}"\' for n, d in dn.items()]\nL += ["SCATRA FLUX CALC LINE CONDITIONS:", "  - E: 5"]\nDECK = "deck.4C.yaml"; Path(DECK).write_text("\\n".join(L) + "\\n")\nFOURC_BIN = CFG.get("fourc_bin", "__FOURC_BIN__")\nenv = dict(os.environ, LD_LIBRARY_PATH=CFG.get("fourc_ld", "/opt/4C-dependencies/lib") + ":" + os.environ.get("LD_LIBRARY_PATH", ""))\nwith open("run.log", "w") as lf:\n    subprocess.run(["stdbuf", "-oL", "-eL", FOURC_BIN, DECK, "out"], env=env, stdout=lf, stderr=lf, text=True)\n\n\n\n\n'
# The generated participant is written to disk and run by a plain Python, so
# it cannot call back into the test's helpers. The 4C binary is substituted
# into the template here, as a literal path, when the template is defined.
FILL_4C = FILL_4C.replace("__FOURC_BIN__", str(FOURC))


@pytest.mark.skipif(not (FOURC.is_file() and FOURC_PY.is_file()), reason="4C or its python not on this host")
@pytest.mark.parametrize("side", ["dirichlet", "neumann"])
def test_the_4c_scaffold_recovers_the_manufactured_flux_in_both_roles(tmp_path, side):
    parts = _scaffold("fourc").split(BAR)
    assert len(parts) == 3, "the 4C scaffold has one hole"
    (tmp_path / "participant_B.py").write_text(parts[0] + FILL_4C + parts[2])
    (tmp_path / "config.json").write_text(json.dumps({
        "level": 1, "nx": 8, "ny": 10, "x0": 0.6, "x1": 1.4, "y0": 0.0, "y1": 1.0, "k": 5.0,
        "iface": "left", "side": side, "source_expr": "5*pi^2*(1.4-x)*sin(pi*y)",
        "fourc_bin": str(FOURC), "fourc_ld": "/opt/4C-dependencies/lib"}))
    ys = [i / 10 for i in range(11)]
    (tmp_path / "imports.json").write_text(json.dumps({"A": {
        "field_name": "u", "n_points": 11, "coordinates": [[0.6, y] for y in ys],
        "values": [0.8 * math.sin(math.pi * y) for y in ys],
        "normal_fluxes": [5.0 * math.sin(math.pi * y) for y in ys]}}))
    r = subprocess.run([str(FOURC_PY), "participant_B.py"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=600, env=dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg"))
    assert r.returncode == 0, r.stderr[-1500:]
    e = json.loads((tmp_path / "exports.json").read_text())
    err = max(abs(q + 5.0 * math.sin(math.pi * c[1])) for c, q in zip(e["coordinates"], e["normal_fluxes"]))
    assert err < 0.1, f"{side}: flux error {err:.3f} against the manufactured solution"
    if side == "neumann":
        ev = max(abs(v - 0.8 * math.sin(math.pi * c[1])) for c, v in zip(e["coordinates"], e["values"]))
        assert ev < 0.02, f"neumann trace error {ev:.4f}"
    else:
        assert e["values"] == []


def test_a_wrongly_scaled_neumann_load_is_refused_by_the_served_check(tmp_path):
    if not (FOURC.is_file() and FOURC_PY.is_file()):
        pytest.skip("4C not on this host")
    parts = _scaffold("fourc").split(BAR)
    bad = FILL_4C.replace('f"    VAL: [{(h/6.0)*(gp + 4*g + gn)}]"', 'f"    VAL: [{g}]"')
    assert bad != FILL_4C
    (tmp_path / "participant_B.py").write_text(parts[0] + bad + parts[2])
    (tmp_path / "config.json").write_text(json.dumps({
        "level": 1, "nx": 8, "ny": 10, "x0": 0.6, "x1": 1.4, "y0": 0.0, "y1": 1.0, "k": 5.0,
        "iface": "left", "side": "neumann", "source_expr": "5*pi^2*(1.4-x)*sin(pi*y)",
        "fourc_bin": str(FOURC), "fourc_ld": "/opt/4C-dependencies/lib"}))
    ys = [i / 10 for i in range(11)]
    (tmp_path / "imports.json").write_text(json.dumps({"A": {
        "field_name": "u", "n_points": 11, "coordinates": [[0.6, y] for y in ys],
        "values": [0.0] * 11, "normal_fluxes": [5.0 * math.sin(math.pi * y) for y in ys]}}))
    r = subprocess.run([str(FOURC_PY), "participant_B.py"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=600, env=dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg"))
    assert r.returncode != 0 and "does not match the load you applied" in (r.stdout + r.stderr)
    assert not (tmp_path / "exports.json").exists()


FILL_DUNE_1 = 'from dune.grid import cartesianDomain\nfrom dune.alugrid import aluConformGrid\nfrom dune.fem.space import lagrange\nfrom ufl import SpatialCoordinate\ngridView = aluConformGrid(cartesianDomain([X0, Y0], [X1, Y1], [NX, NY]))\nspace = lagrange(gridView, order=1)\nx = SpatialCoordinate(space)\n'
FILL_DUNE_2 = 'import math\nfrom dune.fem.scheme import galerkin\nfrom dune.ufl import DirichletBC\nfrom ufl import TrialFunction, TestFunction, dot, grad, dx, sin, pi, conditional, lt\nu, v = TrialFunction(space), TestFunction(space)\na_form = K_UFL*dot(grad(u), grad(v))*dx + C_UFL*u*v*dx\nb_form = pi*pi*x[0]*sin(pi*x[1])*v*dx\non_if_ufl = conditional(lt(abs(x[IF_COORD] - IF_VAL), _EPS), 1, 0)\nscheme = galerkin([a_form == b_form, DirichletBC(space, 0.0, 1 - on_if_ufl), DirichletBC(space, gtrace, on_if_ufl)], solver="cg")\nuh = space.interpolate(0, name="uh")\ninfo = scheme.solve(target=uh)\nF_SRC = lambda px, py: math.pi**2*px*math.sin(math.pi*py)\n'


@pytest.mark.skipif(not DUNE_PY.is_file(), reason="DUNE-fem python not on this host")
def test_the_dune_scaffold_recovers_the_manufactured_flux(tmp_path):
    parts = _scaffold("dune").split(BAR)
    assert len(parts) == 5, "the DUNE scaffold has two holes"
    (tmp_path / "participant_A.py").write_text(parts[0] + FILL_DUNE_1 + parts[2] + FILL_DUNE_2 + parts[4])
    (tmp_path / "config.json").write_text(json.dumps({"level": 1, "nx": 6, "ny": 10, "x0": 0.0, "x1": 0.6,
                                                      "y0": 0.0, "y1": 1.0, "k": 1.0, "reaction": 0.0, "iface": "right"}))
    ys = [i / 10 for i in range(11)]
    (tmp_path / "imports.json").write_text(json.dumps({"B": {
        "field_name": "u", "n_points": 11, "coordinates": [[0.6, y] for y in ys],
        "values": [0.6 * math.sin(math.pi * y) for y in ys], "normal_fluxes": [0.0] * 11}}))
    r = subprocess.run([str(DUNE_PY), "participant_A.py"], cwd=tmp_path, capture_output=True, text=True,
                       timeout=1500, env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode == 0, r.stderr[-1500:]
    e = json.loads((tmp_path / "exports.json").read_text())
    err = max(abs(q + math.sin(math.pi * c[1])) for c, q in zip(e["coordinates"], e["normal_fluxes"]))
    assert err < 0.05, f"DUNE flux error {err:.3f} against the manufactured solution"


FILL_DUNE_2_SERVED_SOURCE = (
    "from dune.fem.scheme import galerkin\nfrom dune.ufl import DirichletBC\n"
    "from ufl import TrialFunction, TestFunction, dot, grad, dx, conditional, lt\n"
    "u, v = TrialFunction(space), TestFunction(space)\n"
    "a_form = K_UFL*dot(grad(u), grad(v))*dx + C_UFL*u*v*dx\n"
    "from ufl import sin as _usin, pi as _upi\n"
    "b_form = eval(SRC_EXPR, {'__builtins__': {}}, {'x': x[0], 'y': x[1], 'sin': _usin, 'pi': _upi})*v*dx\n"
    "on_if_ufl = conditional(lt(abs(x[IF_COORD] - IF_VAL), _EPS), 1, 0)\n"
    "scheme = galerkin([a_form == b_form, DirichletBC(space, 0.0, 1 - on_if_ufl), DirichletBC(space, gtrace, on_if_ufl)], solver='cg')\n"
    "uh = space.interpolate(0, name='uh')\n"
    "info = scheme.solve(target=uh)\n")


def _dune_run(tmp_path, fill2, source_expr):
    parts = _scaffold("dune").split(BAR)
    (tmp_path / "participant_A.py").write_text(parts[0] + FILL_DUNE_1 + parts[2] + fill2 + parts[4])
    (tmp_path / "config.json").write_text(json.dumps({"level": 1, "nx": 6, "ny": 10, "x0": 0.0, "x1": 0.6,
                                                      "y0": 0.0, "y1": 1.0, "k": 1.0, "reaction": 0.0, "iface": "right",
                                                      "source_expr": source_expr}))
    ys = [i / 10 for i in range(11)]
    (tmp_path / "imports.json").write_text(json.dumps({"B": {
        "field_name": "u", "n_points": 11, "coordinates": [[0.6, y] for y in ys],
        "values": [0.6 * math.sin(math.pi * y) for y in ys], "normal_fluxes": [0.0] * 11}}))
    return subprocess.run([str(DUNE_PY), "participant_A.py"], cwd=tmp_path, capture_output=True, text=True,
                          timeout=1500, env=dict(os.environ, MPLBACKEND="Agg"))


@pytest.mark.skipif(not DUNE_PY.is_file(), reason="DUNE-fem python not on this host")
def test_the_dune_scaffold_carries_the_task_source_from_config(tmp_path):
    """The task's source as a config string: served F_SRC for the consistent load; the
    fill writes the form's source itself from the same string (the test fixture's own
    eval, never served)."""
    r = _dune_run(tmp_path, FILL_DUNE_2_SERVED_SOURCE, "pi^2*x*sin(pi*y)")
    assert r.returncode == 0, r.stderr[-1500:]
    e = json.loads((tmp_path / "exports.json").read_text())
    err = max(abs(q + math.sin(math.pi * c[1])) for c, q in zip(e["coordinates"], e["normal_fluxes"]))
    assert err < 0.05, f"DUNE flux error {err:.3f} with the served source carrier"


@pytest.mark.skipif(not DUNE_PY.is_file(), reason="DUNE-fem python not on this host")
def test_a_form_that_drops_the_configured_source_is_refused(tmp_path):
    """Measured on a development cell: the polynomial source carried nowhere gave a
    smooth field 1.2e-2 off at every level, order 0.00. With the source in config and
    not in the form, the served interior-residual check must stop the export."""
    dropped = FILL_DUNE_2_SERVED_SOURCE.replace("b_form = eval(SRC_EXPR, {'__builtins__': {}}, {'x': x[0], 'y': x[1], 'sin': _usin, 'pi': _upi})*v*dx\n", "b_form = 1e-12*v*dx\n")
    assert dropped != FILL_DUNE_2_SERVED_SOURCE
    r = _dune_run(tmp_path, dropped, "pi^2*x*sin(pi*y)")
    assert r.returncode != 0
    assert "your form and config disagree" in r.stderr, r.stderr[-1500:]
    assert not (tmp_path / "exports.json").is_file()
