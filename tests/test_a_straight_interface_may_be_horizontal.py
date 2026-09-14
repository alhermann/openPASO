"""A served participant expresses a straight interface on EITHER axis, not only the vertical one.

Every contract in the corpus hard-coded the split as the line x = IFACE_X: the outward-normal selector,
the partner-sample column and the exported coordinates all assumed it. One development problem's
interface is the horizontal line y = 5/8, so its agents would have had to edit served lines -- the one
thing their brief tells them not to do -- and the export self-check would then have fired on a correct
side. The handshake is OASiS's own interface; expressing it on one axis only was our limitation.

IFACE_AXIS defaults to "x", so every vertical problem is unchanged, and the horizontal path is executed
here rather than asserted: the fill below is the test's own, never served."""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
ROOT = Path(__file__).resolve().parents[1]
PARTICIPANTS = ROOT / "data" / "coupling_participants"
VENV = Path("/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python")

AXIS_AWARE = ["participant_skfem.py", "participant_ngsolve.py", "participant_fenics.py",
              "participant_dune.py", "participant_febio.py", "participant_kratos_neumann.py",
              "participant_skfem_elastic.py", "participant_ngsolve_elastic.py",
              "participant_fenics_elastic.py", "participant_dune_elastic.py",
              "participant_febio_elastic.py"]


@pytest.mark.parametrize("name", AXIS_AWARE)
def test_the_contract_declares_the_axis_and_derives_everything_from_it(name):
    t = (PARTICIPANTS / name).read_text()
    assert 'IFACE_AXIS = "x"' in t, f"{name}: no axis knob"
    assert 'AX = 0 if IFACE_AXIS == "x" else 1' in t, f"{name}: the derived block ignores the knob"
    assert "AL = 1 - AX" in t
    # nothing outside the elided holes may still assume the vertical case
    B, E = "OASiS DOES NOT SERVE THIS ─ begin", "OASiS DOES NOT SERVE THIS ─ end"
    served, inside = [], False
    for line in t.split("\n"):
        if B in line:
            inside = True
            continue
        if E in line:
            inside = False
            continue
        if not inside:
            served.append(line)
    body = "\n".join(l for l in served if not l.strip().startswith("#"))
    assert 'c[1] for c in imp["coordinates"]' not in body, f"{name}: the partner sample still reads the y column"
    assert "[[float(IFACE_X), float(" not in body, f"{name}: the export still places the interface on x"


@pytest.mark.skipif(not VENV.is_file(), reason="the scikit-fem interpreter is not on this machine")
def test_the_vector_contract_runs_on_a_horizontal_interface(tmp_path):
    """Subdomain (0,1) x (0, 5/8) with the interface along its TOP edge, y = 5/8."""
    s = (PARTICIPANTS / "participant_skfem_elastic.py").read_text()
    for old, new in (('IFACE_AXIS = "x"', 'IFACE_AXIS = "y"'),
                     ("X0, X1    = 0.0, 0.55", "X0, X1    = 0.0, 1.0"),
                     ("Y0, Y1    = 0.0, 0.4", "Y0, Y1    = 0.0, 0.625"),
                     ("IFACE_X   = 0.55", "IFACE_X   = 0.625")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    # THE FILL IS THE TEST'S OWN. The hole is the agent's work, and for a horizontal split it selects
    # the interface on the other axis -- which is exactly what the served derived block now supports.
    old_hole = '''px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
y_if = py[iface_n]
outer_n = np.where((np.abs(px - OUTER_X) < TOL) |
                   (np.abs(py - Y0) < TOL) | (np.abs(py - Y1) < TOL))[0]'''
    new_hole = '''px, py = mesh.p[0], mesh.p[1]
_fix, _along = (px, py) if AX == 0 else (py, px)
iface_n = np.where(np.abs(_fix - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(_along[iface_n])]
y_if = _along[iface_n]
outer_n = np.where((np.abs(_fix - OUTER_X) < TOL) |
                   (np.abs(_along - ALO) < TOL) | (np.abs(_along - AHI) < TOL))[0]'''
    assert s.count(old_hole) == 1
    s = s.replace(old_hole, new_hole)
    old_bc = '''iface_bc_n = iface_n[(np.abs(py[iface_n] - Y0) > TOL) &
                     (np.abs(py[iface_n] - Y1) > TOL)]'''
    new_bc = '''iface_bc_n = iface_n[(np.abs(_along[iface_n] - ALO) > TOL) &
                     (np.abs(_along[iface_n] - AHI) > TOL)]'''
    assert s.count(old_bc) == 1
    s = s.replace(old_bc, new_bc)
    (tmp_path / "p.py").write_text(s)
    r = subprocess.run([str(VENV), "p.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"the horizontal side did not run:\\n{(r.stderr or r.stdout)[-1500:]}"
    e = json.loads((tmp_path / "exports.json").read_text())
    co = np.asarray(e["coordinates"], float)
    assert len(co) > 5
    assert np.allclose(co[:, 1], 0.625), "the exported points do not lie on the horizontal interface"
    assert np.all(np.diff(co[:, 0]) > 0), "the exported points are not ordered along the interface"
    vals = np.asarray(e["values"], float)
    assert vals.shape == (len(co), 2), "a vector side exports two components per point"
    assert np.isfinite(vals).all() and np.isfinite(np.asarray(e["normal_fluxes"], float)).all()
