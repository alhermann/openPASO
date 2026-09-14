"""A served participant expresses a straight interface on EITHER axis, and one that BENDS.

Every contract in the corpus hard-coded the split as the line x = IFACE_X: the outward-normal selector,
the partner-sample column and the exported coordinates all assumed it. One development problem's
interface is the horizontal line y = 5/8, so its agents would have had to edit served lines -- the one
thing their brief tells them not to do -- and the export self-check would then have fired on a correct
side. The handshake is openPASO's own interface; expressing it on one axis only was our limitation.

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
VENV = Path("/home/user/Schreibtisch/open-fem-agent/.venv/bin/python")

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
    B, E = "openPASO DOES NOT SERVE THIS ─ begin", "openPASO DOES NOT SERVE THIS ─ end"
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


@pytest.mark.skipif(not VENV.is_file(), reason="the scikit-fem interpreter is not on this machine")
def test_the_contract_runs_on_an_interface_that_bends(tmp_path):
    """One development problem's interface is a two-leg polyline: a square corner cut out of the other
    subdomain, so the interface runs up one edge and along the next. No contract could express that --
    every one of them matched the partner by a single coordinate, which is not monotone along two legs.

    IFACE_SEGMENTS lists the legs in order and the exchange is matched by distance ALONG them. The case
    below is exactly that problem's side B: the square (1/2, 1) x (0, 1/2), whose left and top edges are
    the interface and whose bottom and right edges are the outer boundary. The fill is the test's own."""
    import json
    import numpy as np
    s = (PARTICIPANTS / "participant_skfem.py").read_text()
    for old, new in (("X0, X1    = 0.0, 0.6", "X0, X1    = 0.5, 1.0"),
                     ("Y0, Y1    = 0.0, 0.4", "Y0, Y1    = 0.0, 0.5"),
                     ("IFACE_SEGMENTS = ()",
                      'IFACE_SEGMENTS = (("x", 0.5, 0.0, 0.5), ("y", 0.5, 0.5, 1.0))'),
                     ("K         = 0.8", "K         = 2.5"),
                     ("T_OUTER   = 320.0", "T_OUTER   = 0.0"),
                     ("T_INIT    = 310.0", "T_INIT    = 0.0"),
                     ("NX, NY    = 24, 16", "NX, NY    = 16, 16")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    old_hole = """px, py = mesh.p[0], mesh.p[1]
iface_n = np.where(np.abs(px - IFACE_X) < TOL)[0]
iface_n = iface_n[np.argsort(py[iface_n])]             # sorted by y
y_if = py[iface_n]
iface_dofs = n2d[iface_n]
outer_dofs = n2d[np.where(np.abs(px - OUTER_X) < TOL)[0]]"""
    new_hole = """px, py = mesh.p[0], mesh.p[1]
iface_n = np.where((np.abs(px - 0.5) < TOL) | (np.abs(py - 0.5) < TOL))[0]
iface_n = iface_n[np.argsort(arc_length(np.column_stack([px[iface_n], py[iface_n]])))]
y_if = np.column_stack([px[iface_n], py[iface_n]])
iface_dofs = n2d[iface_n]
outer_dofs = n2d[np.where((np.abs(py - 0.0) < TOL) | (np.abs(px - 1.0) < TOL))[0]]"""
    assert s.count(old_hole) == 1
    s = s.replace(old_hole, new_hole)
    old_fb = """fbasis = FacetBasis(mesh, elem,
                    facets=mesh.facets_satisfying(
                        lambda p: np.abs(p[0] - IFACE_X) < TOL))"""
    new_fb = """fbasis = FacetBasis(mesh, elem,
                    facets=mesh.facets_satisfying(
                        lambda p: (np.abs(p[0] - 0.5) < TOL) | (np.abs(p[1] - 0.5) < TOL)))"""
    assert s.count(old_fb) == 1
    s = s.replace(old_fb, new_fb)
    (tmp_path / "p.py").write_text(s)
    pts, vals = [], []
    for t in np.linspace(0, 0.5, 11):
        pts.append([0.5, float(t)]); vals.append(float(np.sin(np.pi * t)))
    for t in np.linspace(0.5, 1.0, 11):
        pts.append([float(t), 0.5]); vals.append(float(np.sin(np.pi * t)))
    (tmp_path / "imports.json").write_text(json.dumps({"right": {
        "field_name": "temperature", "n_points": len(pts), "coordinates": pts,
        "values": vals, "normal_fluxes": [0.0] * len(pts)}}))
    r = subprocess.run([str(VENV), "p.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"the bent side did not run:\n{(r.stderr or r.stdout)[-1500:]}"
    e = json.loads((tmp_path / "exports.json").read_text())
    co = np.asarray(e["coordinates"], float)
    q = np.asarray(e["normal_fluxes"], float).ravel()
    on_leg1 = np.isclose(co[:, 0], 0.5)
    on_leg2 = np.isclose(co[:, 1], 0.5)
    assert on_leg1.sum() > 5 and on_leg2.sum() > 5, "the export covers only one leg"
    assert (on_leg1 | on_leg2).all(), "a point sits on neither leg"
    assert np.all(np.abs(q) > 1e-12), "the recovered flux is zero somewhere: the interface weights "\
                                      "did not cover both legs"
    # the corner (1/2, 1/2) lies on both legs; the first leg owns it, as arc_length does
    s_of = np.where(on_leg1, co[:, 1], 0.5 + (co[:, 0] - 0.5))
    assert np.all(np.diff(s_of) > -1e-12), "the exported points are not ordered along the polyline"


FENICS = Path("/home/user/miniconda3/envs/fenics/bin/python")


@pytest.mark.skipif(not FENICS.is_file(), reason="the FEniCSx interpreter is not on this machine")
def test_the_fenics_contract_runs_on_the_bent_interface_too(tmp_path):
    """The bent problem's side B is FEniCSx, so the bent path has to work there as well.

    This also pins the served interface DUMP, which wrote `IFACE_X` as the x of every row and so
    could not describe a bent interface at all -- it takes the point's own coordinates now."""
    import json
    import numpy as np
    s = (PARTICIPANTS / "participant_fenics.py").read_text()
    for old, new in (("X0, X1    = 0.0, 0.6", "X0, X1    = 0.5, 1.0"),
                     ("Y0, Y1    = 0.0, 0.4", "Y0, Y1    = 0.0, 0.5"),
                     ("IFACE_SEGMENTS = ()",
                      'IFACE_SEGMENTS = (("x", 0.5, 0.0, 0.5), ("y", 0.5, 0.5, 1.0))'),
                     ("IFACE_X   = 0.6", "IFACE_X   = 0.5"),
                     ('SIDE      = "dirichlet"', 'SIDE      = "neumann"'),
                     ('PARTNER   = "right"', 'PARTNER   = "left"'),
                     ("K         = 0.8", "K         = 2.5"),
                     ("T_OUTER   = 320.0", "T_OUTER   = 0.0"),
                     ("T_INIT    = 310.0", "T_INIT    = 0.0"),
                     ("NX, NY    = 24, 16", "NX, NY    = 16, 16")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    old_sel = """iface_dofs = np.where(np.abs(xy[:, 0] - IFACE_X) < 1e-10)[0]
iface_dofs = iface_dofs[np.argsort(xy[iface_dofs, 1])]   # constant order, always
y_if = xy[iface_dofs, 1]"""
    new_sel = """iface_dofs = np.where((np.abs(xy[:, 0] - 0.5) < 1e-10) | (np.abs(xy[:, 1] - 0.5) < 1e-10))[0]
iface_dofs = iface_dofs[np.argsort(arc_length(xy[iface_dofs, :2]))]
y_if = xy[iface_dofs, :2]"""
    assert s.count(old_sel) == 1
    s = s.replace(old_sel, new_sel)
    old_fac = """facets_if = dmesh.locate_entities_boundary(domain, fdim,
                                           lambda x: np.isclose(x[0], IFACE_X))"""
    new_fac = """facets_if = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[0], 0.5) | np.isclose(x[1], 0.5))"""
    assert s.count(old_fac) == 1
    s = s.replace(old_fac, new_fac)
    old_out = """outer = dmesh.locate_entities_boundary(domain, fdim,
                                       lambda x: np.isclose(x[0], OUTER_X))"""
    new_out = """outer = dmesh.locate_entities_boundary(
    domain, fdim, lambda x: np.isclose(x[1], 0.0) | np.isclose(x[0], 1.0))"""
    assert s.count(old_out) == 1
    s = s.replace(old_out, new_out)
    (tmp_path / "p.py").write_text(s)
    pts, q = [], []
    for t in np.linspace(0, 0.5, 11):
        pts.append([0.5, float(t)]); q.append(float(2.0 * np.sin(np.pi * t)))
    for t in np.linspace(0.5, 1.0, 11):
        pts.append([float(t), 0.5]); q.append(float(2.0 * np.sin(np.pi * t)))
    (tmp_path / "imports.json").write_text(json.dumps({"left": {
        "field_name": "temperature", "n_points": len(pts), "coordinates": pts,
        "values": [0.0] * len(pts), "normal_fluxes": q}}))
    r = subprocess.run([str(FENICS), "p.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"the bent FEniCSx side did not run:\n{(r.stderr or r.stdout)[-1500:]}"
    e = json.loads((tmp_path / "exports.json").read_text())
    co = np.asarray(e["coordinates"], float)
    q_out = np.asarray(e["normal_fluxes"], float).ravel()
    assert np.isclose(co[:, 0], 0.5).sum() > 5 and np.isclose(co[:, 1], 0.5).sum() > 5
    assert np.abs(q_out).max() > 1e-3, "the imported load never entered the system"
    dump = next(tmp_path.glob("interface_level*.csv"))
    rows = [l.split(",") for l in dump.read_text().strip().splitlines()[1:]]
    assert len(rows) == len(co)
    xs = {round(float(r[0]), 6) for r in rows}
    assert len(xs) > 2, "the interface dump still writes one x for every row"


@pytest.mark.skipif(not VENV.is_file(), reason="the NGSolve interpreter is not on this machine")
def test_the_ngsolve_vector_contract_runs_on_a_horizontal_interface(tmp_path):
    """The vector problem with a horizontal interface puts NGSolve on the side whose TOP edge is the
    interface. That path -- vector exchange, axis 'y', Neumann role -- is the one a round would use, so
    it is executed here rather than inferred from the axis knob being present."""
    import json
    import numpy as np
    s = (PARTICIPANTS / "participant_ngsolve_elastic.py").read_text()
    for old, new in (('IFACE_AXIS = "x"', 'IFACE_AXIS = "y"'),
                     ("X0, X1    = 0.0, 0.55", "X0, X1    = 0.0, 1.0"),
                     ("Y0, Y1    = 0.0, 0.4", "Y0, Y1    = 0.0, 0.625"),
                     ("IFACE_X   = 0.55", "IFACE_X   = 0.625"),
                     ('SIDE      = "dirichlet"', 'SIDE      = "neumann"'),
                     ('PARTNER   = "right"', 'PARTNER   = "top"'),
                     ("NX, NY    = 24, 16", "NX, NY    = 16, 10")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)
    old_geo = """geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("outer", "interface", "outer", "outer") if ON_RIGHT else
                      ("outer", "outer", "outer", "interface")))"""
    new_geo = """geo.AddRectangle((X0, Y0), (X1, Y1),
                 bcs=(("outer", "outer", "interface", "outer") if ON_RIGHT else
                      ("interface", "outer", "outer", "outer")))"""
    assert s.count(old_geo) == 1
    s = s.replace(old_geo, new_geo)
    old_sel = """iface_v = np.where(np.abs(vxy[:, 0] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, 1])]           # sorted by y
y_if = vxy[iface_v, 1]"""
    new_sel = """iface_v = np.where(np.abs(vxy[:, AX] - IFACE_X) < TOL)[0]
iface_v = iface_v[np.argsort(vxy[iface_v, AL])]
y_if = vxy[iface_v, AL]"""
    assert s.count(old_sel) == 1
    s = s.replace(old_sel, new_sel)
    old_out = "outer_v = np.where((np.abs(vxy[:, 0] - OUTER_X) < TOL) |"
    assert s.count(old_out) == 1
    s = s.replace(old_out, "outer_v = np.where((np.abs(vxy[:, AX] - OUTER_X) < TOL) |")
    (tmp_path / "p.py").write_text(s)
    xs = np.linspace(0, 1, 17)
    (tmp_path / "imports.json").write_text(json.dumps({"top": {
        "field_name": "displacement", "n_points": len(xs),
        "coordinates": [[float(x), 0.625] for x in xs],
        "values": [[0.0, 0.0] for _ in xs],
        "normal_fluxes": [[0.0, float(300.0 * np.sin(np.pi * x))] for x in xs]}}))
    r = subprocess.run([str(VENV), "p.py"], cwd=tmp_path, capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"the horizontal vector side did not run:\n{(r.stderr or r.stdout)[-1500:]}"
    e = json.loads((tmp_path / "exports.json").read_text())
    co = np.asarray(e["coordinates"], float)
    u = np.asarray(e["values"], float)
    assert np.allclose(co[:, 1], 0.625), "the exported points are not on the horizontal interface"
    assert u.shape == (len(co), 2), "a vector side exports two components per point"
    assert np.abs(u).max() > 1e-4, "the imposed traction produced no displacement"
    assert np.isfinite(np.asarray(e["normal_fluxes"], float)).all()
