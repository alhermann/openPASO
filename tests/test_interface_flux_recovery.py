"""ASSEMBLY SELF-CONSISTENCY of the interface mechanism. NOT a convergence study.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS FILE. The convergence order of
the reaction recovery is measured in
`tests/test_interface_flux_converges_to_a_known_exact_flux.py`, against an
analytic flux on the Dirichlet side. It is NOT measured here, and it cannot be.

WHAT THIS FILE DOES. It imposes a flux that VARIES along the interface,
q(y) = 2 + 3 sin(4y), hands it to the NEUMANN side as if a partner had exported
it, and asks the participant to give it back. The participant applies the
partner's number as +int_Gamma g v ds, so its own outward flux must come back
as -g.

WHY THAT IS AN IDENTITY AND NOT A MEASUREMENT. On the Neumann side the
participant solves `A u = b_vol + M_Gamma g`. The recovery forms
`r = A u - b_vol`, so on every FREE interface row

    r_i = (M_Gamma g)_i          EXACTLY, for any A, any b_vol, any solution,

and the export is `-r_i / w_i` with `w_i = (M_Gamma 1)_i`. What comes back is
therefore the consistent-to-nodal conversion of the P1 boundary mass matrix
applied to g — the PDE, the conductivity, the mesh interior and the solver
never enter it. Its deviation from -g has the closed form

    Q_i + g(y_i) = -(h^2 / 6) g''(y_i) + O(h^4),

which is second order for ANY correct assembly of ANY equation. MEASURED here,
max over interior interface nodes at n = 8/16/32, by all four backends this
fixture runs:

    FEniCSx        1.9643385e-02  4.9833201e-03  1.2496293e-03
    scikit-fem     1.9643385e-02  4.9833201e-03  1.2496293e-03
    NGSolve        1.9643385e-02  4.9833201e-03  1.2496293e-03
    DUNE-fem       1.9643386e-02  4.9832892e-03  1.2496477e-03
    NumPy, no FEM  1.9643386e-02  4.9833201e-03  1.2494690e-03

The last row is a bare 1-D P1 mass matrix in twenty lines of NumPy with no FEM,
no PDE, no solver and no material in it. Four codes agreeing to seven
significant figures here is NOT a cross-code check; it is four evaluations of
the same algebra. The earlier version of this docstring read an "order 2.00"
off exactly these numbers and called it a convergence result. It is not one,
and the `min(orders) >= 1.8` assertion it carried could not fail for any
correct assembly.

WHY THE TEST IS KEPT ANYWAY — IT HAS REAL, NARROW VALUE. The identity above is
what the export SHOULD equal, and checking it against the exact discrete form
catches, decisively and cheaply, exactly the errors this interface mechanism is
prone to and that no field-error check sees:

  * a flipped sign convention (the export lands near +g, off by ~2 |g|);
  * a wrong or lumped interface weight w_i;
  * the wrong facets tagged, or `ds` taken over the whole boundary;
  * a blocked-dof mapping that divides a node's component by another node's
    weight (the vector participants' standing footgun);
  * a residual taken against a load that already contains the interface term,
    which zeroes the free rows.

So the assertion below is against the CLOSED FORM, not against an order. An
order assertion here is unfalsifiable; this one is not.

The pair test next door cannot see any of this: its placeholder problem has a
zero source and equal outer temperatures, so the interface flux is ONE CONSTANT
(18.46 along the whole edge) and a P1 gradient is exact on a linear field.
Every recovery, right or wrong, agrees to machine precision there.

THE RETIRED PROJECTED-GRADIENT RECOVERY, STATED HONESTLY. The measurement that
retired it was reported here as "order 0.00, it never converges", which was
norm-shopping: the same measurement recorded 0.93 away from the ends and 0.50
in rms. What it actually showed is that the L2-projected boundary gradient is
order ~1 in the interior AWAY FROM THE ENDS, and non-convergent in the max norm
that includes the nodes near the interface ends, where its error stalls at 2.6
against a true flux of size 2 to 5. Order ~1 is the expected accuracy of a P1
gradient evaluated ON a boundary and is the honest reason to prefer the
second-order reaction; "it never converges" is true only of one norm, and that
norm was not stated. Those figures are from the original 8/16/32/64/128 sweep
and have NOT been re-measured since the retired branch was deleted.

The end nodes are reported apart from the interior throughout, because an
interface node that also sits on the outer Dirichlet boundary carries the outer
reaction too and is physically a different quantity.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
PART_DIR = REPO / "data" / "coupling_participants"

Y0, Y1 = 0.0, 0.4
X0, X1 = 0.6, 1.1
IFACE_X = 0.6                      # the Neumann side's LEFT edge


def q_partner(y):
    """What the partner claims to be exporting, in ITS outward normal."""
    return 2.0 + 3.0 * np.sin(4.0 * np.asarray(y, float))


# A non-zero volume source, so that subtracting the WRONG vector is visible:
# with f == 0 the volume load is zero and a residual taken against the full
# load would still be near zero for the wrong reason.
SRC_LINE = "return np.zeros_like(x)"
SRC_REPL = "return -K * (6.0 * x * y**2 + 2.0 * x**3)"

BASE_EDITS = {
    "SIDE": '"neumann"', "PARTNER": '"left"',
    "X0, X1": f"{X0}, {X1}", "Y0, Y1": f"{Y0}, {Y1}",
    "IFACE_X": str(IFACE_X), "K": "1.5", "T_OUTER": "300.0",
    "T_INIT": "310.0", "Q_INIT": "0.0",
}


def _edit(text: str, edits: dict) -> str:
    for k, v in edits.items():
        pat = re.compile(rf"^({re.escape(k)}\s*=\s*)(.*?)(\s*(?:#.*)?)$", re.M)
        assert pat.search(text), f"edit key {k!r} not found in the shipped script"
        text = pat.sub(lambda m: m.group(1) + v + m.group(3), text, count=1)
    return text


def _interpreter(backend: str):
    if backend == "fenics":
        from backends.fenics.backend import _find_fenics_python
        p = _find_fenics_python()
        return str(p) if p else None
    if backend == "ngsolve":
        # See test_coupling_participants_run._interpreter: NGSolve is no
        # longer required to live in openPASO's own venv.
        from backends.ngsolve.backend import _find_ngsolve_python
        found = _find_ngsolve_python()
        return str(found) if found else None
    if backend == "kratos":
        from backends.kratos.backend import _find_kratos_python
        return _find_kratos_python()
    if backend == "skfem":
        return sys.executable
    if backend == "dune":
        from backends.dune.backend import _find_dune_python
        return _find_dune_python()
    return None


def _available(name: str) -> bool:
    from core.registry import get_backend, load_all_backends
    load_all_backends()
    b = get_backend(name)
    return bool(b) and b.check_availability()[0].value == "available"


def _run_once(tmp: Path, backend: str, n: int):
    """One solve at mesh resolution n; returns (y, q_exported) or None."""
    src = PART_DIR / f"participant_{backend}.py"
    if not src.is_file():
        pytest.skip(f"no shipped participant for {backend}")
    text = _edit(src.read_text(), {**BASE_EDITS, "NX, NY": f"{n}, {n}"})
    assert SRC_LINE in text, f"{backend}: F_SRC placeholder not found"
    text = text.replace(SRC_LINE, SRC_REPL)

    wd = tmp / f"{backend}_{n}"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "participant.py").write_text(text)

    # A partner export dense enough that the participant's own interpolation of
    # it is far below the discretisation error being measured.
    ys = np.linspace(Y0, Y1, 2001)
    (wd / "imports.json").write_text(json.dumps({
        "left": {"field_name": "temperature",
                 "n_points": len(ys),
                 "coordinates": [[IFACE_X, float(y)] for y in ys],
                 "values": [310.0] * len(ys),
                 "normal_fluxes": [float(v) for v in q_partner(ys)]}}))

    r = subprocess.run([_interpreter(backend), "participant.py"], cwd=str(wd),
                       capture_output=True, text=True, timeout=900)
    ep = wd / "exports.json"
    assert ep.is_file(), (f"{backend} n={n} wrote no exports.json, "
                          f"rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    d = json.loads(ep.read_text())
    assert "normal_fluxes" in d, f"{backend}: exported no normal_fluxes"
    y = np.array([c[1] for c in d["coordinates"]], float)
    q = np.array(d["normal_fluxes"], float)
    o = np.argsort(y)
    return y[o], q[o]


BACKENDS = ["fenics", "skfem", "ngsolve", "dune"]


def _consistent_to_nodal(y, g):
    """(M_Gamma g)_i / (M_Gamma 1)_i on the 1-D P1 grid the interface nodes
    define. This is the EXACT value the Neumann-side recovery must return, up
    to solver roundoff — no Taylor truncation, and no assumption that the nodes
    are equally spaced."""
    n = len(y)
    mg = np.zeros(n)
    w = np.zeros(n)
    for e in range(n - 1):
        h = y[e + 1] - y[e]
        mg[e] += h * (2.0 * g[e] + g[e + 1]) / 6.0
        mg[e + 1] += h * (g[e] + 2.0 * g[e + 1]) / 6.0
        w[e] += h / 2.0
        w[e + 1] += h / 2.0
    return mg / w


@pytest.mark.parametrize("backend", BACKENDS)
def test_neumann_export_is_the_consistent_nodal_form_of_the_applied_load(
        tmp_path, backend):
    """The export must equal -(M_Gamma g)/(M_Gamma 1) on the free rows.

    NOT AN ORDER CHECK — see the module docstring. On the Neumann side those
    rows carry the applied interface load identically, so this asserts the
    identity itself, which is falsifiable: a flipped sign, a wrong weight, a
    mistagged facet set or a blocked-dof mix-up all break it by O(1), while the
    order it used to assert could not fail for any correct assembly.

    Tolerance. Measured deviations at n = 8/16/32 on this host: FEniCSx
    3.1e-12 / 1.7e-11 / 1.6e-07, scikit-fem 3.9e-12 / 1.3e-11 / 1.6e-07,
    NGSolve 2.6e-12 / 7.0e-12 / 1.6e-07, DUNE-fem 7.9e-09 / 5.2e-08 / 1.9e-07.
    The ~1.6e-07 at n = 32 is the participant's own np.interp of the
    2001-sample partner export, not the recovery. 1e-3 leaves four orders of
    margin above the worst of those and stays twenty times below the identity's
    own O(h^2) offset from -g on the coarsest mesh, so the check is still sharp
    at every level. A flipped sign lands at ~10.0 — measured, by mutating this
    assertion's `pred`.
    """
    if not _available(backend):
        pytest.skip(f"{backend} not available on this install")
    if not _interpreter(backend):
        pytest.skip(f"no interpreter resolved for {backend}")

    ns = (8, 16, 32)
    errs, devs = [], []
    for n in ns:
        y, q = _run_once(tmp_path, backend, n)
        g = q_partner(y)
        # Drop the two end nodes: they carry the outer reaction as well.
        inner = (y > Y0 + 1e-12) & (y < Y1 - 1e-12)
        assert inner.sum() >= 3, f"{backend}: interface too small to judge"
        pred = -_consistent_to_nodal(y, g)
        errs.append(float(np.max(np.abs(q[inner] + g[inner]))))
        devs.append(float(np.max(np.abs(q[inner] - pred[inner]))))

    assert all(math.isfinite(d) for d in devs), f"{backend}: {devs}"
    assert max(devs) <= 1e-3, (
        f"{backend}: the Neumann export is NOT the consistent-to-nodal form of "
        f"the load it applied — deviations {devs} at n={ns}. On free interface "
        f"rows r = A u - b_vol IS M_Gamma g, so this is an assembly, sign, "
        f"weight or dof-mapping defect, not a discretisation error. "
        f"(Offsets from -g were {errs}; the closed form says "
        f"-(h^2/6) g''(y), i.e. 8 h^2 sin(4y) for this g.)")

    # Reported, never asserted: this "order" is entailed by the identity above
    # and is 2.00 for any correct assembly. The real order lives in
    # tests/test_interface_flux_converges_to_a_known_exact_flux.py.
    print(f"{backend}: offsets from -g {errs}; identity deviations {devs}")


def test_kratos_3d_measures_what_its_conditions_assembled(tmp_path):
    """The 3-D Kratos participant, whose Neumann side has no reaction to read.

    Kratos stores REACTION_FLUX only on FIXED dofs, so the Dirichlet branch's
    route is closed on the Neumann side. Echoing the imported array would be
    algebraically exact and useless as evidence — applied on the wrong facets
    or with the wrong sign it would read the same, and the two-sided balance
    check would still report roundoff. Summing each condition's own
    right-hand side is a MEASUREMENT of what entered the linear system.

    SAME CAVEAT AS THE 2-D FIXTURE ABOVE: this is an assembly check, not a
    convergence study. What the summed condition right-hand sides carry IS the
    applied interface load, so the figures below say the conditions were built
    and integrated on the right facets with the right sign — they are not
    evidence that the recovery converges to a true interface flux. That
    evidence is in tests/test_interface_flux_converges_to_a_known_exact_flux.py
    for the reaction route; there is no 3-D equivalent, and none is claimed.
    The numbers below are from the original measurement. The assertion in this
    test re-checks only that the last order stays >= 1.5; it does not re-derive
    the three error values, so treat them as recorded rather than reproduced.

    Measured against q(y,z) = 2 + 3 sin(4y) cos(3z), interior nodes of the
    interface plane (the rim carries the outer reaction too):
        assembled conditions   5.16e-01, 1.77e-01, 4.77e-02  order 1.55, 1.89
        gradient averaging     1.27e+00, 8.37e-01, 3.95e-01  order 0.60, 1.08
    """
    if not _available("kratos"):
        pytest.skip("kratos not available on this install")
    import numpy as _np
    src = PART_DIR / "participant_kratos_3d.py"
    if not src.is_file():
        pytest.skip("no 3-D Kratos participant")

    def q_ex(y, z):
        return 2.0 + 3.0 * _np.sin(4.0 * y) * _np.cos(3.0 * z)

    errs = []
    for n in (4, 8, 16):
        wd = tmp_path / f"k{n}"
        wd.mkdir(parents=True, exist_ok=True)
        (wd / "p.py").write_text(_edit(src.read_text(), {
            "SIDE": '"neumann"', "PARTNER": '"left"', "IFACE_AXIS": "0",
            "IFACE_POS": "0.5", "X0, X1": "0.5, 1.0", "Y0, Y1": "0.0, 1.0",
            "Z0, Z1": "0.0, 1.0", "NX, NY, NZ": f"{n}, {n}, {n}",
            "K": "1.5", "DIRICHLET_FACES": '("x1",)', "LIN_SOLVER": '"direct"'}))
        m = 41
        ys, zs = _np.meshgrid(_np.linspace(0, 1, m), _np.linspace(0, 1, m),
                              indexing="ij")
        ys, zs = ys.ravel(), zs.ravel()
        (wd / "imports.json").write_text(json.dumps({"left": {
            "field_name": "temperature", "n_points": len(ys),
            "coordinates": [[0.5, float(a), float(b)] for a, b in zip(ys, zs)],
            "values": [300.0] * len(ys),
            "normal_fluxes": [float(v) for v in q_ex(ys, zs)]}}))
        # NOT sys.executable: Kratos lives in its own environment and the
        # backend now finds it there, so a participant launched with the test
        # runner's Python fails on ModuleNotFoundError for a reason that has
        # nothing to do with the flux recovery under test.
        kratos_py = _interpreter("kratos") or sys.executable
        r = subprocess.run([str(kratos_py), "p.py"], cwd=str(wd),
                           capture_output=True, text=True, timeout=2400)
        ep = wd / "exports.json"
        assert ep.is_file(), (f"kratos_3d n={n} wrote no exports.json, "
                              f"rc={r.returncode}\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        d = json.loads(ep.read_text())
        c = _np.array(d["coordinates"], float)
        q = _np.array(d["normal_fluxes"], float)
        yy, zz = c[:, 1], c[:, 2]
        rim = (_np.isclose(yy, 0.0) | _np.isclose(yy, 1.0) |
               _np.isclose(zz, 0.0) | _np.isclose(zz, 1.0))
        errs.append(float(_np.max(_np.abs(q[~rim] + q_ex(yy, zz)[~rim]))))

    orders = [math.log(errs[i - 1] / errs[i]) / math.log(2.0)
              for i in range(1, len(errs))]
    assert orders[-1] >= 1.5, (
        f"kratos_3d interface flux converges at {orders} (errors {errs}); "
        f"the gradient averaging this replaced gives ~1.0 and the assembled "
        f"condition measurement gives ~1.9")


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_participant_projects_the_boundary_gradient(backend):
    """The retired recovery must not come back by copy-paste.

    Cheap, and it covers the participants this host cannot execute.
    """
    hits = [p.name for p in PART_DIR.glob(f"participant_{backend}*.py")
            if "MUST NOT be used here" in p.read_text()]
    assert not hits, (
        f"{hits} still carry the retired 'the reaction formula MUST NOT be "
        f"used here' branch. It is true only when the residual is taken "
        f"against a load that already contains the interface term; against "
        f"the volume load alone those rows ARE the interface functional.")


def test_served_guidance_does_not_recommend_the_retired_recovery():
    """What we TELL agents must match what we SHIP them.

    The participants were fixed while the served FEniCSx guidance still read
    "The exported flux is an L2 projection of -K*S*grad(T)[0] onto the same CG1
    space" — a recommendation to do the thing every shipped participant had
    just stopped doing. Served text that contradicts the shipped code is the
    same defect class as a path we promise and do not have: the agent reads it
    as an instruction and spends its budget on it.
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO / "src"))
    from tools.coupling_knowledge import coupling_knowledge

    for solver in ("", "fenics", "skfem", "ngsolve", "dune", "kratos", "febio"):
        text = coupling_knowledge(solver)
        low = text.lower()
        # The phrasing that told agents to EXPORT a projection.
        for bad in ("exported flux is an l2 projection",
                    "export the l2 projection",
                    "exported traction is an l2 projection"):
            assert bad not in low, (
                f"solver={solver!r}: served guidance still recommends the "
                f"retired recovery ({bad!r}). Measured, it is order ~1 in the "
                f"interior away from the interface ends and does not converge "
                f"at all in the max norm that includes the near-end nodes.")
