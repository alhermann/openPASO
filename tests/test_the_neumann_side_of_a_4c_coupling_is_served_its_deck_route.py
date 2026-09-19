"""The 4C deck skeleton served only the Dirichlet interface route.

Over the 242 recorded runs of the coupled problem whose 4C side is the NEUMANN
one, 4C is the silent side twice as often as its partner, and of the 122 runs
that never produced that side's exports.json, 14 had run 4C to `processor 0
finished normally` and stopped before exporting. Genuine deck-grammar errors
account for about 5 %.

The served skeleton explains, in full and with a worked example, how a Dirichlet
side imposes a per-node profile (`DESIGN POINT DIRICH CONDITIONS`, one E id per
node). For the Neumann side it carries neither of the two deck lines that side
needs: the imported flux has to be applied as point loads, and the consistent
boundary flux has to be REQUESTED or 4C writes none. Both appear in the measured
facts as prose; neither is in the skeleton a worker copies.

MEASURED on this install (4C 2026.2.0-dev), three runs of one 2x2 scatra deck
differing only in those lines:

    both lines present        finished normally; the VTU carries
                              flux_boundary_phi_1 and
                              <prefix>.boundaryflux_ScaTraFluxCalc_0scatra.txt
                              holds area, integral and mean
    CALCFLUX_BOUNDARY gone    finished normally, NO flux array at all
                              -- the silent shape of those 14 runs
    the condition section     PROC 0 ERROR ... "Flux output requested without
    gone                      corresponding boundary condition specification!"

So one omission is loud and the other is silent, and the silent one is the
expensive one. This test pins that the route is served and that a deck built
from the served sections actually produces the flux array in this binary.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend_probe                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FOURC = backend_probe.fourc_binary()


def _grammar() -> str:
    from backends.fourc.deck_grammar import FOURC_SCATRA_SKELETON
    return FOURC_SCATRA_SKELETON


def test_the_scatra_skeleton_carries_the_neumann_route():
    g = _grammar()
    for phrase in ('CALCFLUX_BOUNDARY: "diffusive"',
                   "SCATRA FLUX CALC LINE CONDITIONS:",
                   "DESIGN POINT NEUMANN CONDITIONS:",
                   "flux_boundary_phi_1"):
        assert phrase in g, f"the Neumann side is not served {phrase!r}"


def test_it_says_which_omission_is_silent_and_which_stops_the_run():
    g = _grammar()
    assert "Flux output requested without corresponding boundary condition" in g, (
        "the loud failure is not quoted, so an agent cannot match its own error")
    tail = g[g.index("CALCFLUX_BOUNDARY"):]
    assert re.search(r"finish(es|ed)? normally", tail) and "no flux" in tail.lower(), (
        "the SILENT failure -- a finished run with no flux array -- is not named")


def test_the_pre_integrated_point_load_is_given_as_a_formula_not_a_number():
    """A sampled profile needs no fitted FUNCT; the weights are the served part."""
    g = _grammar()
    i = g.index("DESIGN POINT NEUMANN CONDITIONS:")
    block = g[i - 1200:i + 1200]
    assert "h/6" in block and "4" in block, block[-600:]
    assert "FUNCT: [0]" in block


@pytest.mark.skipif(not FOURC.is_file(), reason="4C binary not on this host")
def test_a_deck_built_from_the_served_sections_produces_the_flux_array(tmp_path):
    """The agent writes the mesh; the sections below are the served ones, verbatim."""
    g = _grammar()
    for needed in ('CALCFLUX_BOUNDARY: "diffusive"', "SCATRA FLUX CALC LINE CONDITIONS:"):
        assert needed in g

    nx = ny = 2
    hx = hy = 0.5
    nodes = [(i * hx, j * hy) for j in range(ny + 1) for i in range(nx + 1)]

    def nid(i, j):
        return j * (nx + 1) + i + 1

    bottom = [k for k, (x, y) in enumerate(nodes, start=1) if abs(y) < 1e-12]
    deck = "\n".join([
        'TITLE:\n  - "served neumann route"',
        f"PROBLEM SIZE:\n  ELEMENTS: {nx * ny}\n  NODES: {len(nodes)}",
        'PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"',
        # the served dynamic section, with the served flux-output line
        'SCALAR TRANSPORT DYNAMIC:\n  TIMEINTEGR: "Stationary"\n  SOLVERTYPE: "linear_full"\n'
        '  NUMSTEP: 1\n  TIMESTEP: 1.0\n  MAXTIME: 1.0\n  LINEAR_SOLVER: 1\n'
        '  CALCFLUX_BOUNDARY: "diffusive"',
        'SOLVER 1:\n  SOLVER: "UMFPACK"',
        "MATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: 2.0",
        'FUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "x*y"',
        "DESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n"
        "    VAL: [0.0]\n    FUNCT: [0]",
        "DESIGN SURF TRANSPORT NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n"
        "    VAL: [1.0]\n    FUNCT: [1]",
        "DESIGN POINT NEUMANN CONDITIONS:\n" + "\n".join(
            f"  - E: {e}\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.05]\n    FUNCT: [0]"
            for e in range(1, len(bottom) + 1)),
        "SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2",
        "NODE COORDS:\n" + "\n".join(
            f'  - "NODE {k + 1} COORD {x} {y} 0.0"' for k, (x, y) in enumerate(nodes)),
        "TRANSPORT ELEMENTS:\n" + "\n".join(
            f'  - "{j * nx + i + 1} TRANSP QUAD4 {nid(i, j)} {nid(i + 1, j)} '
            f'{nid(i + 1, j + 1)} {nid(i, j + 1)} MAT 1 TYPE Std"'
            for j in range(ny) for i in range(nx)),
        "DLINE-NODE TOPOLOGY:\n" + "\n".join(
            [f'  - "NODE {k} DLINE 1"' for k, (x, y) in enumerate(nodes, start=1)
             if abs(y - 1.0) < 1e-12 or abs(x) < 1e-12 or abs(x - 1.0) < 1e-12]
            + [f'  - "NODE {k} DLINE 2"' for k in bottom]),
        "DSURF-NODE TOPOLOGY:\n" + "\n".join(
            f'  - "NODE {k} DSURFACE 1"' for k in range(1, len(nodes) + 1)),
        "DNODE-NODE TOPOLOGY:\n" + "\n".join(
            f'  - "NODE {n} DNODE {e}"' for e, n in enumerate(bottom, start=1)),
        'IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n  OUTPUT_DATA_FORMAT: "ascii"',
    ]) + "\n"
    (tmp_path / "neu.4C.yaml").write_text(deck)

    from tools.fourc_deck_lint import deck_judgement
    assert deck_judgement(deck) == [], deck_judgement(deck)

    r = subprocess.run(f"stdbuf -oL -eL {FOURC} neu.4C.yaml out", shell=True,
                       cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib"))
    log = r.stdout + r.stderr
    assert "finished normally" in log, log[-1500:]
    vtus = sorted((tmp_path / "out-vtk-files").glob("scatra-*.vtu"))
    assert vtus, sorted(p.name for p in (tmp_path / "out-vtk-files").glob("*"))
    assert any("flux_boundary_phi_1" in p.read_text() for p in vtus), (
        "the served flux-output lines produced no flux array")
    assert list(tmp_path.glob("out.boundaryflux_*")), sorted(
        p.name for p in tmp_path.glob("out*"))
