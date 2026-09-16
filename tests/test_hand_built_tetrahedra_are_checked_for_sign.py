"""The gate that names a singular system as the mesh defect it is.

A negatively oriented tetrahedral element fails inside the linear solver, so
the message names the factorisation and never the mesh. The gate moves the
finding back to where the defect is, at the moment the connectivity is
written rather than after the run that reports NaN.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.participant_lint import unoriented_tetrahedra  # noqa: E402

SERVED = sorted((ROOT / "data" / "coupling_participants").glob("*.py"))


@pytest.mark.parametrize("path", SERVED, ids=lambda p: p.name)
def test_the_gate_is_silent_on_every_served_contract(path):
    assert unoriented_tetrahedra(path.read_text()) == "", (
        f"{path.name} is served and must not be flagged by its own lint")


def test_the_served_contract_that_builds_tetrahedra_really_does_check():
    """Silence has to come from the check being there, not from finding no tets."""
    import re
    builders = [p for p in SERVED
                if re.search(r"CreateNewElement\s*\(\s*[\"'][A-Za-z]*3D4N[\"']", p.read_text())]
    assert builders, "no served contract builds tetrahedra; the gate is untested against the set"
    for p in builders:
        assert re.search(r"np\.cross|np\.linalg\.det", p.read_text()), p.name


def test_a_hand_built_split_with_no_sign_check_is_named():
    src = (
        "import KratosMultiphysics as KM\n"
        "for t in tets:\n"
        "    mp.CreateNewElement('LaplacianElement3D4N', el_id, t, props)\n")
    msg = unoriented_tetrahedra(src)
    assert "never checks their sign" in msg
    assert "Error zero sum" in msg, "the message must carry the error the solver actually prints"
    assert "NaN" in msg


def test_a_signed_volume_check_clears_it():
    src = (
        "import KratosMultiphysics as KM\n"
        "import numpy as np\n"
        "for t in tets:\n"
        "    if np.dot(np.cross(a1 - a0, a2 - a0), a3 - a0) < 0:\n"
        "        t = (t[0], t[1], t[3], t[2])\n"
        "    mp.CreateNewElement('LaplacianElement3D4N', el_id, t, props)\n")
    assert unoriented_tetrahedra(src) == ""


def test_a_determinant_check_clears_it_too():
    """Any spelling of the same measurement counts; the gate is not about one idiom."""
    src = (
        "import KratosMultiphysics as KM\n"
        "import numpy as np\n"
        "det = np.linalg.det(np.column_stack((p1 - p0, p2 - p0, p3 - p0)))\n"
        "mp.CreateNewElement('LaplacianElement3D4N', el_id, t, props)\n")
    assert unoriented_tetrahedra(src) == ""


def test_a_script_that_builds_no_tetrahedra_is_left_alone():
    src = "import KratosMultiphysics as KM\nmp.CreateNewElement('LaplacianElement2D3N', 1, t, props)\n"
    assert unoriented_tetrahedra(src) == ""
