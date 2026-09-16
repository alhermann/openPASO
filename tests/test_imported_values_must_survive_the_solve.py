"""The gate that names the import which never reached the answer.

This is the defect that still converges: a Dirichlet side that loads the
partner's trace into its solution vector and then loses it is a fixed point,
so the interface residual collapses and every downstream check reports
agreement on a field the partner never influenced.

The two properties that make it shippable are measured here: it is silent on
every served contract, and it fires on the recorded shapes.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.participant_lint import imported_values_not_held  # noqa: E402

SERVED = sorted((ROOT / "data" / "coupling_participants").glob("*.py"))


@pytest.mark.parametrize("path", SERVED, ids=lambda p: p.name)
def test_the_gate_is_silent_on_every_served_contract(path):
    """A gate that names an absent defect costs an action for nothing."""
    assert imported_values_not_held(path.read_text()) == "", (
        f"{path.name} is served and must not be flagged by its own lint")


def _ngsolve_head(dirichlet="'outer|interface'"):
    return (
        "import ngsolve\n"
        "from ngsolve import H1, BilinearForm, LinearForm, GridFunction, Mesh, dx, grad, ds\n"
        "mesh = Mesh(ngmesh)\n"
        f"fes = H1(mesh, order=1, dirichlet={dirichlet})\n"
        "u, v = fes.TnT()\n"
        "a = BilinearForm(fes)\n"
        "a += grad(u) * grad(v) * dx\n"
        "a.Assemble()\n"
        "f_vol = LinearForm(fes)\n"
        "f_vol.Assemble()\n"
        "gfu = GridFunction(fes)\n"
        "for d, t in zip(iface_dofs, T_if):\n"
        "    gfu.vec[int(d)] = float(t)\n"
    )


def test_a_solve_that_assigns_over_the_loaded_entries_is_named():
    src = _ngsolve_head() + "gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f_vol.vec\n"
    msg = imported_values_not_held(src)
    assert "replaces the whole vector" in msg
    assert "still converges" in msg, "the message must say why the residual will not warn you"


def test_the_two_line_inverse_spelling_is_named_as_well():
    """`inv = A.Inverse(...)` then `u.vec.data = inv * f` is the common shape."""
    src = (_ngsolve_head()
           + "inv = a.mat.Inverse(fes.FreeDofs(), inverse='sparsecholesky')\n"
             "gfu.vec.data = inv * f_vol.vec\n")
    assert "replaces the whole vector" in imported_values_not_held(src)


def test_an_update_onto_the_loaded_entries_is_not_flagged():
    """The served idiom: solve the residual on the free dofs and ADD it."""
    src = (_ngsolve_head()
           + "r = f_vol.vec.CreateVector()\n"
             "r.data = f_vol.vec - a.mat * gfu.vec\n"
             "gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * r\n")
    assert imported_values_not_held(src) == ""


def test_assembling_without_ever_solving_is_named():
    src = _ngsolve_head()          # no solve at all
    msg = imported_values_not_held(src)
    assert "never solves" in msg


def test_interface_entries_left_out_of_the_essential_set_are_named():
    """Held only by naming the boundary or by masking the BitArray; neither here."""
    src = (_ngsolve_head(dirichlet="'outer'")
           + "r = f_vol.vec.CreateVector()\n"
             "r.data = f_vol.vec - a.mat * gfu.vec\n"
             "gfu.vec.data += a.mat.Inverse(fes.FreeDofs()) * r\n")
    assert "nothing holds those entries fixed" in imported_values_not_held(src)


def test_a_hand_masked_bitarray_counts_as_holding_them():
    """The recorded CORRECT cell names no boundary and masks the dofs itself."""
    src = (_ngsolve_head(dirichlet="None")
           + "free_dofs = fes.FreeDofs()\n"
             "for d in iface_dofs:\n"
             "    free_dofs[d] = False\n"
             "u_free = a.mat.Inverse(free_dofs) * r\n"
             "gfu.vec.data += u_free\n")
    assert imported_values_not_held(src) == ""


def test_a_volume_coupled_import_through_the_load_vector_is_not_this_defect():
    """The import enters as a source field, so assigning the solution is correct."""
    src = (
        "import ngsolve\n"
        "from ngsolve import H1, BilinearForm, LinearForm, GridFunction, Mesh, dx, grad\n"
        "fes = H1(mesh, order=1, dirichlet='left|right')\n"
        "gth = GridFunction(fes)\n"
        "for i, d in enumerate(vdof):\n"
        "    gth.vec[int(d)] = float(theta_nodal[i])\n"
        "a = BilinearForm(fes)\n"
        "a.Assemble()\n"
        "gfu = GridFunction(fes)\n"
        "gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec\n")
    assert imported_values_not_held(src) == ""


def test_a_script_that_is_not_ngsolve_is_left_alone():
    assert imported_values_not_held("import dolfinx\nu.vector[:] = 0\n") == ""
