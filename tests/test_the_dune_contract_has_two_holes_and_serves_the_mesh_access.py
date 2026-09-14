"""The DUNE-fem participant contract serves everything a weak model was measured
to get wrong AROUND the solve, and elides exactly the solve.

Six worker trials (2026-09-10, 27B) failed 0/3 each while the hole demanded a
simplex grid, triangle connectivity and vertex ordering; the first passes came
once the scaffold had TWO holes (grid+space | form+source+BCs+solve) with the
handshake-onto-dofs and the vertex-ordered mesh access served between them,
validated by a manufactured solution (flux error 2e-2 at h = 0.1). This pins
that shape so an edit cannot quietly widen the hole again.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _contract() -> str:
    from tools import consolidated as C
    return C._coupling_participant_script("dune")


def test_two_holes_and_the_names_each_must_leave():
    t = _contract()
    assert t.count("openPASO DOES NOT SERVE THIS") == 4          # two holes, each opened and closed
    assert "HOLE 1 OF 2" in t and "HOLE 2 OF 2" in t
    # the lean view thins comment runs, so pin the names in the leave-behind block
    # that survives it, not a particular explanatory line
    lb = t[t.index("WHAT YOUR TWO HOLES MUST LEAVE BEHIND"):]
    for name in ("gridView", "space", "x (SpatialCoordinate(space))", "uh", "F_SRC"):
        assert name in lb, name


def test_the_served_part_carries_the_handshake_onto_dofs_and_the_mesh_access():
    t = _contract()
    for served in ("gtrace = space.interpolate(0", "on_iface = ", "for _v in gridView.vertices",
                   "_idx.subIndex(_e, _i, 2)", "len(_e.geometry.corners)", "u_vert = np.array(uh.as_numpy",
                   "interior = _if_nodes[1:-1]", "EXPORT SELF-CHECK",
                   "K_UFL = _Constant(KV", "C_UFL = _Constant(CV"):
        assert served in t, served


def test_the_solve_itself_is_not_served():
    t = _contract()
    code = "\n".join(l for l in t.splitlines() if l.strip() and not l.strip().startswith("#"))
    for solve_only in ("galerkin(", "scheme.solve(", "aluConformGrid(", "TrialFunction(space)", "structuredGrid("):
        assert solve_only not in code, solve_only
