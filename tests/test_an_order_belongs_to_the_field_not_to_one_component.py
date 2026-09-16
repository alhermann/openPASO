"""A convergence order is a property of a field, not of one Cartesian column.

The interface exclusion in the order check already learned half of this: an
interface node with a one-sided boundary weight converges at order 1 while the
interior runs at 2, and judging it cost a false alarm on the only coupled run
then verified correct.

The other half is the minor component of a vector. C9 seed 8791 is the first
VECTOR cell ever graded CORRECT -- order 1.927, errors 2.95e-08 -> 7.83e-09 ->
2.04e-09. Its side-A ux self-difference improves at 1.92; its uy, seven times
smaller and sitting near the coupling iteration's own floor, improves at 0.96.
The check fired on that one column and told a correct agent that its exchanged
datum was O(h) and its coupling first-order. The vector built from both columns
improves at 1.88 -- what the grader measures, and what is true.

Measured over every graded cell with an order (28 CORRECT, 14 unphysical, 8
confidently wrong): grouping moves exactly one verdict, 8791 from firing to
silent, and keeps all 7 catches on wrong cells. Every scalar problem has one
component per side, so its view is unchanged -- which is why 26 scalar CORRECT
cells never showed this.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.result_audit import _vector_order_view, audit  # noqa: E402


def _grid(n=8):
    return [(i / float(n), j / float(n)) for i in range(n + 1)
            for j in range(n + 1)]


def _cell(tmp_path, per_level, fields=("ux", "uy")):
    """Three levels whose per-side field values are given per level.

    per_level[k][side][field] is the constant offset written at level k, so the
    self-difference between levels is exactly the difference of the offsets.
    """
    pts = _grid()
    for k in (1, 2, 3):
        for side in ("A", "B"):
            vals = per_level[k][side]
            hdr = "x,y," + ",".join(fields)
            body = "".join(
                f"{x},{y}," + ",".join(f"{vals[f]:.17e}" for f in fields) + "\n"
                for x, y in pts)
            (tmp_path / f"solution_level{k}_{side}.csv").write_text(
                hdr + "\n" + body)
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "y,ux,uy\n0.0,1.0,2.0\n0.5,1.5,2.5\n")
    return tmp_path


def _ladder(major, minor):
    """Offsets whose successive differences follow the given two drops."""
    a0, a1, a2 = 0.0, major[0], major[0] + major[1]
    b0, b1, b2 = 0.0, minor[0], minor[0] + minor[1]
    return {k: {s: {"ux": a, "uy": b} for s in ("A", "B")}
            for k, a, b in ((1, a0, b0), (2, a1, b1), (3, a2, b2))}


def _orders(work):
    return [f for f in audit(str(work)).get("findings", [])
            if "IMPROVE AT ONLY" in str(f.get("finding", ""))]


def test_a_second_order_field_is_not_called_first_order_by_its_small_column(
        tmp_path):
    # 8791's own numbers: ux drops 1.101e-08 -> 2.908e-09 (order 1.92), uy
    # drops 1.665e-09 -> 8.584e-10 (order 0.96). The vector runs at 1.88.
    _cell(tmp_path, _ladder((1.101e-08, 2.908e-09), (1.665e-09, 8.584e-10)))
    assert _orders(tmp_path) == [], (
        "the order check judged one column of a vector and called a "
        "second-order field first-order")


def test_a_genuinely_first_order_vector_is_still_named(tmp_path):
    # 8792's own numbers: every column near 1, and the vector too.
    _cell(tmp_path, _ladder((5.158e-08, 2.531e-08), (6.398e-09, 2.875e-09)))
    assert _orders(tmp_path), (
        "a vector field whose every component improves at ~1 must still be "
        "named -- that is the catch this check exists for")


def test_a_scalar_beside_a_vector_is_judged_on_its_own(tmp_path):
    # 43 solution files here ship `T, ux, uy`. A temperature of order 300 and a
    # displacement of order 1e-9 must never be folded into one norm, or the
    # displacement can never be seen again.
    pts = _grid()
    off = {1: (0.0, 0.0), 2: (3.0e+02, 1.101e-08),
           3: (3.0e+02 + 1.5e+02, 1.101e-08 + 2.908e-09)}
    for k in (1, 2, 3):
        t, u = off[k]
        for side in ("A", "B"):
            (tmp_path / f"solution_level{k}_{side}.csv").write_text(
                "x,y,T,ux,uy\n" + "".join(
                    f"{x},{y},{t:.17e},{u:.17e},{u:.17e}\n" for x, y in pts))
            (tmp_path / f"interface_level{k}_{side}.csv").write_text(
                "y,ux,uy\n0.0,1.0,2.0\n0.5,1.5,2.5\n")
    view = _vector_order_view(
        __import__("tools.result_audit", fromlist=["x"])
        ._sequences_from_level_csvs(tmp_path))
    labels = {k for k in view if k.startswith("selfdiff_solution_A")}
    assert "selfdiff_solution_A_T" in labels, (
        "a field with no axis suffix must keep its own sequence")
    assert "selfdiff_solution_A_u|vector" in labels, (
        "ux and uy must be judged as one vector")
    assert "selfdiff_solution_A_ux" not in labels, (
        "a component folded into a vector must not also be judged alone")


def test_a_scalar_problem_is_left_exactly_as_it_was(tmp_path):
    # One field per side is not a vector, and 26 scalar CORRECT cells depend on
    # this path being untouched.
    pts = _grid()
    for k, off in ((1, 0.0), (2, 1.0e-06), (3, 1.25e-06)):
        for side in ("A", "B"):
            (tmp_path / f"solution_level{k}_{side}.csv").write_text(
                "x,y,u\n" + "".join(
                    f"{x},{y},{off:.17e}\n" for x, y in pts))
    from tools.result_audit import _sequences_from_level_csvs
    seqs = _sequences_from_level_csvs(tmp_path)
    assert _vector_order_view(seqs) == seqs, (
        "a scalar workspace must pass through the grouping unchanged")


def test_one_axis_alone_is_not_a_vector(tmp_path):
    # A lone `ux` has no sibling to pair with; folding it would rename a
    # sequence other checks report by name for no gain.
    pts = _grid()
    for k, off in ((1, 0.0), (2, 1.0e-06), (3, 1.25e-06)):
        for side in ("A",):
            (tmp_path / f"solution_level{k}_{side}.csv").write_text(
                "x,y,ux\n" + "".join(
                    f"{x},{y},{off:.17e}\n" for x, y in pts))
    from tools.result_audit import _sequences_from_level_csvs
    seqs = _sequences_from_level_csvs(tmp_path)
    assert _vector_order_view(seqs) == seqs, (
        "a single axis column is not a vector field")


# ---------------------------------------------------------------------------
# The same instrument in the per-level path.
#
# field_quality_due() is what reaches the agent DURING couple(), level by
# level, and it judges the same sequences column by column. On 8791 it
# reported FLOOR on `selfdiff_interface_A_uy` -- 2.109e-09 -> 2.033e-09, 3.6%
# -- and told a correct agent that refinement was changing nothing at its
# seam. Side A's interface displacement VECTOR moves 59.3% per level
# (1.465e-08 -> 5.965e-09). Measured over the same 50 graded cells: CORRECT
# goes from 1 firing to 0, and the wrong cells that fire stay at 6 unphysical
# and 2 confidently wrong, the same cells by name.
# ---------------------------------------------------------------------------

from tools.result_audit import field_quality_due  # noqa: E402


def _interface(tmp_path, k, side, ux, uy):
    pts = [(0.25 + i / 40.0, 0.625) for i in range(12)]
    (tmp_path / f"interface_level{k}_{side}.csv").write_text(
        "x,y,ux,uy\n" + "".join(
            f"{x:.11e},{y:.11e},{ux:.17e},{uy:.17e}\n" for x, y in pts))


def _floors(work):
    return [f for f in field_quality_due(work, [1, 2, 3])
            if "FLOOR" in str(f.get("finding", ""))]


def test_a_moving_seam_is_not_called_a_floor_by_its_small_column(tmp_path):
    # 8791's own interface numbers: ux 1.45e-08 -> 5.608e-09 (61.3% per level),
    # uy 2.109e-09 -> 2.033e-09 (3.6%). The vector moves 59.3%.
    ux = [0.0, 1.45e-08, 1.45e-08 + 5.608e-09]
    uy = [0.0, 2.109e-09, 2.109e-09 + 2.033e-09]
    for k in (1, 2, 3):
        for side in ("A", "B"):
            _interface(tmp_path, k, side, ux[k - 1], uy[k - 1])
    assert _floors(tmp_path) == [], (
        "the per-level path called a seam moving 59% per level a floor, "
        "because one of its two columns was flat")


def test_a_seam_that_really_has_stopped_moving_is_still_named(tmp_path):
    # Both columns flat, so the vector is flat: this is the catch the FLOOR
    # check exists for and it must survive the grouping.
    ux = [1.0e-06, 1.01e-06, 1.02e-06]
    uy = [2.0e-07, 2.02e-07, 2.04e-07]
    for k in (1, 2, 3):
        for side in ("A", "B"):
            _interface(tmp_path, k, side, ux[k - 1], uy[k - 1])
    assert _floors(tmp_path), (
        "a seam whose whole vector stops moving must still be named")
