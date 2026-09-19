"""The check answered on a grid whose quadrature it does not have, and blessed a wrong field.

The identity is evaluated with a MIDPOINT rule: every point carries the same
cell volume, which is exact to second order when the points are the centres of
a uniform tiling of the box. `_detect_midpoint_grid` checked that the points
form a uniform tensor grid and never checked that they are the CENTRES of the
box it was handed -- it was not handed the box at all. A vertex grid (the
corners of the same tiling, which is what a participant's own mesh dump is)
passed as a midpoint grid, and then the outermost row carries a full cell of
weight where it should carry half.

MEASURED on the two coupled runs that graded correct this week, over side A's
own mesh dumps (54, 187 and 693 vertices):

    field as delivered (correct)    0.370  0.200  0.104   CONSISTENT
    field scaled by 1.20 (wrong)    0.245  0.040  0.076   CONSISTENT

The 20 % error reads BETTER than the correct field, and both pass. The
quadrature error at the boundary is 10-37 % here, far larger than the defect
being looked for, so the verdict was noise wearing a verdict's clothes. On the
same runs' probe-grid deliverables -- true cell midpoints -- the same check
reads 1.79e-02 -> 1.63e-03 for the correct field and a flat 1.9e-01 for the
same 20 % error.

So the arrangement has to be checked against the box: refuse, and say which
arrangement was found, rather than answer from a rule that does not apply. This
matters now because the verdict is about to be computed automatically from the
agent's own files, and the file it must never read is exactly this one.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.pde_consistency import _detect_midpoint_grid, check_levels   # noqa: E402

BOX = [(0.0, 1.0), (0.0, 0.625)]
SOURCE = f"{math.pi ** 2} * sin(pi*x) * (y + 0.3)"


def _u(x, y):
    return math.sin(math.pi * x) * (y + 0.3)


def _midpoints(n):
    hx = (BOX[0][1] - BOX[0][0]) / n
    hy = (BOX[1][1] - BOX[1][0]) / n
    return [(BOX[0][0] + (i + 0.5) * hx, BOX[1][0] + (j + 0.5) * hy)
            for i in range(n) for j in range(n)]


def _vertices(n):
    hx = (BOX[0][1] - BOX[0][0]) / n
    hy = (BOX[1][1] - BOX[1][0]) / n
    return [(BOX[0][0] + i * hx, BOX[1][0] + j * hy)
            for i in range(n + 1) for j in range(n + 1)]


def _levels(points_of, scale=1.0):
    return {lvl: [(x, y, scale * _u(x, y)) for x, y in points_of(n)]
            for lvl, n in ((1, 8), (2, 16), (3, 32))}


def test_a_midpoint_grid_of_the_box_is_accepted():
    w, why = _detect_midpoint_grid(_midpoints(8), BOX)
    assert w is not None, why
    assert abs(w - (1.0 / 8) * (0.625 / 8)) < 1e-12


def test_a_vertex_grid_is_refused_and_named():
    w, why = _detect_midpoint_grid(_vertices(8), BOX)
    assert w is None, "a vertex grid still passes as a grid of cell midpoints"
    assert "midpoint" in why.lower() or "centre" in why.lower(), why


def test_the_refusal_reaches_the_verdict_instead_of_a_number():
    res = check_levels(_levels(_vertices), SOURCE, 1.0, BOX).as_dict()
    assert res["verdict"] in ("NOT_APPLICABLE", "REFUSED"), res
    blob = str(res)
    assert "midpoint" in blob.lower() or "centre" in blob.lower(), blob[:400]


def test_the_midpoint_path_is_unchanged():
    good = check_levels(_levels(_midpoints), SOURCE, 1.0, BOX).as_dict()
    assert good["verdict"] == "CONSISTENT", good
    bad = check_levels(_levels(_midpoints, scale=1.2), SOURCE, 1.0, BOX).as_dict()
    assert bad["verdict"] == "INCONSISTENT", bad


def test_the_box_stays_optional_for_callers_that_have_none():
    """Nothing that cannot supply a box loses its answer."""
    w, why = _detect_midpoint_grid(_midpoints(8))
    assert w is not None, why
