"""The one check that separates right from converged-wrong was called once in 1118 runs.

`verify_pde_consistency` is the only thing in openPASO that can tell a field
converging cleanly to the RIGHT function from one converging just as cleanly to
a wrong one. A refinement study cannot: measured over 464 runs, result sets with
a complete level set self-converge at a median order of 1.96 to 1.99 whether or
not they solve the stated problem. The run's own mesh-independence verdict
cannot either -- it fires on half the correct runs.

MEASURED over every coupled cell on disk (1118 runs, openPASO arm): the tool was
called in ONE. `verify_interface_flux` in none, `audit_results` in 66. Two
wordings of the invitation were measured before this (2 calls against 14 asks,
then 0 against 7). The conclusion the sign check already drew applies here:
voluntary checking does not happen, so the check belongs in the audit that runs
on delivery, computed from files the agent has already written.

WHAT IT IS ALLOWED TO READ, and why this leaks nothing. The operator comes from
the side's OWN ./config.json -- the file the served contract already reads for
its level keys, carrying the box, the coefficient, the reaction and the source
the agent implemented. So the audit asks "does the field you delivered solve the
equation you say you solved", with no task file, no key and no reference
solution in it.

WHICH FILE IT MAY JUDGE. Only a per-level file whose points are the cell
MIDPOINTS of that side's box, found by shape and never by name. A participant's
own mesh dump is uniform but sits on the cell corners, and under a midpoint
quadrature its boundary row carries twice its share: measured on a run that
graded correct, judging its mesh dumps gave 0.370, 0.200, 0.104 for the
delivered field and 0.245, 0.040, 0.076 for the same field scaled by 1.20 --
the 20 % error scoring better, both "CONSISTENT". On the same run's probe-grid
deliverables the separation is clean, and that is the file this reads.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.result_audit import audit, equation_findings        # noqa: E402

BOX_A = {"x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 0.625}
K_A, C_A = 1.0, 10.0
SRC_A = f"{math.pi ** 2}*sin(pi*x)*(y + 0.3) + {C_A}*sin(pi*x)*(y + 0.3)"


def _u(x, y):
    return math.sin(math.pi * x) * (y + 0.3)


def _side(work: Path, name: str, *, scale: float = 1.0, source: str = SRC_A,
          config_extra: dict | None = None, dumps: bool = True,
          deliverables: bool = True) -> None:
    d = work / name
    d.mkdir(parents=True, exist_ok=True)
    cfg = dict(BOX_A, k=K_A, reaction=C_A, source_expr=source, level=1, nx=8, ny=8)
    cfg.update(config_extra or {})
    (d / "config.json").write_text(json.dumps(cfg))
    # each side's own box, so its files land inside the subdomain its config states
    bx = {q: float(cfg[q]) for q in ("x0", "x1", "y0", "y1")}
    for lvl, n in ((1, 8), (2, 16), (3, 32)):
        hx = (bx["x1"] - bx["x0"]) / n
        hy = (bx["y1"] - bx["y0"]) / n
        if deliverables:                      # the task's probe grid: cell midpoints
            rows = [(bx["x0"] + (i + .5) * hx, bx["y0"] + (j + .5) * hy)
                    for i in range(n) for j in range(n)]
            (work / f"solution_level{lvl}_{name[-1]}.csv").write_text(
                "x,y,u\n" + "".join(f"{x},{y},{scale * _u(x, y)}\n" for x, y in rows))
        if dumps:                             # the participant's own mesh nodes
            verts = [(bx["x0"] + i * hx, bx["y0"] + j * hy)
                     for i in range(n + 1) for j in range(n + 1)]
            (d / f"field_level{lvl}.csv").write_text(
                "x,y,u\n" + "".join(f"{x},{y},{scale * _u(x, y)}\n" for x, y in verts))


def _finding(work: Path, side: str) -> str:
    for f in equation_findings(work):
        if f["sequence"].endswith(side):
            return f["finding"]
    return ""


def test_a_side_that_solves_its_stated_equation_is_told_so(tmp_path):
    _side(tmp_path, "side_A")
    got = _finding(tmp_path, "side_A")
    assert "SATISFIES ITS OWN EQUATION" in got, got
    assert "+ 10 u = f" in got, "the reaction the config states is not in the verdict: " + got


def test_a_wrong_field_is_named_and_the_finding_outranks_the_bookkeeping(tmp_path):
    _side(tmp_path, "side_A", scale=1.2)
    fs = [f for f in equation_findings(tmp_path) if f["sequence"].endswith("side_A")]
    assert fs and "DOES NOT SATISFY THE EQUATION" in fs[0]["finding"], fs
    assert fs[0]["priority"] <= 10, (
        "a field converging to the wrong function must not sit below the "
        "bookkeeping findings: " + str(fs[0]["priority"]))


def test_the_mesh_dump_alone_is_not_judged(tmp_path):
    """The file whose quadrature does not fit may not produce a verdict."""
    _side(tmp_path, "side_A", scale=1.2, deliverables=False)
    got = _finding(tmp_path, "side_A")
    assert "NOT CHECKED" in got, got
    assert "MIDPOINTS" in got or "midpoint" in got, got
    assert "SATISFIES" not in got


def test_a_config_without_the_operator_names_the_missing_key(tmp_path):
    _side(tmp_path, "side_A", source="")
    got = _finding(tmp_path, "side_A")
    assert "NOT CHECKED" in got and "source_expr" in got, got


def test_both_sides_are_checked_and_named_separately(tmp_path):
    _side(tmp_path, "side_A")
    _side(tmp_path, "side_B", config_extra={"y0": 0.625, "y1": 1.5,
                                            "k": 2.0, "reaction": 0.0})
    seqs = sorted(f["sequence"] for f in equation_findings(tmp_path))
    assert seqs == ["equation check side side_A", "equation check side side_B"], seqs


def test_the_audit_carries_it_without_anyone_calling_a_tool(tmp_path):
    """audit_results and the delivery hook both run audit(); this must be in it."""
    _side(tmp_path, "side_A", scale=1.2)
    out = audit(str(tmp_path))
    blob = json.dumps(out)
    assert "DOES NOT SATISFY THE EQUATION" in blob, (
        "the equation verdict does not reach the audit reply: "
        + blob[:1200])


LIVE = ROOT / "campaign3_blind/runs/C3_27b_MCP_seed8862/work"


@pytest.mark.skipif(not LIVE.is_dir(), reason="the recorded cell is not present")
def test_the_two_recorded_correct_sides_come_back_satisfied():
    """C3 seed 8862, graded CORRECT at order 1.977, from its own files."""
    fs = {f["sequence"]: f["finding"] for f in equation_findings(LIVE)}
    assert len(fs) == 2, fs
    for seq, text in fs.items():
        assert "SATISFIES ITS OWN EQUATION" in text, (seq, text[:400])


def test_good_news_and_not_yet_do_not_make_an_audit_dirty(tmp_path):
    """Only a field that does not satisfy its equation is a DEFECT.

    A verdict of satisfied, and a side whose deliverable does not exist yet,
    are news: an audit run after level 1 would otherwise be 'dirty' on every
    coupled run for as long as the deliverables are unwritten, which is most of
    the run.
    """
    _side(tmp_path, "side_A")
    for f in equation_findings(tmp_path):
        assert f.get("informational") is True, f
    _side(tmp_path, "side_B", scale=1.2,
          config_extra={"y0": 0.625, "y1": 1.5, "k": 2.0, "reaction": 0.0})
    bad = [f for f in equation_findings(tmp_path)
           if "DOES NOT SATISFY" in f["finding"]]
    assert bad and not bad[0].get("informational"), bad
