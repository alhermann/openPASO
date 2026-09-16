"""Two served participants, through the real driver, leave every per-level piece.

WHY THIS TEST EXISTS. The deliverable chain has five links, and each was found
broken on its own at some point: the participant must dump its field and
interface data per level; it must print the run-log contract line from the space
it actually built; the driver must keep each side's console under that level's
name; the iteration must leave a residual history; and the counts must actually
grow between levels. Every one of those was fixed against a measurement, and
none of the fixes was exercised END TO END until here -- a cell that has four of
the five hands in nothing gradeable.

It asserts what a grader reads, not how it was produced: the files exist per
level and per side, and the degree-of-freedom count GROWS. The physics is not
checked here; two other suites do that.

Deliberately the two fastest served pairs and tiny meshes, because this runs two
real solvers per level.
"""
from __future__ import annotations

import asyncio
import inspect
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CONTRACTS = ROOT / "data" / "coupling_participants"
sys.path.insert(0, str(SRC))

VENV = Path("/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python")
LEVELS = ((1, 6), (2, 12))

pytestmark = pytest.mark.skipif(
    not VENV.is_file() or not (CONTRACTS / "participant_skfem.py").is_file(),
    reason="needs the server venv and the served contracts")


def _tools():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools

    class _MCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    mcp = _MCP()
    register_consolidated_tools(mcp)
    return mcp.tools


def test_every_per_level_piece_is_left_behind(tmp_path):
    try:
        import skfem  # noqa: F401
        import KratosMultiphysics  # noqa: F401
    except Exception:                                    # noqa: BLE001
        pytest.skip("this interpreter does not carry both codes")
    couple = _tools()["couple"]
    for side, src in (("side_A", "participant_skfem.py"), ("side_B", "participant_kratos.py")):
        (tmp_path / side).mkdir()
        shutil.copy(CONTRACTS / src, tmp_path / side / f"participant_{side[-1]}.py")
        (tmp_path / side / "imports.json").write_text("{}")
    parts = [{"name": s[-1], "command": [str(VENV), f"participant_{s[-1]}.py"],
              "work_dir": str(tmp_path / s)} for s in ("side_A", "side_B")]
    loop = asyncio.new_event_loop()
    try:
        for lvl, n in LEVELS:
            for side in ("side_A", "side_B"):
                (tmp_path / side / "config.json").write_text(
                    json.dumps({"level": lvl, "nx": n, "ny": n}))
            r = couple(participants=json.dumps(parts), max_iter=30, tol=1e-9,
                       history_path=str(tmp_path / f"residual_level{lvl}.csv"),
                       critic_approved=True)
            if inspect.isawaitable(r):
                r = loop.run_until_complete(r)
    finally:
        loop.close()

    missing = []
    for lvl, _ in LEVELS:
        for side in ("side_A", "side_B"):
            for stem in (f"field_level{lvl}.csv", f"interface_level{lvl}.csv",
                         f"participant_output_level{lvl}.log"):
                if not (tmp_path / side / stem).is_file():
                    missing.append(f"{side}/{stem}")
        if not (tmp_path / f"residual_level{lvl}.csv").is_file():
            missing.append(f"residual_level{lvl}.csv")
    assert not missing, (
        "the driver and the served contracts did not leave every per-level "
        f"piece a grader reads: {missing}")


def test_the_dof_count_is_reported_and_grows(tmp_path):
    """The contract line exists per level AND the mesh really refined."""
    try:
        import skfem  # noqa: F401
        import KratosMultiphysics  # noqa: F401
    except Exception:                                    # noqa: BLE001
        pytest.skip("this interpreter does not carry both codes")
    couple = _tools()["couple"]
    for side, src in (("side_A", "participant_skfem.py"), ("side_B", "participant_kratos.py")):
        (tmp_path / side).mkdir()
        shutil.copy(CONTRACTS / src, tmp_path / side / f"participant_{side[-1]}.py")
        (tmp_path / side / "imports.json").write_text("{}")
    parts = [{"name": s[-1], "command": [str(VENV), f"participant_{s[-1]}.py"],
              "work_dir": str(tmp_path / s)} for s in ("side_A", "side_B")]
    loop = asyncio.new_event_loop()
    counts: dict = {"side_A": [], "side_B": []}
    try:
        for lvl, n in LEVELS:
            for side in ("side_A", "side_B"):
                (tmp_path / side / "config.json").write_text(
                    json.dumps({"level": lvl, "nx": n, "ny": n}))
            r = couple(participants=json.dumps(parts), max_iter=30, tol=1e-9,
                       history_path=str(tmp_path / f"residual_level{lvl}.csv"),
                       critic_approved=True)
            if inspect.isawaitable(r):
                r = loop.run_until_complete(r)
            for side in ("side_A", "side_B"):
                log = tmp_path / side / f"participant_output_level{lvl}.log"
                text = log.read_text(errors="replace") if log.is_file() else ""
                got = [ln for ln in text.splitlines() if ln.strip().startswith("NDOF = ")]
                assert got, (
                    f"{side} level {lvl}: the captured console carries no "
                    f"`NDOF = <integer>` line on a line of its own, so a grader "
                    f"cannot tell this level's mesh from any other")
                counts[side].append(int(got[0].split("=")[1]))
    finally:
        loop.close()
    for side, seq in counts.items():
        assert seq[-1] > seq[0], (
            f"{side}: the reported dof count did not grow across the levels "
            f"({seq}) -- a refinement that does not change the count reads as "
            f"one mesh solved twice")
