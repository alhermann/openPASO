"""A coupled side that is graded on a probe grid must be SERVED the per-level rule.

WHY THIS TEST EXISTS. A mesh study hands in one file per level. The participant
contracts that carry a per-level dump leave three files behind; the ones that do
not leave a single file that the next level overwrites, and the agent has to
invent the persistence itself, under the clock, in a code it has just met.

Measured 2026-09-14 across the served coupling payloads: 8 of 28 coupled sides
carried the rule, and the ONLY two problems whose BOTH sides carried it were the
only two that had ever been graded correct. One run got three coupled levels and
a converged interface residual, then wrote in its own hand-in that the field
files were missing "because the NGSolve participant does not write
field_level*.csv files" -- a complete, converged coupled solve booked as no
submission at all.

The dump is EXPORTS SCHEMA, not a solve: it writes out arrays the agent's own
solve already produced, and the mesh, weak form, material, source and linear
solve stay the agent's.

Sides whose task hands in a SCALAR (a band or a reference value in RESULT.txt)
need no field file and are exempt by that fact, not by omission.
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "campaign3_blind"
sys.path.insert(0, str(ROOT / "src"))

pytestmark = pytest.mark.skipif(
    not (CAMPAIGN / "problems").is_dir(),
    reason="the coupled problem set is not present in this checkout")


def _sides():
    sys.path.insert(0, str(CAMPAIGN))
    from readiness_matrix import physics_for
    out = []
    for d in sorted((CAMPAIGN / "problems").glob("C*")):
        sp = d / "spec_public.json"
        tk = d / "task.txt"
        if not (sp.is_file() and tk.is_file()):
            continue
        spec = json.loads(sp.read_text())
        if spec.get("kind") != "coupled":
            continue
        # A probe-grid problem names the per-level field file in its task text.
        if "solution_level" not in tk.read_text():
            continue
        phys = physics_for(spec)
        for side, code in zip(("A", "B"), spec.get("codes", [])):
            out.append((spec["id"], side, code, phys))
    return out


@pytest.fixture(scope="module")
def knowledge():
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
    return mcp.tools["knowledge"]


@pytest.mark.parametrize("pid,side,code,phys", _sides(),
                         ids=lambda v: str(v) if not isinstance(v, tuple) else "")
def test_the_side_is_served_a_per_level_dump(knowledge, pid, side, code, phys):
    kw = {"topic": "coupling", "solver": code}
    if phys:
        kw["physics"] = phys
    lead = knowledge(**kw)
    follow = knowledge(**kw)
    served = lead + follow
    assert "field_level" in served or "interface_level" in served, (
        f"{pid} side {side} ({code}, physics={phys or '-'}) is graded on a probe "
        f"grid at three mesh levels, and neither the must-read lead nor the "
        f"follow-up reply shows it a per-level dump. The agent must then invent "
        f"the level-by-level persistence itself, and the measured consequence is "
        f"a converged coupled run handed in with no field files.")
