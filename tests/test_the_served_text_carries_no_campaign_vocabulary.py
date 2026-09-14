"""Nothing a solver agent reads may mention the evaluation around it.

OASiS is a product; the tasks it is measured on are not part of it. A served payload that says a run
was "graded", or cites "round 45", or calls its own evidence a "step-trial", tells the reader there is
an evaluation and invites them to write for it. Measured 2026-09-14: thirty such occurrences across the
coupling replies, seven of them added that same day while writing measured facts -- which is exactly
how this leaks, one honest sentence at a time.

The physics sense of "graded" (a graded material, a graded source) is a different word and stays.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FORBIDDEN = ["step-trial", "step trial", "the grader", "campaign", "benchmark",
             "evaluation cell", "graded as", "graded interface", "is graded",
             "round 4", "round 5", "blind evaluation"]
CODES = ["fourc", "fenics", "dealii", "ngsolve", "skfem", "kratos", "dune", "febio", "sparta"]
VARIANTS = ["", "thermoelastic", "elasticity", "transient", "3d", "fsi"]


class _MCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def _knowledge():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _MCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


@pytest.mark.parametrize("code", CODES)
def test_no_coupling_reply_mentions_the_evaluation(code):
    K = _knowledge()
    for physics in VARIANTS:
        for _ in range(2):      # the session's first reply and a worker's own call
            out = K(topic="coupling", solver=code, physics=physics) if physics else K(topic="coupling", solver=code)
        for word in FORBIDDEN:
            m = re.search(re.escape(word), out, re.I)
            assert not m, (f"{code}/{physics or 'base'} serves {word!r}: "
                           f"...{out[max(0, m.start() - 70):m.start() + 70]}...")


def test_the_participants_themselves_are_clean():
    for p in sorted((ROOT / "data" / "coupling_participants").glob("participant_*.py")):
        t = p.read_text()
        for word in FORBIDDEN:
            assert not re.search(re.escape(word), t, re.I), f"{p.name} contains {word!r}"


def test_the_physics_sense_of_graded_is_untouched():
    """A graded material and a graded source are ordinary physics and must survive the rule."""
    hits = [p.name for p in (ROOT / "data" / "coupling_participants").glob("participant_*.py")
            if "graded / manufactured" in p.read_text()]
    assert hits, "the physics sense of the word was removed along with the grading sense"
