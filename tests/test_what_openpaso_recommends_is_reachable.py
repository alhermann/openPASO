"""A tool openPASO tells an agent to call must be one the agent can call.

THE CLASS OF BUG, which has now bitten three times in two days across two
trees: a capability exists, the served text recommends it, and it is
unreachable for the population it was written for.

  1. The campaign harness: `couple_levels` was written on 2026-09-11 to solve
     the per-level budget problem -- "six proven couplings never reached level 3
     when every level cost ten calls" -- and the tool allowlist was frozen ten
     days earlier and never updated. It was invoked 3 times in 2095 trajectories
     against 1076 for couple(), while openPASO's own text recommending it
     appeared in 260 of them.

  2. This fork's product path: one flag gated BOTH the campaign's grading hooks
     and the script lints, and the product set it False, so nothing linted a
     written script at all. Every lint added that day reached the evaluation arm
     and no product user.

  3. Found by this test on its first run: `check_input`, `couple_levels` and
     `verify_interface_flux` are all named by the served coupling text and none
     is on CAMPAIGN_MCP_TOOL_ALLOWLIST here.

Each was invisible because the two halves are correct on their own. Only the
join is wrong, and nothing was checking the join.

WHY DEPRECATION IS HANDLED EXPLICITLY. `coupled_solve` is also named in that
text, and is named in order to say DEPRECATED. Absent from the allowlist is the
right state for it, so the test reads the sentence rather than the token.
"""
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def served():
    """Every tool name openPASO's own text tells an agent to call, and how."""
    import logging

    logging.disable(logging.INFO)
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools

    load_all_backends()
    captured = {}

    class _Recorder:
        def tool(self, *a, **k):
            def deco(fn):
                captured[fn.__name__] = fn
                return fn
            return deco

    register_consolidated_tools(_Recorder())
    knowledge = captured["knowledge"]
    texts = {
        "coupling": knowledge(topic="coupling"),
        "universal": knowledge(topic="universal"),
    }
    recommended, deprecated = set(), set()
    for text in texts.values():
        for name in set(re.findall(r"\b([a-z_]{4,30})\s*\(", text)):
            if name not in captured:
                continue                     # an API call in an example, not a tool
            for line in text.splitlines():
                if name not in line:
                    continue
                if "DEPRECATED" in line:
                    deprecated.add(name)
                else:
                    recommended.add(name)
    return {"registered": set(captured),
            "recommended": recommended - deprecated,
            "deprecated": deprecated}


def test_every_recommended_tool_is_registered(served):
    """The weaker half: the text must not name a tool that does not exist."""
    missing = sorted(served["recommended"] - served["registered"])
    assert not missing, (
        "the served text tells agents to call tools that are not registered: "
        + ", ".join(missing))


def test_every_recommended_tool_is_on_the_campaign_surface(served):
    """The half that has actually been wrong, three times.

    `surface='campaign'` is the narrower of the two openPASO serves. A tool the
    text recommends that is missing from it is recommended to a population that
    cannot call it -- which is precisely how couple_levels went 3 calls in 2095
    trajectories while being recommended in 260 of them.
    """
    from langgraph_eval.agent import CAMPAIGN_MCP_TOOL_ALLOWLIST

    unreachable = sorted(served["recommended"] - set(CAMPAIGN_MCP_TOOL_ALLOWLIST))
    assert not unreachable, (
        "openPASO's own text tells an agent to call these, and the campaign "
        "surface does not carry them:\n    " + "\n    ".join(unreachable)
        + "\n\nEither add them to CAMPAIGN_MCP_TOOL_ALLOWLIST, or stop "
          "recommending them. A capability that is recommended and unreachable "
          "is worse than one that does not exist: the agent spends calls "
          "looking for it.")


def test_a_deprecated_tool_is_allowed_to_be_absent(served):
    """Read the sentence, not the token: `coupled_solve` is named to warn."""
    assert served["deprecated"], (
        "no tool is marked DEPRECATED in the served text; if that is now true, "
        "this test is measuring nothing and should be removed")
    for name in served["deprecated"]:
        assert name not in served["recommended"], (
            f"{name} is both recommended and deprecated in the same text")
