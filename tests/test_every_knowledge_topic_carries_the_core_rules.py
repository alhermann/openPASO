"""Every topic an agent can ask for must come back with the core rules.

WHY THIS TEST EXISTS. `_UNIVERSAL` was appended on exactly ONE of the 31 return
paths of `knowledge()` — the topic="physics" path. Measured over this campaign's
995 knowledge calls from 193 openPASO-arm runs: topic="pitfalls" was 66.3% of calls
and topic="physics" 11.5%, so 75.6% of openPASO-arm runs received NONE of the
universal guidance. Every rule written during development — where the deliverable
goes, that a deck is not Python, that a "broken" solver is usually an unread log,
the refinement ladder — reached at most a quarter of the runs it was aimed at.

This asks the question the recurring defect class demands: not "does the code
append the block somewhere", but "what does an agent END UP WITH for the topics
it actually calls". It drives the real tool, over the real topic list, and it
fails if any topic loses the rules — including a topic added later.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _CollectingMCP:
    """Captures the functions register_consolidated_tools() decorates."""

    def __init__(self):
        self.tools: dict[str, object] = {}

    def tool(self, *a, **kw):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture(scope="module")
def knowledge_tool():
    # THE SERVER LOADS THE BACKENDS BEFORE REGISTERING TOOLS (server.py:205).
    # Without this the registry is empty, every solver resolves to
    # "Unknown solver" and the test would grade the dead-end path while
    # believing it had exercised the real one.
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _CollectingMCP()
    register_consolidated_tools(mcp)
    assert "knowledge" in mcp.tools, (
        f"the consolidated server no longer registers a tool called "
        f"'knowledge'; it registers {sorted(mcp.tools)}"
    )
    return mcp.tools["knowledge"]


def _declared_topics() -> list[str]:
    """The topics the tool itself branches on — read from its own source."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    start = src.index("def _knowledge_body(")
    end = src.index("\n    @mcp.tool()", start)
    body = src[start:end]
    topics = set()
    for m in re.finditer(r"topic\s*(?:==|in)\s*(\(|\[)?([^\n:]+)", body):
        for lit in re.findall(r"""['"]([a-z_]+)['"]""", m.group(2)):
            topics.add(lit)
    return sorted(topics)


# The topics the campaign's own trajectories show agents calling, with their
# measured share of 995 calls. `physics` is the one that used to be served.
MEASURED_TOPICS = ["pitfalls", "physics", "coupling", "input_guide", "overview",
                   "postmortems", "cross_backend", "precice", "materials",
                   "install"]

CORE_MARKERS = [
    "THE DELIVERABLE GOES IN THE DIRECTORY YOU WERE GIVEN",
    "A SOLVER'S INPUT LANGUAGE IS NOT PYTHON",
    "DO NOT CONCLUDE A SOLVER IS BROKEN",
    # renamed when the ladder was corrected; the check that matters is that
    # the core routes to the falsifiable test, not that it restates it.
    "verify_pde_consistency",
    # Rule 5, added because C2_27b_MCP_seed70 concluded "4C cannot accept
    # per-node Dirichlet values" and stopped, while the served 4C grammar shows
    # DESIGN POINT DIRICH CONDITIONS doing exactly that. seed70 and seed71 made
    # 25 knowledge calls between them and not one was topic="physics", so the
    # grammar never reached either.
    "INPUT FILE IS THE RUN INTERFACE",
    "4C DOES ACCEPT PER-NODE DIRICHLET VALUES",
]


def _has_core(text: str) -> bool:
    from tools.knowledge import _UNIVERSAL
    if _UNIVERSAL in text:          # the full block subsumes the core
        return True
    return all(m in text for m in CORE_MARKERS)


@pytest.mark.parametrize("topic", MEASURED_TOPICS)
def test_a_topic_agents_actually_call_comes_back_with_the_core(topic, knowledge_tool):
    out = knowledge_tool(topic=topic, solver="fourc", physics="heat")
    assert isinstance(out, str) and out, f"topic={topic!r} returned {out!r}"
    assert _has_core(out), (
        f"topic={topic!r} returned {len(out)} characters WITHOUT the core rules. "
        f"This is the defect that left 75.6% of openPASO-arm runs unguided."
    )


def test_no_declared_topic_loses_the_core(knowledge_tool):
    """Covers topics added after this test was written, not just measured ones."""
    missing = []
    for topic in _declared_topics():
        try:
            out = knowledge_tool(topic=topic, solver="fenicsx", physics="heat")
        except Exception as exc:                      # a raise is a separate bug
            missing.append(f"{topic}: raised {type(exc).__name__}: {exc}")
            continue
        if isinstance(out, str) and not _has_core(out):
            missing.append(f"{topic}: {len(out)} ch, no core")
    assert not missing, (
        "these topics reach the agent without the core rules:\n  "
        + "\n  ".join(missing)
    )


def test_an_unknown_solver_still_gets_the_core(knowledge_tool):
    """The dead-end replies are exactly where guidance matters most."""
    out = knowledge_tool(topic="physics", solver="not_a_solver", physics="heat")
    assert _has_core(out), (
        "an agent that names a solver wrong gets a bare 'Unknown solver' line "
        "and no guidance at all — the least helpful reply in the tool"
    )


def test_an_unknown_physics_still_gets_the_core(knowledge_tool):
    out = knowledge_tool(topic="physics", solver="fourc", physics="not_a_physics")
    assert _has_core(out), (
        "an agent asking for a physics the backend has no row for gets a "
        "dead end with no guidance"
    )


def test_a_future_return_path_cannot_leak(knowledge_tool):
    """The wrapper, not the 30 returns, is what guarantees this.

    If someone re-inlines the append into individual returns, the next path
    added leaks again — which is how this defect arose. Assert the structure
    that makes leaking impossible.
    """
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "@_functools.wraps(_knowledge_body" in src, (
        "knowledge() no longer wraps a body function, so the core is being "
        "appended per-return again and the next new return path will leak"
    )


def test_the_core_is_cheap_enough_to_send_every_time(knowledge_tool):
    """It is appended to every call and shares the model's context window."""
    from tools.knowledge import _UNIVERSAL, _UNIVERSAL_CORE
    # THE BOUND IS A DELIBERATE TRADE, RAISED ONCE, WITH THE NUMBERS.
    #
    # The core rides on every knowledge call, so it costs the arm that already
    # accumulates context fastest. Measured: openPASO runs take a median 39 tool
    # calls against bare's 95, at 91k input tokens per call against 60k, so
    # ACTIONS are the binding constraint. At ~5 knowledge calls per run this
    # block costs roughly 6k tokens of a 262k window.
    #
    # Against that, every rule in it was written from a run that lost its whole
    # attempt: a deliverable written where nobody looked, `**` in a deck, a
    # solver declared broken that was not, a field never read back at the probe
    # points, and three backend facts each of which ended a run on its own —
    # 4C's per-node Dirichlet, 4C's one-based design ids, FEBio's type="math".
    #
    # Raised from 4500 to 5500 when the one-based-id segfault was added. Adding
    # anything further should displace something, not push this again.
    assert len(_UNIVERSAL_CORE) < 5500, (
        f"the core is {len(_UNIVERSAL_CORE)} chars; it rides on EVERY knowledge "
        f"call, and the openPASO arm already accumulates context faster than bare. "
        f"Displace something rather than raising this bound again."
    )
    assert len(_UNIVERSAL_CORE) < len(_UNIVERSAL), "the core must be a subset"


def test_the_core_rules_arrive_exactly_once_on_the_physics_path(knowledge_tool):
    """ONCE — not zero, and not twice.

    This test used to demand ZERO, on the reasoning that the physics path
    carries the long block and appending the core would repeat the rules. The
    two blocks were independent literals, so what actually happened is that the
    long block went stale: measured 2026-09-01, `verify_pde_consistency` (the
    name of the verification tool), 4C's `E: 0` segfault, the halvings-not-cells
    refinement rule and the 13%/36% failure decomposition were ALL in the core
    and NONE of them in the 22,501-char block — and topic="physics" is the reply
    an agent asks for when it is stuck on precisely those things.

    `_UNIVERSAL` is now `_UNIVERSAL_CORE + <detail>`, so the paths agree by
    construction. The property worth testing is what the AGENT ends up with:
    the rules present, and present once.
    """
    from tools.knowledge import _UNIVERSAL_CORE
    out = knowledge_tool(topic="physics", solver="fourc", physics="heat")
    assert out.count(_UNIVERSAL_CORE) == 1, (
        f"the core rules appear {out.count(_UNIVERSAL_CORE)} times in the "
        f"physics reply; they must appear exactly once — zero means this path "
        f"is served a stale copy, twice means the agent reads them twice"
    )


def test_the_long_block_cannot_go_stale_against_the_core(knowledge_tool):
    """The guard that makes the defect above unrepeatable.

    A future edit that turns `_UNIVERSAL` back into its own copy of the rules
    would restore exactly the divergence this replaced, and no behavioural test
    would notice until an agent needed the missing fact.
    """
    from tools.knowledge import _UNIVERSAL, _UNIVERSAL_CORE
    assert _UNIVERSAL.startswith(_UNIVERSAL_CORE), (
        "the long physics block must BEGIN with the core verbatim: by "
        "construction so it cannot drift, and first because truncation cuts "
        "tails"
    )
    for fact in ("verify_pde_consistency", "E: 0", "refine_global",
                 "REFINEMENT COUNTS HALVINGS"):
        assert fact in _UNIVERSAL, f"{fact!r} reaches only one of the two paths"
