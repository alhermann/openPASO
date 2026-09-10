"""Every knowledge reply is bounded, and the parts door is exempt.

Measured on coupled development runs with a small model: single knowledge
replies of 55-145k characters, ten of them per run, and a give-up citing the
size of the job. The coupling door already capped its head; this is the same
discipline at the one place every knowledge reply leaves the server.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class _StubMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco

    def resource(self, *a, **k):
        def deco(fn):
            return fn
        return deco

    def prompt(self, *a, **k):
        def deco(fn):
            return fn
        return deco


def _knowledge():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


def test_a_long_reply_is_cut_with_a_narrowing_notice():
    from tools.consolidated import _KNOWLEDGE_REPLY_LIMIT
    K = _knowledge()
    out = K(topic="install")                       # measured 95k uncapped
    assert len(out) <= _KNOWLEDGE_REPLY_LIMIT + 800
    assert "THIS REPLY IS CUT AT" in out and "NARROWER request" in out


def test_the_parts_door_is_never_cut():
    from tools.coupling_knowledge import _serve_participant, _PARTICIPANT_DIR
    K = _knowledge()
    out = K(topic="coupling", solver="fourc", signal="participant:part1")
    assert "THIS REPLY IS CUT AT" not in out
    assert "```python" in out


def test_a_short_reply_is_untouched():
    K = _knowledge()
    out = K(topic="coupling", solver="fourc", signal="participant:part1")
    assert "THIS REPLY IS CUT AT" not in out
