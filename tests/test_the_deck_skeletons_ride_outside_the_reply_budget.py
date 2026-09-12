"""Both 4C deck skeletons (Scalar_Transport and the TSI slab) reach the agent WHOLE on every coupling
reply for 4C -- the parent's first reply and the worker's pointer-mode reply -- while the text before
them keeps the reply budget. Measured 2026-09-12: the grammar sat last, the cap cut it, and the TSI
skeleton never reached a worker; round 40's C1 cells spent their budgets on that grammar."""
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


def _knowledge():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _StubMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["knowledge"]


def test_both_skeletons_arrive_whole_in_the_first_and_the_pointer_reply():
    from tools.consolidated import _KNOWLEDGE_REPLY_LIMIT, _UNIVERSAL_CORE
    from backends.fourc.deck_grammar import FOURC_DECK_SKELETONS
    K = _knowledge()
    first = K(topic="coupling", solver="fourc", physics="thermoelastic")
    pointer = K(topic="coupling", solver="fourc", physics="thermoelastic")
    for name, out in (("first", first), ("pointer", pointer)):
        for phrase in ("SCATRA FLUX CALC LINE CONDITIONS:", "THE TSI SLAB DECK GRAMMAR", 'PROBLEMTYPE: "Thermo_Structure_Interaction"',
                       "TAG: monitor_reaction", "DVOL-NODE TOPOLOGY:", "Run it as `stdbuf"):
            assert phrase in out, f"{name} reply lacks {phrase!r}"
        assert out.count("THE TSI SLAB DECK GRAMMAR") == 1, name
        assert len(out) <= _KNOWLEDGE_REPLY_LIMIT + len(FOURC_DECK_SKELETONS) + len(_UNIVERSAL_CORE) + 800, (name, len(out))
        assert out.rstrip().endswith(_UNIVERSAL_CORE.rstrip()), name          # the core still closes the reply
    # the worker's reply keeps the contract block whole in front of the skeletons
    assert "WHAT YOUR SOLVE MUST LEAVE BEHIND" in pointer and pointer.index("WHAT YOUR SOLVE MUST LEAVE BEHIND") < pointer.index("THE TSI SLAB DECK GRAMMAR")
