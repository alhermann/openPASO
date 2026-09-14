"""The physics reply can be halved, and whether that HELPS is not yet known.

MEASURED. Served volume is dominated by topic="physics": 51,402 characters for
4C, of which `_UNIVERSAL` is 22,501 — and 93% of the Kratos reply, whose own
knowledge is 1,595 characters. The arm that receives all this carries 90,740
input tokens per call against the unassisted arm's 58,626, takes 41 tool calls
against 99, and stops at 44% of its time budget while its give-up texts blame
the clock in 86.5% of cases.

Within the assisted arm the correlation is monotone: runs graded CORRECT carry
70,273 tokens per call and make 52 calls; runs that give up carry 87,523 and
make 37. That is CORRELATION ONLY. A run that solves quickly naturally makes
fewer heavy knowledge calls, which is reverse causation and fits the numbers
just as well.

So this is a switch, default OFF, to be settled by running the same cells and
seeds both ways — not a fix applied on a hunch. These tests pin that: the
default is byte-identical to the full block, the lean form still carries every
core rule, and it offers the elaboration rather than deleting it.

THE EXPERIMENT HAS SINCE BEEN RUN AND IT REFUTED THE HYPOTHESIS. NG1, KR1 and
FC1, one seed each per arm, only the flag differing:

    FULL block   CORRECT, CORRECT, CORRECT   (orders 2.063, 2.005, 2.000)
    LEAN         CONFIDENTLY_WRONG (0.071), MALFORMED, CONFIDENTLY_WRONG (-0.033)

Total served tokens fell 4.2M -> 3.5M as designed; context per call moved only
-9.3% and ACTIONS fell 7.7%, with FC1 reversing both signs. The correlation was
at least partly reverse causation. The elaboration the lean form removes is
LOAD-BEARING, so the default must stay as it is — which is what
test_the_default_is_the_full_block_unchanged now protects with a reason rather
than a precaution.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _tail(flag: str) -> str:
    old = os.environ.get("OPENPASO_LEAN_PHYSICS")
    os.environ["OPENPASO_LEAN_PHYSICS"] = flag
    try:
        import tools.knowledge as K
        importlib.reload(K)
        return K._physics_tail()
    finally:
        if old is None:
            os.environ.pop("OPENPASO_LEAN_PHYSICS", None)
        else:
            os.environ["OPENPASO_LEAN_PHYSICS"] = old
        import tools.knowledge as K2
        importlib.reload(K2)


def test_the_default_is_the_full_block_unchanged():
    import tools.knowledge as K
    assert _tail("") == K._UNIVERSAL, (
        "the default physics reply changed; an unmeasured reduction must not "
        "ship silently"
    )


def test_the_lean_switch_no_longer_cuts_anything_and_is_retired():
    """The premise of the experiment is gone: the DEFAULT was cut instead.

    This test used to require lean < 40 % of the default, which was the point
    when the default carried 22,501 characters of universal block. It no longer
    holds, and not because lean grew: the default was cut to 4,099 by the later
    size reduction, which moved the long form behind
    knowledge(topic='universal_full'). Lean is _UNIVERSAL_CORE plus an offer
    paragraph -- 4,266 -- so the switch now returns MORE text than leaving it
    off, and there is nothing left for it to cut.

    The experiment it existed for was also already run, and refuted. NG1, KR1
    and FC1, one seed each per arm, only the flag differing: the full block gave
    CORRECT, CORRECT, CORRECT at orders 2.063, 2.005 and 2.000, and lean gave
    CONFIDENTLY_WRONG (0.071), MALFORMED and CONFIDENTLY_WRONG (-0.033). The
    elaboration lean removes is load-bearing.

    So this records the outcome rather than asserting a cut that cannot happen.
    Turning the switch on today serves more text AND the worse text. Whether to
    delete it is Alexander's call, not something to do while fixing a test --
    the remaining tests in this file still pin that the default is the full
    block, that lean carries every core rule, and that the route it offers
    works.
    """
    full, lean = _tail(""), _tail("1")
    assert len(lean) >= len(full), (
        f"lean is {len(lean)} against a default of {len(full)}. If lean has "
        f"become a real cut again, the default must have grown back -- check "
        f"whether the long form has returned to the physics reply, and restore "
        f"the original assertion with the measured sizes.")


def test_lean_still_carries_every_core_rule():
    import tools.knowledge as K
    lean = _tail("1")
    for rule in ("THE DELIVERABLE GOES IN THE DIRECTORY YOU WERE GIVEN",
                 "A SOLVER'S INPUT LANGUAGE IS NOT PYTHON",
                 "DO NOT CONCLUDE A SOLVER IS BROKEN",
                 "verify_pde_consistency",
                 "INPUT FILE IS THE RUN INTERFACE",
                 "4C DOES ACCEPT PER-NODE DIRICHLET VALUES"):
        assert rule in lean, f"lean mode drops the core rule {rule!r}"


def test_lean_offers_the_rest_rather_than_deleting_it():
    lean = _tail("1")
    assert "available on request" in lean
    assert "signal='probe evaluation'" in lean, (
        "the agent must be told the exact call that returns the elaboration"
    )
    assert "costs you actions" in lean, (
        "and why it is offered rather than pushed"
    )


def test_the_signal_route_it_promises_actually_returns_the_longer_form():
    """A promise openPASO cannot keep is worse than an honest cap — measured
    once already on the coupling payload's escape hatch."""
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools

    class _MCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco
    os.chdir(ROOT)
    mcp = _MCP()
    register_consolidated_tools(mcp)
    out = mcp.tools["knowledge"](topic="physics", solver="fourc",
                                 physics="heat", signal="probe evaluation")
    assert "PROBE POINTS" in out.upper() or "nearest node" in out.lower(), (
        "the call the lean note names does not return the probe material it "
        "promises"
    )
