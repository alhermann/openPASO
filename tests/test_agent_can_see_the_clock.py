"""The agent is told how much wall-clock is left, on every command.

Measured on C9, 27B, seeds 22 and 23: both wrote COULD_NOT_COMPLETE blaming
the budget — "Time constraints (45 minutes) prevented completion",
"Insufficient time ... within the 45-minute budget" — after using 1093 s and
1110 s of 2700. They stopped at eighteen minutes believing they were out of
forty-five, and threw away 59% of the run each.

The prompt states the budget once and nothing updates it, so the agent has no
way to know. This is a harness fact, not a capability: BOTH arms get it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "langgraph_eval"))


def test_the_note_states_time_remaining():
    import agent as A
    A._DEADLINE = (time.time() + 1800.0, 2700.0)
    try:
        note = A._time_left_note()
        assert "clock:" in note
        assert "min left of 45" in note, note
        assert "29 min left" in note or "30 min left" in note, note
    finally:
        A._DEADLINE = None


def test_no_deadline_means_no_note():
    """Nothing is stamped outside a measured run."""
    import agent as A
    A._DEADLINE = None
    A._ACTIONS_USED = 0        # other tests in the session drive tools; isolate
    assert A._time_left_note() == ""


def test_a_spent_budget_says_so():
    import agent as A
    A._DEADLINE = (time.time() - 5.0, 2700.0)
    try:
        assert "budget spent" in A._time_left_note()
    finally:
        A._DEADLINE = None


def test_the_shell_tool_stamps_it(tmp_path):
    import agent as A
    A._DEADLINE = (time.time() + 600.0, 2700.0)
    try:
        run_bash = A._bash_tool_for(tmp_path)
        out = run_bash.invoke({"command": "echo hello"})
        assert "hello" in out
        assert "clock:" in out and "min left of 45" in out
    finally:
        A._DEADLINE = None


def test_both_arms_get_it():
    """A clock is not an openPASO capability.

    The bare arm builds its shell tool from the same factory, so the stamp is
    arm-neutral by construction. This pins that the factory is shared rather
    than duplicated per arm.
    """
    src = (REPO / "langgraph_eval" / "agent.py").read_text()
    assert src.count("def _bash_tool_for(") == 1
    assert "_time_left_note()" in src
    # AND BOTH ARMS MUST ACTUALLY GET IT — asserted by behaviour, not by the
    # absence of a string in the source.
    #
    # This used to require that "audit_on_submit" appear nowhere inside
    # _bash_tool_for, as a stand-in for "the clock is not arm-specific". The
    # shell tool now carries openPASO's submission audit, exactly as write_file
    # does, so the proxy fails while the property it stands for still holds.
    import tempfile
    import agent as A
    # SAME MODULE OBJECT. `agent` and `langgraph_eval.agent` import as two
    # separate modules here, so patching _DEADLINE on one leaves the other
    # reading None and the note comes back empty for both arms.
    _bash_tool_for = A._bash_tool_for
    A._DEADLINE = (time.time() + 900.0, 2700.0)
    try:
        with tempfile.TemporaryDirectory() as d:
            bare = _bash_tool_for(Path(d), audit_on_submit=False).invoke(
                {"command": "echo x"})
            mcp = _bash_tool_for(Path(d), audit_on_submit=True).invoke(
                {"command": "echo x"})
    finally:
        A._DEADLINE = None
    assert "[clock:" in bare and "[clock:" in mcp, (
        f"the clock must reach BOTH arms; bare={bare[-60:]!r} mcp={mcp[-60:]!r}")


def test_the_note_carries_no_domain_content():
    import agent as A
    A._DEADLINE = (time.time() + 900.0, 2700.0)
    try:
        low = A._time_left_note().lower()
        for leak in ("flux", "interface", "converge", "mesh", "solve",
                     "dirichlet", "neumann", "submit", "result"):
            assert leak not in low, f"the clock note leaks {leak!r}"
    finally:
        A._DEADLINE = None


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
