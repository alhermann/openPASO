"""A turn that ends with no tool call and no RESULT.txt must not end the run.

The defect this guards against was worth 112 of the campaign's 813 runs — and
it was not arm-neutral in effect: 19.5% of openPASO runs against 8.1% of bare.
The stopped runs were mid-task, with most of their clock unspent, announcing
the next action they never got to emit.

These tests drive run_one with a stub agent, so they measure the loop itself
rather than a model's behaviour.
"""
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class _Msg:
    """Minimal stand-in for a LangChain message."""

    def __init__(self, content="", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class StubAgent:
    """Replays a scripted list of turns and records the prompts it was given.

    Each scripted turn is (final_content, writes_answer). A turn that carries
    no tool calls is exactly the shape that used to end the run.
    """

    def __init__(self, turns, work):
        self.turns = list(turns)
        self.work = Path(work)
        self.seen = []

    async def ainvoke(self, state, config=None):
        self.seen.append(state["messages"])
        content, writes = self.turns.pop(0) if self.turns else ("done", True)
        if writes:
            (self.work / "RESULT.txt").write_text("LEVELS = 1\n")
        return {"messages": list(state["messages"]) + [_Msg(content)]}


def _run(turns, tmp, monkeypatch, timeout_s=60):
    import run_blind as R
    holder = {}

    def _build(size, seed, workdir):
        holder["ag"] = StubAgent(turns, workdir)
        return holder["ag"]

    monkeypatch.setattr(R, "build_mcp_agent", _build)
    monkeypatch.setattr(R, "build_bare_agent", _build)
    monkeypatch.setattr(R, "per_cell_exposure_check", lambda *a, **k: None)
    # The seal guard reads HERE/keys, which the tmp tree has not got; absence
    # is correctly NOT a seal, and that property has its own test.
    monkeypatch.setattr(R, "keys_are_sealed", lambda: True)
    tmp = Path(tmp)
    (tmp / "problems" / "C1").mkdir(parents=True, exist_ok=True)
    (tmp / "problems" / "C1" / "task.txt").write_text("TASK: solve it.\n")
    monkeypatch.setattr(R, "HERE", tmp)
    rec = R.run_one("C1", "27b", "MCP", 99, timeout_s)
    return rec, holder["ag"]


def test_a_stop_without_the_answer_file_is_continued(tmp_path, monkeypatch):
    """Two empty turns, then the agent writes the file. The run must survive
    both stops and end with the deliverable, not after the first turn."""
    rec, ag = _run([("Now let me run the coupling again:", False),
                    ("Let me write the participant:", False),
                    ("Done.", True)], tmp_path, monkeypatch)
    assert rec["continuations"] == 2, rec
    assert rec["error"] is None, rec
    assert len(ag.seen) == 3, f"agent invoked {len(ag.seen)} times"


def test_the_nudge_names_the_file_and_says_nothing_else(tmp_path, monkeypatch):
    """It must carry the contract and no domain content — both arms get it."""
    _, ag = _run([("stopped", False), ("Done.", True)], tmp_path, monkeypatch)
    nudge = ag.seen[1][-1][1]
    assert "RESULT.txt" in nudge and "COULD_NOT_COMPLETE" in nudge
    for leak in ("order", "flux", "interface", "mesh", "solution",
                 "Dirichlet", "Neumann", "converge"):
        assert leak.lower() not in nudge.lower(), f"nudge leaks '{leak}'"


def test_an_answered_run_is_never_nudged(tmp_path, monkeypatch):
    rec, ag = _run([("Done.", True)], tmp_path, monkeypatch)
    assert rec["continuations"] == 0
    assert len(ag.seen) == 1, "nudged a run that had already answered"


def test_continuations_are_capped(tmp_path, monkeypatch):
    """An agent that never answers must not loop forever."""
    rec, ag = _run([("nope", False)] * 40, tmp_path, monkeypatch)
    assert rec["continuations"] == 8, rec
    assert len(ag.seen) == 9


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
