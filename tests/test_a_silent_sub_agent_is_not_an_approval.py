"""A sub-agent that returns nothing handed the parent an empty string, and "" reads as assent.

MEASURED on a live run of the browser interface (Qwen 3.5 122B), in one turn:
the model spawned a critic, the critic ran for 2m16s and returned a result of
length ZERO, and the model then composed "CRITIC REVIEW ... VERDICT: APPROVED
with minor notes. Ready to run.", filed it through submit_critic_review, took
the token and passed it to the run. The gate accepted it, because the gate
matches the SETUP and records nothing about who wrote the review.

The empty return is the part this test is about. `spawn_subagent` ends with

    report = out["messages"][-1].content
    ...
    return report

and when a sub-agent finishes with no text that is "". An empty string handed
back to a model is indistinguishable from "no objection" -- it is the absence of
a complaint, which is exactly what assent looks like. The sub-agent may have
produced nothing because it ran out of steps, because its tool call failed, or
because it genuinely had nothing to say; none of those is approval, and the
parent cannot tell them apart from a blank.

The exception path was already loud ("[SUBAGENT FAILED -- this is NOT a review
verdict ...]"). Silence was not, which is the worse half: a crash is visible and
a blank is not.

The sentence below is shared word for word with the browser interface's own
handler (webui/runner.py), so a person reading a transcript and a person reading
a campaign trajectory see the same words. Only the product name differs while
this branch keeps the old one; the rename script maps it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytest.importorskip("langchain_core")

from langgraph_eval import agent as A                       # noqa: E402


MARK = "NOT approval and NOT a review"


def test_the_sentence_exists_and_says_what_silence_is_not():
    msg = A.SILENT_SUBAGENT_REPORT.format(who="critic")
    assert MARK in msg, msg
    assert "returned no text" in msg, msg
    assert "Treat the step as not done" in msg, msg
    assert "holds no review for this setup" in msg, msg


def test_it_names_the_role_that_was_silent():
    """'the critic returned no text' is what a reader needs, not 'a sub-agent'."""
    assert "{who}" in A.SILENT_SUBAGENT_REPORT
    assert A.SILENT_SUBAGENT_REPORT.format(who="critic").startswith("[the critic returned no text")
    assert A.SILENT_SUBAGENT_REPORT.format(who="sub-agent").startswith("[the sub-agent returned")


def test_the_guard_fires_only_on_emptiness():
    """A short real verdict must pass through untouched; only a blank is replaced.

    Read from the source of spawn_subagent rather than by driving it, because
    the tool is a closure built per cell; what is pinned is that the branch
    tests emptiness explicitly and that the real report is returned otherwise.
    """
    src = (ROOT / "langgraph_eval" / "agent.py").read_text()
    i = src.index("SILENT_SUBAGENT_REPORT.format(")
    guard = src[max(0, i - 500):i]
    assert "if not str(report).strip():" in guard, guard[-300:]
    after = src[i:i + 1400]
    assert "return report" in after, "the real report is no longer returned"


def test_the_words_match_the_browser_interface_exactly():
    """One sentence in both places, so a transcript and a trajectory agree.

    The browser interface ships in the product tree only, so this is skipped
    where that file is absent rather than pinning a copy of its text. The
    product name differs while this branch keeps the old one.
    """
    other = ROOT / "webui" / "runner.py"
    if not other.is_file():
        pytest.skip("the browser interface is not in this tree")
    theirs = other.read_text()
    if MARK not in theirs:
        pytest.skip("the browser interface does not carry the sentence here")
    # COMPARE THE SENTENCES, NOT A LIST OF PHRASES. This checked five hand-listed phrases, so a
    # clause added to ONE copy passed unnoticed -- which happened within the hour, while three
    # sessions were agreeing to add exactly one clause to all three copies. A test named "match
    # exactly" that matches a subset is the same shape as the defects it guards: right about one
    # direction, silent about the other.
    import re as _re
    mine = A.SILENT_SUBAGENT_REPORT.format(who="critic")
    found = _re.search(r"\[the \{who\} returned no text.*?\]", theirs, _re.S)
    assert found, "the browser interface's sentence is not in the shape this compares"
    interface = _re.sub(r'"\s*\n\s*(?:f?")?', "", found.group(0)).replace("{who}", "critic")
    norm = lambda s: " ".join(s.split())
    assert norm(interface) == norm(mine), (
        "the two copies have drifted:\n  here:      " + norm(mine)
        + "\n  interface: " + norm(interface))
