"""The first thing a coupled run is told to copy had a half-written sentence.

`YOUR FIRST SUB-AGENT, NOW` is the block the parent is told to copy verbatim
into its first spawn_subagent call, and the run's first worker sees nothing but
that string. Since commit c26ea0ce (2026-09-11) its deck clause reads

    Write ./side_A/config.json for
    level 1 and a synthetic ./side_A/imports.json; if the code takes an input deck, run
    the deck is READ AND JUDGED THE MOMENT YOU WRITE IT -- write it to a
    .yaml, .yml or .dat file and the defects come back in that reply, before
    the binary runs, so do not spend an action asking for the check; run the
    script with that code's own interpreter ...

Two sentences were spliced: the original told the worker to `run check_input`
until it named no defect, a later edit replaced that advice with the
write-time judgement, and the verb `run` was left hanging. The sibling copy in
the ladder (result_audit.py, `Write ./config.json for level 1 and a synthetic
./imports.json; if the code takes an input deck, run check_input(...)`) is
whole, so the two briefs also disagree.

This matters for the side that fails most. Over the 242 recorded C3 runs, 4C is
the silent side twice as often as DUNE, and of the 122 runs that never wrote
`side_B/exports.json`, 71 never wrote the participant at all -- the worker whose
brief this is. A dangling verb in the one sentence about decks is not a
hypothesis about why, but it is ours to fix and it costs nothing.

The test pins the defect, not the prose: whatever the block says about decks has
to parse as a sentence, and the clause introducing an input deck may not end on
a verb with nothing to govern.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tools.consolidated import _COUPLING_MUST_READ          # noqa: E402

MARK = "YOUR FIRST SUB-AGENT, NOW"


def _brief() -> str:
    """The block the parent copies -- the LAST mention of the marker is the block
    itself; earlier ones point at it."""
    j = _COUPLING_MUST_READ.index("THEN A SECOND WORKER FOR SIDE B")
    i = _COUPLING_MUST_READ.rindex(MARK, 0, j)
    return _COUPLING_MUST_READ[i:j]


def _flat(text: str) -> str:
    """One line, so a clause that wraps reads the same as one that does not."""
    return re.sub(r"\s+", " ", text)


def test_the_deck_clause_does_not_end_on_a_dangling_verb():
    brief = _flat(_brief()).lower()
    assert "input deck" in brief, "the brief no longer mentions a deck at all"
    for verb in ("run", "write", "call", "check"):
        bad = re.search(rf"input deck, {verb}(?! check_input\()\b", brief)
        assert not bad, (
            f"the deck clause ends on '{verb}' with nothing to govern: "
            f"{brief[bad.start():bad.start() + 120]!r}")


def test_the_deck_sentence_says_what_happens_when_the_deck_is_written():
    """The judgement arrives in the write reply; that is what the worker needs."""
    brief = _flat(_brief())
    assert "READ AND JUDGED THE MOMENT YOU WRITE IT" in brief
    head = brief[:brief.index("READ AND JUDGED THE MOMENT YOU WRITE IT")]
    tail = head[head.rindex("."):] if "." in head else head
    assert "deck" in tail.lower(), (
        "the judgement sentence does not name the deck it is about: "
        f"{tail[-160:]!r}")


def test_the_worker_is_still_told_to_run_the_script_afterwards():
    brief = _flat(_brief())
    assert re.search(r"[Rr]un the script with that code's own interpreter", brief), \
        "the brief must still tell the worker to run the script with that code's interpreter"
    assert "exports.json appears with finite values" in brief


def test_the_ladders_copy_of_the_brief_is_whole_too():
    """The ladder serves the same brief for a failed step; it may not dangle either."""
    src = (ROOT / "src" / "tools" / "result_audit.py").read_text()
    clauses = re.findall(r"input deck, (\w+)", src)
    assert clauses, "the ladder brief no longer mentions an input deck"
    for verb in clauses:
        assert verb in ("run", "the"), verb
        if verb == "run":
            i = src.index("input deck, run")
            after = src[i + len("input deck, run"):i + len("input deck, run") + 40]
            assert re.match(r"\s+check_input\(", after), (
                f"the ladder's 'run' governs nothing: {after!r}")
