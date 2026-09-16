"""SUPERSEDED BY couple() RUNNING THE CHECK ITSELF -- kept for the record.

The history this file pins, because it is the evidence for the change that
replaced it:

  * 513 graded cells produced zero calls to verify_pde_consistency, so the ask
    was moved to LEAD the converged couple() reply.
  * Over every recorded cell after that: served 14 times in 10 cells, called
    twice, in one cell.
  * The ask was then made a COUNT carrying its own evidence ("you have coupled
    n levels and checked none of them"). Round 63 measured it: fired 7 times
    across 3 cells, called ZERO times.

Three wordings, three failures. couple() now takes the four strings the agent
already has and runs the check itself, which is what
test_couple_runs_the_equation_check_itself.py pins.

What survives here is the piece that change still depends on: the verification
tool must record itself in the session journal, because openPASO was otherwise
blind to whether any level had been checked at all.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.session_journal import get_journal  # noqa: E402


def test_the_verification_tool_records_itself_like_couple_does():
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert 'record("tool_call", "couple"' in src, "couple no longer records itself"
    assert 'record("tool_call", "verify_pde_consistency"' in src, (
        "verify_pde_consistency must record itself or openPASO cannot see whether "
        "any level was ever checked")


def test_the_journal_can_still_tell_the_two_apart():
    j = get_journal()
    j.events.clear()
    j.record("tool_call", "couple", solver="general", physics="coupling")
    j.record("tool_call", "verify_pde_consistency", solver="general", physics="coupling")
    names = [e.tool_name for e in j.events]
    assert names.count("couple") == 1 and names.count("verify_pde_consistency") == 1
    j.events.clear()


def test_the_invitation_wording_is_gone_from_the_reply():
    """It was measured three times and failed three times; it must not linger."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "CHECKED NONE OF THEM AGAINST ITS OWN" not in src, (
        "the counting escalation was superseded by running the check")
    assert "verify_pde_consistency(solution_files='<this level's" not in src, (
        "the paragraph telling the agent to call the tool itself is superseded")
