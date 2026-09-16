"""Every event_type written anywhere in src/ is one the journal accepts.

MEASURED. `consolidated.py` records the critic review as
`event_type="critic_review"`. That string is not in EVENT_TYPES, so
JournalEvent.__post_init__ raises ValueError, and SessionJournal.record()
catches every exception and returns None. The result is that no critic review
has ever been recorded: openPASO has no memory that a critic ran, and nothing
anywhere reports a problem.

The defect is invisible by construction -- a swallowed exception on a write path
nobody reads back -- so the only way to catch the next one is to compare the two
lists mechanically. The journal is openPASO's only memory of what the agent has
already done, and exactly one shipped check reads it
(participant_lint.contract_never_fetched), so a dropped event class is a
capability that quietly does not exist.
"""
import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.session_journal import EVENT_TYPES, JournalEvent  # noqa: E402


def _event_types_written_in_src():
    """Every literal event type handed to a .record() call under src/.

    record(event_type, tool_name, *, ...) takes it FIRST and POSITIONALLY, which
    is how every call site in the tree writes it; the keyword form is accepted
    here too so the test does not go blind if a call site is reworded.
    """
    written = {}
    for f in (ROOT / "src").rglob("*.py"):
        try:
            tree = ast.parse(f.read_text())
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            is_record = (isinstance(node.func, ast.Attribute)
                         and node.func.attr == "record")
            found = []
            if is_record and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                found.append(node.args[0])
            for kw in node.keywords:
                if kw.arg == "event_type" and isinstance(kw.value, ast.Constant) \
                        and isinstance(kw.value.value, str):
                    found.append(kw.value)
            for c in found:
                written.setdefault(c.value, set()).add(
                    f"{f.relative_to(ROOT)}:{c.lineno}")
    return written


def test_every_event_type_written_in_src_is_accepted_by_the_journal():
    written = _event_types_written_in_src()
    assert written, "found no event_type= call sites — re-point this test"
    unknown = sorted(set(written) - set(EVENT_TYPES))
    assert not unknown, (
        "these event types are written but would be rejected and the "
        "exception swallowed, so the events are silently lost: "
        + "; ".join(f"{t!r} at {sorted(written[t])}" for t in unknown))


def test_the_rejection_really_is_silent():
    # The reason this needs a test rather than a code review: the failure mode
    # produces no error anywhere the caller can see.
    from core.session_journal import SessionJournal
    j = SessionJournal()
    before = len(j.events)
    assert j.record(event_type="not_a_real_event_type", tool_name="x") is None
    assert len(j.events) == before, (
        "an invalid event was stored; if this ever starts holding, the "
        "silently-dropped-write failure mode is gone and this test can go too")


def test_a_critic_review_can_be_recorded():
    # The instance that motivated this: the one event class that records a
    # verification step having happened at all.
    ev = JournalEvent(timestamp=0.0, event_type="critic_review",
                      tool_name="submit_critic_review")
    assert ev.event_type == "critic_review"
