""""Never asked for the contract" must be said only when the journal can be seen.

MEASURED 2026-09-17. `contract_never_fetched` fired in 19 recorded runs, and in
all 19 the agent HAD called knowledge(topic='coupling', solver=<that code>) --
including a run that went on to grade correct. It was wrong every time.

The cause is a process boundary, not a threshold. The check runs in the harness
process (the write hook), and reads `get_journal()` there. The journal is filled
in the openPASO server, a SEPARATE process reached over stdio, and was saved only at
server shutdown. From the harness the journal is therefore always empty, so every
participant for a coupled code was told it had never asked -- the opposite of a
check: it nagged exactly the agents who had done the right thing.

Two rules fix it:
  1. the journal is persisted as it is written, one line per event, into the
     session directory that the campaign mounts at the run's work/.openpaso_sessions;
  2. the check reads that record, and when it can see NO record at all it stays
     silent. Absence of evidence is not evidence of absence.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import core.session_journal as sj  # noqa: E402
from tools import participant_lint as pl  # noqa: E402

DUNE_PARTICIPANT = '''import json
from pathlib import Path
from dune.grid import structuredGrid
from dune.fem.space import lagrange
grid = structuredGrid([0, 0], [1, 1], [8, 8])
space = lagrange(grid, order=1)
Path("exports.json").write_text(json.dumps({"values": [], "normal_fluxes": []}))
'''


def _participant(tmp_path):
    side = tmp_path / "work" / "side_A"
    side.mkdir(parents=True)
    p = side / "participant.py"
    p.write_text(DUNE_PARTICIPANT)
    return p


def _live(tmp_path, events):
    d = tmp_path / "work" / ".openpaso_sessions"
    d.mkdir(parents=True, exist_ok=True)
    (d / "session_test.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))


def _empty_in_process(monkeypatch):
    monkeypatch.setattr(sj, "_journal", sj.SessionJournal(), raising=False)


def test_the_participant_is_recognised_as_a_dune_participant():
    # Guard against the silence below being the lint failing to parse the file.
    assert "dune" in pl.backends_in(DUNE_PARTICIPANT)


def test_with_no_journal_visible_it_says_nothing(tmp_path, monkeypatch):
    # The exact harness situation behind the 19-of-19 false alarms.
    _empty_in_process(monkeypatch)
    p = _participant(tmp_path)
    assert pl.contract_never_fetched(DUNE_PARTICIPANT, near=p) == "", (
        "with no observable journal the check claimed the contract was never "
        "fetched -- it cannot know that")


def test_a_live_record_showing_the_fetch_keeps_it_silent(tmp_path, monkeypatch):
    _empty_in_process(monkeypatch)
    p = _participant(tmp_path)
    _live(tmp_path, [{"event_type": "knowledge_lookup", "tool_name": "knowledge",
                      "solver": "dune"}])
    assert pl.contract_never_fetched(DUNE_PARTICIPANT, near=p) == ""


def test_a_live_record_without_the_fetch_names_it(tmp_path, monkeypatch):
    _empty_in_process(monkeypatch)
    p = _participant(tmp_path)
    _live(tmp_path, [{"event_type": "knowledge_lookup", "tool_name": "knowledge",
                      "solver": "fourc"}])
    out = pl.contract_never_fetched(DUNE_PARTICIPANT, near=p)
    assert "dune" in out and "knowledge(topic='coupling', solver='dune')" in out


def test_the_served_message_cites_runs_not_cells(tmp_path, monkeypatch):
    _empty_in_process(monkeypatch)
    p = _participant(tmp_path)
    _live(tmp_path, [{"event_type": "tool_call", "tool_name": "discover"}])
    out = pl.contract_never_fetched(DUNE_PARTICIPANT, near=p)
    assert out and " cell" not in out, out


def test_recording_an_event_persists_it_as_it_happens(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPASO_JOURNAL_LIVE_DIR", str(tmp_path))
    j = sj.SessionJournal()
    j.record("knowledge_lookup", "knowledge", solver="dune")
    files = list(tmp_path.glob("session_*.jsonl"))
    assert len(files) == 1, "no live journal file was written"
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    assert rows and rows[-1]["event_type"] == "knowledge_lookup" and rows[-1]["solver"] == "dune"


def test_a_malformed_event_is_not_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENPASO_JOURNAL_LIVE_DIR", str(tmp_path))
    j = sj.SessionJournal()
    j.record("not_a_real_event_type", "x")
    assert not list(tmp_path.glob("session_*.jsonl"))
