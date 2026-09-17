"""The second code is where coupled runs die, and they never ask about it.

MEASURED over every C9/C10 work dir on disk. 26 cells stopped with exactly one
side exporting, and the silent side was side B in 16 of 17 -- DUNE 10,
scikit-fem 6. They were not out of budget when they started it: a median of 20
tool calls remained after the silent participant was first written, and they
spent them on 265 run_bash and 183 write_file calls between 17 cells, about 26
each of write-run-error-rewrite.

What they did not spend them on was asking. `knowledge` was called 11 times
across those 17 cells, and **17 of 22 never fetched the silent side's coupling
contract at all** (DUNE 10 of 11, scikit-fem 6 of 9).

NOTE ON THE ADMISSION TEST. Every other gate in this suite is required to be
silent on the 33 served contracts as static files. That test does not apply
here, because this gate's verdict depends on SESSION STATE rather than file
content. The equivalent test is below -- silent on every served contract once its
own code has been asked about.

CORRECTED 2026-09-17. This note used to say "with an empty journal every
participant is one whose contract was never fetched, which is correct". It is not:
an empty journal is what the harness sees when the journal lives in another
process, and on that premise the gate fired in 19 recorded runs and was wrong in
all 19. An EMPTY journal now means "cannot tell" and the gate stays silent; it
speaks only when a journal is visible and lacks the fetch. See
tests/test_a_check_that_cannot_see_the_journal_stays_silent.py.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.session_journal import get_journal  # noqa: E402
from tools.participant_lint import backends_in, contract_never_fetched  # noqa: E402

SERVED = sorted((ROOT / "data" / "coupling_participants").glob("*.py"))
DUNE_PARTICIPANT = ('import dune.fem\n'
                    'import json\n'
                    'json.dump({"values": v}, open("exports.json", "w"))\n')


@pytest.fixture(autouse=True)
def _clean_journal(tmp_path, monkeypatch):
    # recording now also appends a live line to disk; keep it out of the repo
    monkeypatch.setenv("OPENPASO_JOURNAL_LIVE_DIR", str(tmp_path))
    j = get_journal()
    j.events.clear()
    yield
    j.events.clear()


def test_it_names_a_code_this_run_never_asked_about():
    # A visible journal that shows this session working, but no fetch for dune.
    get_journal().record("tool_call", "discover")
    out = contract_never_fetched(DUNE_PARTICIPANT)
    assert "has not once" in out
    assert "knowledge(topic='coupling', solver='dune')" in out, (
        "it must name the door, not just the omission")


def test_asking_silences_it():
    get_journal().record("knowledge_lookup", "knowledge", solver="dune", physics="coupling")
    assert contract_never_fetched(DUNE_PARTICIPANT) == ""


def test_asking_about_a_DIFFERENT_code_does_not_silence_it():
    """The measured failure is the SECOND code; the first one having been asked
    about is exactly the state these cells were in."""
    get_journal().record("knowledge_lookup", "knowledge", solver="ngsolve", physics="coupling")
    assert contract_never_fetched(DUNE_PARTICIPANT) != ""


@pytest.mark.parametrize("path", SERVED, ids=lambda p: p.name)
def test_every_served_contract_is_silent_once_its_code_was_asked_about(path):
    src = path.read_text()
    j = get_journal()
    for c in backends_in(src):
        j.record("knowledge_lookup", "knowledge", solver=c, physics="coupling")
    assert contract_never_fetched(src) == "", path.name


def test_a_script_that_is_not_a_participant_is_left_alone():
    assert contract_never_fetched("import dune.fem\nprint('hello')\n") == ""


def test_a_code_with_no_served_contract_is_not_demanded():
    assert contract_never_fetched(
        'import sparta\nimport json\njson.dump(d, open("exports.json","w"))\n') == ""
