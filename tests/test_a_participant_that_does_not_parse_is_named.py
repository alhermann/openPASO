"""A participant script that does not parse is named, even with no recognised solver import.

THE HOLE. `participant_findings` returned early when it recognised no backend
import, which made its own parse finding unreachable for exactly the files most
likely to lack one -- truncated or garbled scripts. Over the campaign record,
144 of 3873 participant scripts do not parse and 47 of those got no note; 34 of
the 47 carry the same garble, `(a)**2 + **(b)2`.

ADMISSION over every .py in the coupled record (7554 files): the note is new
on 47. None of the 48 participant scripts openPASO ships gets it. One
of the 47 is in a CORRECT cell (C3, seed 5733), and the note is TRUE there: its
first script does not parse, and the cell reached CORRECT through six later
scripts. Recorded rather than rounded to zero -- the bar exists to stop false
findings, and this one is not false.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tools.participant_lint import participant_findings  # noqa: E402

HANDSHAKE = 'import json\nimp = json.load(open("imports.json"))\n'


def test_a_garbled_participant_with_no_known_import_is_named_with_its_line():
    text = HANDSHAKE + "dist = (x - px)**2 + **(y - py)2\n" + 'json.dump({}, open("exports.json", "w"))\n'
    found = participant_findings(text)
    assert found and "does not parse" in found[0]
    assert "**(y - py)2" in found[0], "the offending line is what lets a small model find it"


def test_a_parseable_participant_with_no_known_import_stays_silent():
    text = HANDSHAKE + "dist = (x - px)**2 + (y - py)**2\n" + 'json.dump({}, open("exports.json", "w"))\n'
    assert participant_findings(text) == []


def test_an_unparseable_file_that_is_not_a_participant_is_not_judged():
    assert participant_findings("this is ( not python") == []
