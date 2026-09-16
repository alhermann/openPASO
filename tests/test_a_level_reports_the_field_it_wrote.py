"""`couple_levels` reports the size of the field each side just exported.

WHY THIS IS REPORTED RATHER THAN OFFERED. The reply already ended by telling the
agent to call `audit_results`. Measured over the campaign's graded coupled
record, a self-check tool was called in **40 of 217** cells with files. And a
live full-task C9 run on this fork wrote a field of LITERAL ZERO at all three
levels, reported `INTERFACE_RESIDUAL = 6.8e-11` as evidence of success, and
called none of the three self-checks it was holding. `audit_results` fires six
NEAR-ZERO findings on that workspace in a single call.

A zero field is the perfect coupled liar: two sides exchanging nothing cannot
disagree, so the iteration converges at once and every self-consistency measure
reads as success.

This stays inside the no-served-solve rule: it opens a file the AGENT wrote and
reports a number from it. No source term, no reference, no answer.
"""
import csv
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


@pytest.fixture(scope="module")
def peak():
    from tools.consolidated import _level_field_peak
    return _level_field_peak


def _write(path: Path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "u"])
        for i, v in enumerate(values):
            w.writerow([i * 0.1, 0.0, v])


def test_it_reports_the_peak_of_the_file_that_side_wrote(peak, tmp_path):
    _write(tmp_path / "A" / "field_level1.csv", [0.0, 1.5e-3, -4.0e-3])
    out = peak([{"name": "A", "work_dir": str(tmp_path / "A")}], 1)
    assert out["A"]["peak_abs_value"] == pytest.approx(4.0e-3)
    assert out["A"]["points"] == 3


def test_a_literal_zero_field_is_reported_as_near_zero(peak, tmp_path):
    """The exact case measured: 4C decks wiring VAL: [0.0] deliver 0.0 everywhere."""
    _write(tmp_path / "B" / "field_level2.csv", [0.0, 0.0, 0.0])
    out = peak([{"name": "B", "work_dir": str(tmp_path / "B")}], 2)
    assert out["B"]["peak_abs_value"] == 0.0, (
        "an exactly zero field must be reported, not filtered out by a "
        "'0 < x' test -- that is how the three 4C runs slipped through")


def test_it_says_nothing_when_the_side_wrote_nothing(peak, tmp_path):
    assert peak([{"name": "A", "work_dir": str(tmp_path)}], 1) == {}
    assert peak([], 1) == {}
    assert peak([{"name": "A"}], 1) == {}


def test_a_header_only_file_is_not_counted(peak, tmp_path):
    path = tmp_path / "A" / "field_level1.csv"
    path.parent.mkdir(parents=True)
    path.write_text("x,y,u\n")
    assert peak([{"name": "A", "work_dir": str(tmp_path / "A")}], 1) == {}


def test_the_reply_carries_the_finding_not_an_invitation():
    """The wording matters: the agent is told what its file says, not asked."""
    text = (REPO / "src" / "tools" / "consolidated.py").read_text()
    assert "field_finding" in text and "NEAR-ZERO FIELD on side(s)" in text
    assert "two sides exchanging nothing cannot disagree" in text, (
        "the finding should say why a zero field converges so convincingly")
