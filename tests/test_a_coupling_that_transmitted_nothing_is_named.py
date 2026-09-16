"""A converged coupling that exchanged nothing must not pass in silence.

THE HOLE. Both live content checks at `couple()`'s corrective hook abstain on
exactly this input: `flux_cancellation_finding` returns None when its
denominator is zero, and `field_continuity_finding` does the same, deferring in
a comment to "the near-zero check" -- which reads per-level CSVs that DO NOT
EXIST YET when couple() runs. So a pair that transmitted nothing passed every
live check.

MEASURED on a coupled run of this fork: three levels, both codes proven, the
partitioned iteration converged in 7 steps to an interface residual of 6.8e-11,
and the traction columns read -3.04e-18 and 2.17e-19. Two sides exchanging
nothing cannot disagree, so the iteration converges at once and every
self-consistency measure the run reports about itself reads as success.

WHY A FINDING AND NOT A REFUSAL, and why it lives in the driver's reply rather
than in a participant's export self-check: that check ends the coupling when it
fires, and on iteration 1 `imports.json` is `{}` by contract, so a Neumann side
legitimately recovers ~0 there. A fatal check must not fire on an input openPASO
cannot interpret. A reporting one must.

ADMISSION over the graded record, on cells carrying a matched interface pair:
0 of 33 CORRECT; 12 elsewhere (3 malformed, 7 honest-incomplete, 1 unphysical,
1 failed).
"""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PTS = [[0.625, i * 0.1] for i in range(6)]
LIVE = [4.87e-7] * 6


def _export(values, fluxes):
    return {"coordinates": PTS,
            "values": [[v] for v in values],
            "normal_fluxes": [[q] for q in fluxes]}


@pytest.fixture(scope="module")
def finding():
    from tools.result_audit import exchange_carried_nothing_finding
    return exchange_carried_nothing_finding


def test_the_measured_shape_is_named(finding):
    """Displacements real, tractions at round-off and cancelling."""
    got = finding(_export(LIVE, [-3.0e-18] * 6),
                  _export(LIVE, [3.0e-18] * 6), "A", "B")
    assert got, "a dead traction channel under a live displacement must be named"
    assert "TRANSMITTED NOTHING" in got["finding"]
    assert "interface flux" in got["finding"]
    assert "1e-11" in got["finding"], (
        "it should say what the excellent-looking residual actually means")


def test_a_wholly_dead_exchange_is_named(finding):
    got = finding(_export([0.0] * 6, [0.0] * 6),
                  _export([0.0] * 6, [0.0] * 6), "A", "B")
    assert got and "interface value" in got["finding"]


def test_a_healthy_coupling_is_silent(finding):
    assert finding(_export(LIVE, [-0.5] * 6),
                   _export(LIVE, [0.5] * 6), "A", "B") is None


def test_one_dead_side_is_a_different_defect_and_is_left_alone(finding):
    """One side silent against a live partner is the ratio check's business.

    Naming it here too would double-report, and the two have different fixes.
    """
    assert finding(_export([0.0] * 6, [0.0] * 6),
                   _export(LIVE, [0.5] * 6), "A", "B") is None


def test_absent_channels_are_not_reported_as_zero(finding):
    """No `normal_fluxes` key at all is a different defect, said elsewhere."""
    a = {"coordinates": PTS, "values": [[v] for v in LIVE]}
    b = {"coordinates": PTS, "values": [[v] for v in LIVE]}
    assert finding(a, b, "A", "B") is None


def test_it_never_asserts_the_subdomain_should_be_driven(finding):
    """openPASO cannot know that, and must not imply it."""
    got = finding(_export([0.0] * 6, [0.0] * 6),
                  _export([0.0] * 6, [0.0] * 6), "A", "B")
    assert "genuinely" in got["finding"] and "undriven" in got["finding"]


def test_it_runs_before_the_checks_that_abstain_on_its_input():
    text = (REPO / "src" / "tools" / "consolidated.py").read_text()
    order = text.index("exchange_carried_nothing_finding")
    assert order < text.index("_ra.flux_cancellation_finding"), (
        "the other two return None on this input, so this must run first")


def test_more_than_two_participants_reports_coverage_not_silence():
    """Silence from a checker is indistinguishable from a pass."""
    text = (REPO / "src" / "tools" / "consolidated.py").read_text()
    assert "interface content not checked" in text
    assert "COVERAGE, NOT A VERDICT" in text
