"""A signal is not a diagnostic, so openPASO has to name the cause.

MEASURED on C2_27b_MCP_seed76, which lost its whole run to this. The agent
wrote a 2-D Scalar_Transport deck with `E: 0` and `NODE n DLINE 0` — zero-based
design entity ids, the natural thing coming from Python. 4C died with

    Signal: Segmentation fault (11)
    Signal code: Address not mapped (1)                        exit 139

and printed no error, no line number and no mention of conditions. Because the
crash lands during "Read/generate conditions" it reads as a problem with the
condition's CONTENT, so the run tried five repairs — different topology
configurations, DESIGN POINT DIRICH instead of DESIGN LINE, DESIGN LINE DIRICH
instead of DESIGN LINE TRANSPORT DIRICH, TYPE Std against TYPE AdvReac — and
every one crashed identically. It then reported, honestly and wrongly, that 4C
could not run the problem at all.

The cause was one digit. Verified by running that agent's own deck twice with
nothing changed but the id:

    E: 0 / DLINE 0  ->  Segmentation fault (11), exit 139
    E: 1 / DLINE 1  ->  "processor 0 finished normally", exit 0

Both decks and the real crash log are fixtures here, so this test fails if the
grammar stops warning or the diagnostic stops explaining.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIX = ROOT / "tests" / "fixtures" / "fourc_failure_logs"
FOURC = Path("/home/user/4C/build/4C")


def test_the_diagnostic_names_the_cause_of_the_real_crash():
    from backends.fourc.backend import _fourc_diagnostic
    log = (FIX / "L5_zero_based_design_id_segfault.log").read_text()
    out = _fourc_diagnostic(log, "")
    assert "ZERO-BASED DESIGN ENTITY ID" in out, (
        "the agent sees a bare segmentation fault and no explanation"
    )
    assert "ONE-based" in out


def test_the_diagnostic_warns_against_the_five_repairs_that_do_not_work():
    """Each of these leaves the fault in place and crashes identically."""
    from backends.fourc.backend import _fourc_diagnostic
    out = _fourc_diagnostic(
        (FIX / "L5_zero_based_design_id_segfault.log").read_text(), "")
    low = out.lower()
    assert "before rewriting section names" in low
    assert "element type" in low or "element `type`" in low


def test_the_served_grammar_carries_the_trap():
    from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR as G
    assert "A 0 IS A SEGMENTATION FAULT" in G
    assert "exit 139" in G and "finished normally" in G, (
        "the trap must carry the measured evidence, not just an assertion"
    )


def test_a_clean_run_gets_no_segfault_advice():
    """The advice must not fire on a log that never crashed."""
    from backends.fourc.backend import _fourc_diagnostic
    out = _fourc_diagnostic("Read/generate conditions ... 0.002 secs\n"
                            "processor 0 finished normally\n", "")
    assert "ZERO-BASED DESIGN ENTITY ID" not in out


@pytest.mark.skipif(not FOURC.exists(), reason="4C binary not installed here")
@pytest.mark.parametrize("deck,expect_zero_exit", [
    ("L5_zero_based_design_id.4C.yaml", False),
    ("L5_one_based_design_id.4C.yaml", True),
])
def test_the_two_decks_still_behave_as_measured(tmp_path, deck, expect_zero_exit):
    """The claim is about this build; re-measure it rather than trust the note."""
    import os
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")
    src = FIX / deck
    run = tmp_path / deck
    run.write_text(src.read_text())
    p = subprocess.run([str(FOURC), str(run), str(tmp_path / "out")],
                       capture_output=True, text=True, timeout=300, env=env,
                       cwd=tmp_path)
    if expect_zero_exit:
        assert p.returncode == 0, (
            f"the one-based deck no longer runs on this build: {p.stdout[-400:]}")
        assert "finished normally" in (p.stdout + p.stderr)
    else:
        assert p.returncode != 0, (
            "the zero-based deck no longer crashes; if 4C was fixed, the trap "
            "text and this test should be retired together")
