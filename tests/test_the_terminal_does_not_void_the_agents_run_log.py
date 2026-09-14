"""A program that writes without a newline glued its text onto the contract line.

The canonical contract line is matched with `^\\s*NDOF\\s*=\\s*(\\d+)\\s*$`, anchored
so that an agent's PROSE about its own solver cannot satisfy it. That anchoring
is right and it had an unmeasured cost: anything the TERMINAL writes without a
trailing newline lands in front of the agent's correct line, and `^` then cannot
match.

MEASURED in this tree, 19 logs (6 bare, 13 openPASO) state an NDOF the anchor
rejects:

  C2_27b_MCP_seed73   "Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 54"
                      an X11 warning printed with no newline. All three side-A
                      logs of the best coupled run in the campaign were voided,
                      and the run was then also charged with a mesh-sequence
                      complaint on side B alone.
  FB1_27b_MCP_seed2   an OSC terminal-title escape from FEBio itself, then
                      "NDOF = 196".

The repair removes what the terminal wrote before matching what the agent wrote,
rather than relaxing the anchor, which would let narration back in.

Separately `\\d{2,}` demanded two digits, so a coarse first level with 8 or 9
degrees of freedom failed for being small. Any integer is accepted now; whether
the count is PLAUSIBLE is decided by ndof_growth, which rejects "NDOF 0 ... is
not a run" — verified below, so the relaxation cannot turn a zero into proof.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "campaign3_blind"))

from blind_eval.evidence import CANONICAL_NDOF, strip_terminal_noise  # noqa: E402


def _ndof(text: str):
    m = CANONICAL_NDOF.search(strip_terminal_noise(text))
    return int(m.group(1)) if m else None


def test_the_x11_cookie_warning_no_longer_voids_the_line():
    assert _ndof("processor 0 finished normally\n"
                 "Invalid MIT-MAGIC-COOKIE-1 keyNDOF = 54\n") == 54


def test_an_osc_terminal_title_no_longer_voids_the_line():
    assert _ndof("\x1b]0;(100%) level1.feb - FEBio 4.12.0\x07NDOF = 196\n") == 196


def test_ansi_colour_codes_no_longer_void_the_line():
    assert _ndof("\x1b[32m\x1b[0mNDOF = 1024\n") == 1024


def test_a_single_digit_count_is_accepted():
    """A first level can legitimately have 8 or 9 degrees of freedom."""
    assert _ndof("NDOF = 9\n") == 9


def test_the_agents_prose_still_does_not_satisfy_the_contract():
    """The whole point of the anchor. Widening must not reach this."""
    for prose in ("I computed NDOF = 4225 for this mesh",
                  "the solver reports NDOF = 4225 at level 2",
                  "# NDOF = 4225",
                  "then NDOF = 4225 and we refine"):
        assert _ndof(prose) is None, f"narration satisfied the contract: {prose!r}"


def test_a_zero_count_is_not_proof_of_a_run():
    """Relaxing the digit rule must not let NDOF = 0 prove anything."""
    from grading.evidence2 import ndof_growth
    assert _ndof("NDOF = 0\n") == 0                  # it parses ...
    _, problems, _, _ = ndof_growth({"A": {1: 0, 2: 0, 3: 0}}, [1, 2, 3], 2, True)
    assert any("is not a run" in p for p in problems), (
        "a zero degree-of-freedom count must still be refused, in the place "
        "where plausibility is judged"
    )


def test_against_the_run_it_was_measured_on():
    work = ROOT / "campaign3_blind" / "runs" / "C2_27b_MCP_seed73" / "work"
    if not work.is_dir():
        import pytest
        pytest.skip("C2_27b_MCP_seed73 not present")
    got = {f.name: _ndof(f.read_text(errors="replace"))
           for f in sorted(work.glob("run_level*_A.log"))}
    assert all(v is not None for v in got.values()), (
        f"side A's logs are still unreadable to the grader: {got}"
    )
    assert sorted(got.values()) == [54, 187, 400]
