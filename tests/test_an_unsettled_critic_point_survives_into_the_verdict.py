"""A review loop that is allowed to end must not end by hiding the disagreement.

MEASURED on a first real user run of the browser interface: the critic rejected the setup four
times, two rounds demanding the opposite of each other on the same point (the force sign
convention), ten of thirteen minutes went into review, and no solver was ever called. The served
rule now ends the loop after two blocking rounds and asks the critic to submit what is still
disputed as lines beginning "UNRESOLVED:". That is only honest if those lines reach the reader, so
the gate names them in its verdict instead of reporting a clean approval.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.instructions import INSTRUCTIONS  # noqa: E402
from tools.consolidated import _critic_state, _open_concerns  # noqa: E402
from core.critic_gate import CriticRegistry  # noqa: E402
from tools.consolidated import review_digest  # noqa: E402


class _Rec:
    def __init__(self, findings):
        self.findings = findings


def test_open_points_are_named_and_counted():
    note = _open_concerns(_Rec("checked units and mesh\n"
                               "UNRESOLVED: the force sign convention is disputed\n"
                               "UNRESOLVED: the outlet condition may be too short\n"))
    assert "2 point(s) open" in note
    assert "force sign convention" in note and "outlet condition" in note


def test_a_review_without_open_points_reads_clean():
    assert _open_concerns(_Rec("checked units, mesh and boundary conditions; no defect found")) == ""
    assert _open_concerns(None) == ""


def test_a_submitted_review_carries_its_open_points_into_the_verdict(monkeypatch):
    import tools.consolidated as C
    reg = CriticRegistry()
    monkeypatch.setattr(C, "_CRITIC_REGISTRY", reg)
    setup = "MESH: 32\nRE: 100\n"
    reg.submit_review(solver="fenics", findings=(
        "Checked units, boundary conditions and the mesh against the DFG benchmark.\n"
        "UNRESOLVED: the force sign convention, demanded both ways in earlier rounds"),
        digest=review_digest("fenics", setup))
    ok, note = _critic_state("fenics", setup)
    assert ok is True
    assert "1 point(s) open" in note and "force sign convention" in note


def test_the_served_rule_states_what_blocks_and_when_to_stop():
    assert "AT MOST TWO blocking rounds" in INSTRUCTIONS
    assert "UNRESOLVED:" in INSTRUCTIONS
    assert "CONCRETE, CHECKABLE defect" in INSTRUCTIONS
