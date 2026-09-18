"""Silence is not assent, and openPASO must not claim an independence it cannot see.

MEASURED on a live run of the browser interface (record 5bd46d516b48): the model spawned a critic
sub-agent, the sub-agent returned a ZERO-LENGTH result, the model read that empty string as
approval, composed "CRITIC REVIEW ... VERDICT: APPROVED with minor notes" about its own script,
filed it through submit_critic_review, received a token and ran with it. The gate accepted it,
because the gate binds a review to a setup -- which it does correctly -- but cannot see who wrote
the review.

Two things follow, and both are in this repository rather than in the interface:
  * an empty sub-agent return is reported as silence, in both product paths;
  * the VERIFIED text says what the review part actually proves.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.instructions import INSTRUCTIONS  # noqa: E402


def test_both_product_paths_report_an_empty_sub_agent_as_silence():
    for rel in ("langgraph_eval/agent.py", "run_agent.py"):
        src = (ROOT / rel).read_text()
        assert "_EMPTY_SUBAGENT" in src, rel
        assert re.search(r"if not str\(report\)\.strip\(\)|report\.strip\(\) else _EMPTY_SUBAGENT", src), rel
    shared = (ROOT / "langgraph_eval" / "agent.py").read_text()
    i = shared.index("_EMPTY_SUBAGENT = ")
    message = shared[i:i + 400]
    assert "NOT approval" in message and "NOT a review" in message
    assert "Do not write its answer for it" in message


def test_the_verdict_does_not_claim_an_independent_critic():
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    assert "VERIFIED — an independent critic reviewed this exact setup" not in src, (
        "openPASO cannot see who wrote a submitted review, so it must not call it independent")
    assert "it cannot see who wrote it" in src
    assert "a review exists and matches what ran" in src


def test_the_served_rule_forbids_writing_the_review_yourself():
    assert "THE REVIEW MUST BE THE CRITIC'S, NOT YOURS" in INSTRUCTIONS
    assert "never compose the review yourself" in INSTRUCTIONS
    assert "has not approved anything" in INSTRUCTIONS
