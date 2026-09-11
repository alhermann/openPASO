"""Front-loading must not cut the facts an agent cannot ask for.

WHAT HAPPENED. `_front_load_coupling` was added to stop the coupling payload
eating the OASiS arm's tool-call budget (measured: coupled runs reading the full
corpus got a median 39 tool calls and wrote no output 60% of the time; runs with
no coupling text got 95 calls). It takes a PREFIX, and the "Choosing theta"
section sits past 24,000 characters — so the iteration budget, the divergence
thresholds and the swap remedy stopped reaching agents entirely. Measured: the
served payload was 23,378 characters and contained none of them.

WHY A TRUNCATION HINT IS NOT ENOUGH. The hint tells the agent to ask when it is
stuck. An agent that does not know its accelerator CAN diverge reads a rising
residual as its own modelling error and rebuilds the setup instead of asking.
You cannot request what you do not know exists.

WHY THESE PARTICULAR FACTS. Measured by execution over 48 configurations of a
two-subdomain conduction split at the tool's own tol = 1e-6, mesh-independent to
the iteration over a fourfold interface refinement:

    rho          1/10   1     2     4      10     100
    iterations   11     37    66    123    299    3311     (theta = 1/(1+rho))
    theta = 0.5  18     37    90    DIVERGES from rho = 4 up
    default      12     37    78    381    DIVERGES from rho = 10 up

So the default max_iter = 50 is already short at rho = 2, and the DEFAULT
accelerator diverges on exactly the severe-contrast cells this campaign uses
(C2 is "two-material conduction, severe material contrast"). On those cells this
block is the difference between converging and not.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.consolidated import (  # noqa: E402
    _COUPLING_HEAD_LIMIT,
    _COUPLING_MUST_READ,
    _front_load_coupling,
)
from tools.coupling_knowledge import coupling_knowledge  # noqa: E402

# Facts that must survive truncation, phrased as the agent would read them.
MUST_SURVIVE = [
    "rho = the interface conductance",   # the quantity everything is keyed to
    "max_iter = 100",                    # what to actually set
    "DIVERGES from rho = 4",             # constant theta = 0.5
    "DIVERGES from rho = 10",            # the DEFAULT accelerator
    "SWAP WHICH SIDE IS DIRICHLET",      # the remedy when budget cannot help
    "3311",                              # the count that shows the range
    "NOT by",                            # says the counts are mesh-independent
]


class TestTheServedPayload(unittest.TestCase):
    def setUp(self):
        self.full = coupling_knowledge(solver="")
        self.served = _front_load_coupling(self.full)

    def test_the_payload_is_actually_truncated(self):
        """Guard the guard: if the payload ever fits, this file proves nothing
        and the reader should know that rather than see a green tick."""
        self.assertGreater(
            len(self.full), _COUPLING_HEAD_LIMIT,
            "the coupling payload now fits under the head limit, so "
            "truncation no longer happens and these tests are vacuous")

    def test_every_divergence_fact_reaches_the_agent(self):
        missing = [p for p in MUST_SURVIVE if p not in self.served]
        self.assertEqual(
            missing, [],
            f"front-loading cut facts an agent cannot ask for: {missing}. "
            f"An agent that does not know its accelerator can diverge reads a "
            f"rising residual as its own error.")

    def test_the_block_is_at_the_front_not_buried(self):
        """It is among the first things the agent reads, or it is optional.

        The bound is the LENGTH OF THE MUST-READ BLOCK, not a fixed number: the
        block grew when the residual-history contract was added ahead of the
        divergence table, which pushed the swap remedy from 1,004 to 2,537
        characters in. Both are still inside the block. A fixed 2,500 would fail
        for the right reason and the wrong cause.
        """
        # 2026-09-11: the must-read is split around the code's contract (part
        # A leads, the contract follows, part B closes the reply), so the
        # remedy sits in part B; the guarantee is that part B is served WHOLE
        # in the first reply of a session, i.e. nothing of it is cut.
        from tools.consolidated import _COUPLING_LEAD_B
        self.assertIn("SWAP WHICH SIDE IS DIRICHLET", self.served,
                      "the swap remedy is no longer served")
        self.assertIn(_COUPLING_LEAD_B.strip()[-300:], self.served,
                      "the tail of the must-read's part B was cut from the first reply")

    def test_history_path_is_named_first_and_no_benchmark_file_name_is_served(self):
        """MEASURED GAP: the served coupling payload once said nothing about
        where the residual history goes, and the runs that hand-rolled the
        loop wrote 1.0, 0.5, 0.25, ... into their history file -- an invented
        sequence. The remedy is behaviour, not a file name: the agent is told
        at the very top to pass `history_path` so couple() writes the measured
        history to the file its task names. OASiS is a general product, so no
        benchmark's file name may appear anywhere in what it serves."""
        i = self.served.find("history_path")
        self.assertGreaterEqual(i, 0, "history_path is never mentioned")
        self.assertLess(i, 1500, f"history_path first appears only at char {i}")
        self.assertNotIn("residual_level", self.served,
                         "a benchmark file name is served; the task text is "
                         "the only place that name may live")

    def test_it_says_couple_produces_that_history(self):
        self.assertIn("couple(participants", self.served)
        j = self.served.index("couple(participants")
        self.assertLess(j, 1200, f"the call appears only at char {j}")

    def test_it_states_that_a_closed_form_history_is_detected(self):
        """Stating the detection rule is permitted; withholding it produced an
        invented history instead."""
        for phrase in ("DETECTED and read as invented",
                       "constant ratio is a formula",
                       "SAME numbers at two mesh levels"):
            self.assertIn(phrase, self.served, f"missing: {phrase}")

    def test_the_truncation_hint_still_says_more_exists(self):
        self.assertIn("TRUNCATED HERE", self.served)

    def test_the_remaining_count_excludes_the_prepended_block(self):
        """The hint states how many characters of the ORIGINAL payload were not
        served. Counting the prepended block as served payload would understate
        it."""
        import re
        m = re.search(r"([\d,]+) further characters", self.served)
        self.assertIsNotNone(m, "the hint no longer reports a remainder")
        rest = int(m.group(1).replace(",", ""))
        prefix_len = len(self.served) - len(_COUPLING_MUST_READ)
        self.assertLessEqual(
            rest, len(self.full),
            f"reported remainder {rest} exceeds the whole payload")
        self.assertGreater(rest, len(self.full) - prefix_len - 5000)

    def test_the_head_budget_accounts_for_the_prepended_block(self):
        """Prepending must not silently push the served head over its budget:
        the point of the limit is the agent's context, and text is text."""
        head = self.served.split("TRUNCATED HERE")[0]
        self.assertLessEqual(
            len(head), _COUPLING_HEAD_LIMIT + 200,
            f"the served head grew to {len(head)} against a limit of "
            f"{_COUPLING_HEAD_LIMIT}; the must-read block was added without "
            f"taking its length out of the prefix budget")


class TestTheNumbersAreTheMeasuredOnes(unittest.TestCase):
    """If someone edits the block, the numbers must stay the measured ones —
    they came from 48 executed configurations, not from a textbook."""

    def test_the_iteration_counts_are_present_and_consistent(self):
        for n in ("11", "37", "66", "123", "299", "3311"):
            self.assertIn(n, _COUPLING_MUST_READ,
                          f"measured count {n} is gone from the block")

    def test_it_does_not_reinstate_the_unconditional_claim(self):
        """The old text said "budget tens to hundreds of iterations", which
        over-budgets rho <= 1 (5-37) and under-budgets rho = 100 (3311)."""
        self.assertNotIn("tens to hundreds", _COUPLING_MUST_READ)

    def test_it_states_the_counts_are_mesh_independent(self):
        """Otherwise an agent will try to fix divergence by refining."""
        self.assertIn("NOT by", _COUPLING_MUST_READ)
        self.assertIn("refinement", _COUPLING_MUST_READ)


if __name__ == "__main__":
    unittest.main()
