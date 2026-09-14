"""`couple` must point the agent at the field that holds the cause.

Nothing exercised these branches before, which is how the crash case survived
two rounds of fixing the message around it.

THE THREE SITUATIONS, once collapsed into one message:

  * A CRASH. `converged` is False when a participant never ran at all, so a
    crash fell into the convergence branch and the agent was told to read
    `history` for "the residual per iteration" — an empty list. The field that
    holds the cause, `error`, was never named, and neither were the
    per-participant `returncodes`.

  * A GENUINE CONVERGENCE FAILURE, where `history` IS the right thing to read.

  * A CONVERGED RUN that failed a downstream silent-wrong check. Telling that
    agent its coupling "may not have converged" is the cheapest possible reason
    to stop, and it contradicts converged=True in the same payload.

Why this matters more than wording: the coupled arm's dominant failure is
stopping early, not timing out — 63% end HONEST_INCOMPLETE at a median 30% of
budget, with zero timeouts in 112 runs. A message that names no cause and points
at an empty array is a direct contributor.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.tools.consolidated import _couple_failure_reason  # noqa: E402


def _res(**kw):
    """A coupling result stub with the fields the reason logic reads."""
    base = dict(error=None, history=[1.0, 0.3, 0.02], converged=True,
                returncodes=None)
    base.update(kw)
    return SimpleNamespace(**base)


class TestCrash(unittest.TestCase):
    def test_a_crash_names_the_error_not_the_history(self):
        r = _res(error="participant B: ModuleNotFoundError: no module named "
                       "'dolfinx'",
                 history=[], converged=False, returncodes={"A": 0, "B": 1})
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("ModuleNotFoundError", msg,
                      "the error text is not in the message the agent reads")
        self.assertIn("do NOT read", msg)
        self.assertNotIn("see `history` for the residual per iteration", msg,
                         "an agent whose participant crashed is still being "
                         "sent to an empty history")

    def test_a_crash_reports_the_per_participant_exit_codes(self):
        r = _res(error="boom", history=[], converged=False,
                 returncodes={"A": 0, "B": 139})
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("139", msg,
                      "the exit code that identifies WHICH participant died "
                      "is not shown")

    def test_an_empty_history_with_no_error_text_still_says_so(self):
        """The worst case: nothing ran and nothing was captured. The message
        must still not claim a tolerance failure."""
        r = _res(error=None, history=[], converged=False)
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("No error text was captured", msg)
        self.assertIn("failed to run", msg)


class TestGenuineConvergenceFailure(unittest.TestCase):
    def test_a_real_stall_is_still_sent_to_the_history(self):
        """The fix must not cost the case where `history` IS the answer."""
        r = _res(error=None, history=[1.0, 0.9, 0.88, 0.87], converged=False)
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("did not reach the requested tolerance", msg)
        self.assertIn("history", msg)


class TestConvergedWithACaveat(unittest.TestCase):
    def test_a_converged_run_is_never_told_it_diverged(self):
        r = _res(converged=True, history=[1.0, 1e-3, 1e-7])
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("CONVERGED", msg)
        self.assertIn("NOT a failed run", msg)
        self.assertNotIn("did not reach the requested tolerance", msg)

    def test_a_clean_run_gets_no_reason_at_all(self):
        self.assertEqual(_couple_failure_reason(_res(), checks_ok=True), "")


class TestOrdering(unittest.TestCase):
    def test_a_crash_wins_over_the_convergence_branch(self):
        """Both conditions hold (error set AND converged False). The crash
        message must win — it is the one that names a cause."""
        r = _res(error="segfault in participant A", history=[], converged=False)
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("segfault", msg)

    def test_an_error_alongside_a_real_history_still_names_the_error(self):
        """A participant that died mid-iteration leaves a partial history. The
        agent needs the error, not just the residuals."""
        r = _res(error="participant B died at iteration 3",
                 history=[1.0, 0.4], converged=False)
        msg = _couple_failure_reason(r, checks_ok=False)
        self.assertIn("died at iteration 3", msg)


if __name__ == "__main__":
    unittest.main()


class TestAConvergedRunIsNeverToldBothThings(unittest.TestCase):
    """One reply carried 'report the numbers and the caveat' AND 'must NOT be
    reported as a result' in the same paragraph.

    MEASURED, C2_27b_MCP_seed1501: its couple call converged and failed a
    downstream silent-wrong check; the reason string routed on three words
    (balance/conserv/flux), contained none of them, and fell to the harsh
    branch -- so the attest appended "must NOT be reported as a result; revise
    the setup and re-run" directly after couple's own "This is a converged
    result with a caveat, NOT a failed run: report the numbers and the
    caveat." The agent obeyed the harsher imperative, exported a residual its
    own files contradict, and graded COMPLETED_UNPHYSICAL.
    """

    REAL = ("the coupling CONVERGED, and then failed one of openPASO's "
            "silent-wrong checks (see `validation`). This is a converged "
            "result with a caveat, NOT a failed run: report the numbers and "
            "the caveat.")

    def _stamp(self, reason):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from tools.consolidated import _stamp_verification
        r = {}
        _stamp_verification(r, evidence_ok=False, reason=reason)
        return r["verification"]

    def test_the_real_reason_gets_deliverables_first_and_no_contradiction(self):
        v = self._stamp(self.REAL)
        self.assertIn("WRITE YOUR DELIVERABLES FIRST", v)
        self.assertNotIn("must NOT be reported", v)

    def test_a_genuinely_unconverged_run_keeps_the_hard_stop(self):
        v = self._stamp("the solver did not converge "
                        "(residual 3e-2 > tol 1e-6)")
        self.assertIn("must NOT be reported", v)
        self.assertNotIn("WRITE YOUR DELIVERABLES FIRST", v)

    def test_an_unbound_claim_keeps_the_hard_stop(self):
        v = self._stamp("the result is not bound to a check-passing run")
        self.assertIn("must NOT be reported", v)

    def test_no_reply_ever_carries_both_imperatives(self):
        for reason in (self.REAL, "flux balance off by 100.8%",
                       "did not converge", "not bound to a run",
                       "silent-wrong check failed on the coarsest level"):
            v = self._stamp(reason)
            self.assertFalse("report the numbers and the caveat" in v
                             and "must NOT be reported" in v,
                             f"contradictory reply for: {reason!r}")
