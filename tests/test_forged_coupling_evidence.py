"""A forged coupled submission must not grade as a coupled success.

The campaign's only coupled CORRECT, C7_27b_BARE_seed2, was a Python monolith.
Its own IMPLEMENTATION_NOTES.txt says so: "The implementation uses a
Python-based FEM solver rather than FEBio/deal.II" and "The coupling iterations
shown are simulated based on the monolithic solution, rather than actual
separate solves."

It passed because three guards each missed by a hair:

  1. Every run_level<k>_<side>.log held one hand-typed line, `NDOF = <n>`. The
     canonical contract line is code-agnostic on purpose (so a quiet honest
     dolfinx run is not condemned for its print style), so the same four files
     were certified as "structured febio solver output" AND "structured dealii
     solver output" — which cannot both be true.

  2. The guard against exactly that, in assess(), was written
     `all(len(v) == 1 for v in seen.values())` — it fired only when each code
     had exactly ONE file. This run gave each code four, so it never fired, and
     even had it fired it only appends a note that cannot change a verdict.

  3. coupling_evidence() called the residual history "the artefact a monolithic
     solve cannot produce at all". Nobody runs a monolithic solve to make one:
     the agent writes 0.99999 * 0.5^k into a CSV. It converges beautifully and
     passed every shape check there was.

These tests use synthetic histories, not the run tree, so they keep meaning
after the runs are archived.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.blind_eval.evidence import (  # noqa: E402
    SYNTHETIC_RATIO_CV,
    EvidenceItem,
    _decay_ratio_cv,
    assess,
    coupling_evidence,
)


def _write_history(work: Path, level: int, vals: list) -> None:
    (work / f"residual_level{level}.csv").write_text(
        "iteration,interface_residual\n"
        + "".join(f"{i + 1},{v!r}\n" for i, v in enumerate(vals))
    )


def _forged(n: int = 21, rate: float = 0.5, start: float = 0.9999901187215444):
    """Exactly what C7_27b_BARE_seed2 wrote: start * rate**k."""
    return [start * rate**k for k in range(n)]


def _honest():
    """A real Dirichlet-Neumann history: converging, but the rate wanders.

    Taken from the shape of the genuine runs in the tree, where the ratio's
    coefficient of variation runs from ~1e-2 up to O(10).
    """
    return [1.0, 0.42, 0.23, 0.081, 0.049, 0.013, 0.0071,
            0.0019, 4.4e-4, 1.7e-4, 3.1e-5, 8.8e-7]


class TestSyntheticDecayDetector(unittest.TestCase):
    def test_a_closed_form_sequence_is_contradicted(self):
        with _tmp() as work:
            for lvl in (1, 2, 3):
                _write_history(work, lvl, _forged())
            got = coupling_evidence(work)
        self.assertEqual(got["verdict"], "CONTRADICTED")
        self.assertIn("constant ratio", got["detail"])

    def test_an_honest_history_still_proves_coupling(self):
        """The rule must not cost a real partitioned iteration its evidence."""
        with _tmp() as work:
            for lvl in (1, 2, 3):
                _write_history(work, lvl, _honest())
            got = coupling_evidence(work)
        self.assertEqual(got["verdict"], "PROVEN", got["detail"])

    def test_the_measured_separation_still_holds(self):
        """Forged ~1e-12, honest >1e-2, threshold in the empty valley."""
        forged_cv = _decay_ratio_cv(_forged())
        honest_cv = _decay_ratio_cv(_honest())
        self.assertLess(forged_cv, SYNTHETIC_RATIO_CV)
        self.assertGreater(honest_cv, SYNTHETIC_RATIO_CV)
        self.assertGreater(
            honest_cv / max(forged_cv, 1e-300),
            1e3,
            "the two populations are no longer separated by orders of "
            "magnitude; re-measure the threshold against the run tree before "
            "trusting this gate",
        )

    def test_a_short_history_says_nothing_rather_than_passing(self):
        """Silence, not a pass, when there is too little to judge."""
        self.assertIsNone(_decay_ratio_cv([1.0, 0.5, 0.25]))

    def test_a_non_positive_residual_says_nothing(self):
        self.assertIsNone(_decay_ratio_cv([1.0, 0.5, 0.0, -0.1, 0.2, 0.1, 0.05]))


class TestSharedEvidenceCheck(unittest.TestCase):
    """The guard that could not reach the case it was built for."""

    def _report(self, files_a, files_b):
        rep_files = {"febio": files_a, "dealii": files_b}

        def fake(work, code):
            return EvidenceItem(code, "PROVEN", list(rep_files[code]), [], "x")

        import src.blind_eval.evidence as ev

        real, ev.code_evidence = ev.code_evidence, fake
        try:
            with _tmp() as work:
                for lvl in (1, 2, 3):
                    _write_history(work, lvl, _honest())
                return assess(work, ["febio", "dealii"], coupled=True)
        finally:
            ev.code_evidence = real

    def test_four_identical_files_are_now_flagged(self):
        """The C7 shape: same four files prove both codes. Was silent."""
        four = ["level1/run_level1_A.log", "level1/run_level1_B.log",
                "level2/run_level2_A.log", "level2/run_level2_B.log"]
        notes = self._report(four, list(four)).notes
        self.assertTrue(
            any("proving one code also proves the other" in n for n in notes),
            f"identical multi-file evidence went unremarked: {notes}",
        )

    def test_one_identical_file_is_still_flagged(self):
        """The case the original narrow test did catch — do not lose it."""
        notes = self._report(["run.log"], ["run.log"]).notes
        self.assertTrue(
            any("proving one code also proves the other" in n for n in notes)
        )

    def test_genuinely_distinct_evidence_is_not_flagged(self):
        notes = self._report(["a_febio.log"], ["b_dealii.log"]).notes
        self.assertFalse(
            any("proving one code also proves the other" in n for n in notes),
            f"distinct per-code evidence was wrongly flagged: {notes}",
        )


import contextlib  # noqa: E402
import tempfile  # noqa: E402


@contextlib.contextmanager
def _tmp():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


if __name__ == "__main__":
    unittest.main()


class TestLeadingNaNIsOurOwnBookkeeping(unittest.TestCase):
    """openPASO's driver writes history[0] = NaN; copying it must not read as forgery.

    Iteration 1 has no previous iterate to difference against, so the driver
    records NaN. An agent that copies `history` verbatim into the graded
    residual_level<k>.csv writes "1,nan", and a non-finite residual used to be
    read as a history that could not have come from a real iteration —
    FABRICATED_NO_RUN, the forgery verdict, for faithfully copying a number
    openPASO handed it.

    Measured across the campaign: 20 runs wrote a NaN into a residual file, 13
    were graded fabrications, and 11 of those 13 were openPASO-arm runs. The
    served text now tells agents not to write it; this keeps the label honest
    for the runs that already did.

    A NaN anywhere ELSE stays fatal: mid-history it means an iteration
    produced no number, which no honest run does.
    """

    def _seq(self, n=14, seed=7):
        import random
        r = random.Random(seed)
        v, out = 1.0, []
        for _ in range(n):
            v *= r.uniform(0.25, 0.55)
            out.append(v)
        return out

    def test_a_leading_nan_does_not_make_it_a_forgery(self):
        vals = self._seq()
        with _tmp() as work:
            for lvl in (1, 2, 3):
                (work / f"residual_level{lvl}.csv").write_text(
                    "iteration,interface_residual\n1,nan\n"
                    + "".join(f"{i + 2},{v!r}\n" for i, v in enumerate(vals)))
            got = coupling_evidence(work)
        self.assertEqual(got["verdict"], "PROVEN", got["detail"])
        self.assertNotIn("non-finite", got["detail"])

    def test_a_nan_mid_history_is_still_fatal(self):
        vals = self._seq()
        with _tmp() as work:
            for lvl in (1, 2, 3):
                rows = [f"{i + 1},{v!r}" for i, v in enumerate(vals[:5])]
                rows.append(f"{6},nan")
                rows += [f"{i + 7},{v!r}" for i, v in enumerate(vals[5:])]
                (work / f"residual_level{lvl}.csv").write_text(
                    "iteration,interface_residual\n" + "\n".join(rows) + "\n")
            got = coupling_evidence(work)
        self.assertEqual(got["verdict"], "CONTRADICTED")
        self.assertIn("non-finite", got["detail"])

    def test_an_all_nan_history_is_not_quietly_accepted(self):
        with _tmp() as work:
            for lvl in (1, 2, 3):
                (work / f"residual_level{lvl}.csv").write_text(
                    "iteration,interface_residual\n" +
                    "".join(f"{i + 1},nan\n" for i in range(6)))
            got = coupling_evidence(work)
        self.assertEqual(got["verdict"], "CONTRADICTED", got["detail"])


class TestIterationCountIsNotForgeryEvidence(unittest.TestCase):
    """An off-by-one in a reported count must not read as an invented history.

    The COUPLING_ITERATIONS mismatch appended to `problems`, so it flipped the
    whole coupling verdict to CONTRADICTED, which evidence2.py maps to
    FABRICATED_NO_RUN -- the forgery label -- for a miscount.

    C2_27b_MCP_seed15 was exactly that: its SOLE complaint was "claims 7 but
    the finest-level history has 6 rows", while its history ran 6.753e-01 ->
    6.474e-07 with ratio CV 1.427 (forgery threshold 1e-5), both codes proven,
    interface satisfied, complete level set. Regraded after the fix: CORRECT,
    order 1.9547, r^2 1.0. It is the second coupled success of the campaign and
    was recorded as a fabrication.

    The discrepancy is still reported -- it is a real inconsistency in the
    submission -- but as a note, not as evidence of invention.
    """

    def _real_history(self):
        return [6.753e-01, 3.85e-03, 3.34e-03, 7.19e-05, 1.50e-05, 6.474e-07]

    def test_a_miscount_over_a_genuine_history_stays_proven(self):
        with _tmp() as work:
            for lvl in (1, 2, 3):
                _write_history(work, lvl, self._real_history())
            got = coupling_evidence(work, claimed_iterations=7)
        self.assertEqual(got["verdict"], "PROVEN", got["detail"])

    def test_the_discrepancy_is_still_recorded(self):
        with _tmp() as work:
            for lvl in (1, 2, 3):
                _write_history(work, lvl, self._real_history())
            got = coupling_evidence(work, claimed_iterations=7)
        self.assertIn("COUPLING_ITERATIONS=7", got["iteration_count_note"])
        self.assertIn("6 rows", got["iteration_count_note"])

    def test_a_matching_count_leaves_no_note(self):
        with _tmp() as work:
            for lvl in (1, 2, 3):
                _write_history(work, lvl, self._real_history())
            got = coupling_evidence(work, claimed_iterations=6)
        self.assertEqual(got["iteration_count_note"], "")

    def test_a_forged_history_is_still_contradicted_whatever_the_count(self):
        """The fix must not weaken the check that actually finds forgeries."""
        with _tmp() as work:
            for lvl in (1, 2, 3):
                _write_history(work, lvl, _forged())
            got = coupling_evidence(work, claimed_iterations=21)
        self.assertEqual(got["verdict"], "CONTRADICTED")
        self.assertIn("constant ratio", got["detail"])
