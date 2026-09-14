"""Tests for solver smoke tests.

Each of these needs its solver. Whether that solver is here is not decided by
matching words in an error string -- this file used to skip on "No module
named", "not found", "GLIBC" or "ImportError" appearing in the message, which
is a guess about someone else's wording and silently passes over a solver that
is installed and wrong. conftest.backend_state asks openPASO's own
check_availability and then runs the shipped smoke test, and answers absent,
working or broken. Absent skips. Broken fails.

The old comment here said dune could not work on this machine: "conda-forge has
no dune-fem package ... the local source build is ABI-broken". dune passes now.
It was smoke_dune that was broken, running its script in openPASO's interpreter
instead of the one DUNE lives in, and the skip made that invisible for months.
"""
import unittest

import pytest

from backend_probe import backend_state, need_backend
from core.smoke_tests import (
    SmokeResult, smoke_ngsolve, smoke_skfem, smoke_kratos,
    smoke_dune, smoke_fourc, run_all_smoke_tests,
)


class TestSmokeResults(unittest.TestCase):

    def test_ngsolve_smoke(self):
        need_backend("ngsolve")
        r = smoke_ngsolve()
        self.assertTrue(r.passed, f"NGSolve smoke failed: {r.error}")
        self.assertLess(r.duration_ms, 5000)

    def test_skfem_smoke(self):
        need_backend("skfem")
        r = smoke_skfem()
        self.assertTrue(r.passed, f"scikit-fem smoke failed: {r.error}")
        self.assertLess(r.duration_ms, 5000)

    def test_kratos_smoke(self):
        need_backend("kratos")
        r = smoke_kratos()
        self.assertTrue(r.passed, f"Kratos smoke failed: {r.error}")

    def test_dune_smoke(self):
        need_backend("dune")
        r = smoke_dune()
        self.assertTrue(r.passed, f"DUNE smoke failed: {r.error}")

    def test_fourc_smoke(self):
        need_backend("fourc")
        r = smoke_fourc()
        self.assertTrue(r.passed, f"4C smoke failed: {r.error}")

    def test_run_all(self):
        """run_all_smoke_tests must report on exactly what it was asked about."""
        wanted = [s for s in ("skfem", "ngsolve")
                  if backend_state(s)[0] != "absent"]
        if not wanted:
            pytest.skip("neither scikit-fem nor NGSolve is installed here")
        results = run_all_smoke_tests(wanted)
        for solver in wanted:
            self.assertIn(solver, results)
            self.assertTrue(results[solver].passed,
                            f"{solver} smoke failed: {results[solver].error}")

    def test_smoke_result_to_dict(self):
        r = SmokeResult("test", True, version="1.0", duration_ms=100)
        d = r.to_dict()
        self.assertEqual(d["solver"], "test")
        self.assertTrue(d["passed"])
        self.assertNotIn("error", d)  # None should be dropped


if __name__ == "__main__":
    unittest.main()
