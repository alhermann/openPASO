"""`examples` must not dead-end a coupled request.

MEASURED BEFORE THE FIX: every coupling keyword returned "No examples found"
for both fourc and fenics — `coupling`, `coupled`, `partitioned`,
`dirichlet_neumann`, and `fsi`, the flagship. Zero hits, and no hint that the
material exists elsewhere, while the tool's own docstring tells the agent to
ALWAYS call it before writing input files.

It cannot be otherwise by construction: `examples` matches FILENAMES in ONE
backend's test suite, and a partitioned coupled run is two participant programs
plus a driver, so it lives in neither code's test tree. The gap is structural,
not a missing file.

Why it is the worst possible place to be silent: the coupled cells are where the
openPASO arm most needs a worked example, and that arm's measured failure is
running out of tool calls — median 39 against bare's 95, with no output written
in 60% of runs against bare's 31%. A dead end spends a call and returns nothing.

The reply is a POINTER, not the payload: the coupling knowledge is ~66k chars,
and this tool is called early, when the agent has the most calls left to lose.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.consolidated import (  # noqa: E402
    _coupling_example_pointer,
    _is_coupling_request,
)

# Keywords an agent facing a coupled task would plausibly type.
COUPLING_KEYWORDS = [
    "coupling", "coupled", "partitioned", "dirichlet_neumann",
    "dirichlet-neumann", "fsi", "FSI", "tsi", "precice", "cosim",
    "two-way", "two_code", "staggered", "interface_field",
]

# Ordinary single-code searches that must NOT be diverted.
SINGLE_CODE_KEYWORDS = [
    "poisson", "heat", "contact", "navier_stokes", "elasticity",
    "biharmonic", "stokes", "peridynamic", "dem", "transient",
    "hyperelasticity", "maxwell", "eigenvalue", "", "   ",
]


class TestTheClassifier(unittest.TestCase):
    def test_every_coupling_keyword_is_recognised(self):
        missed = [k for k in COUPLING_KEYWORDS if not _is_coupling_request(k)]
        self.assertEqual(missed, [], f"coupled requests not recognised: {missed}")

    def test_no_single_code_keyword_is_diverted(self):
        """A false positive is worse than a false negative here: it would send
        an agent asking about Poisson to the coupling material."""
        wrong = [k for k in SINGLE_CODE_KEYWORDS if _is_coupling_request(k)]
        self.assertEqual(wrong, [], f"single-code searches diverted: {wrong}")

    def test_fsi_counts_as_coupling_even_though_decks_exist(self):
        """A MONOLITHIC fsi deck in one code's tests is not an example of the
        partitioned two-code run the task asks for, and an agent handed one
        believes its question was answered."""
        self.assertTrue(_is_coupling_request("fsi"))


class TestThePointer(unittest.TestCase):
    def test_it_names_the_call_that_actually_works(self):
        p = _coupling_example_pointer("fsi", "fourc")
        self.assertIn('knowledge(topic="coupling"', p)
        self.assertIn("couple(participants=", p)

    def test_it_carries_the_solver_through(self):
        p = _coupling_example_pointer("coupled", "ngsolve")
        self.assertIn('solver="ngsolve"', p)

    def test_it_states_the_shape_of_a_coupled_run(self):
        """An agent that has never run one needs to size the work."""
        p = _coupling_example_pointer("coupling", "fenics")
        for phrase in ("exports.json", "imports.json", "DIRICHLET", "NEUMANN"):
            self.assertIn(phrase, p, f"the pointer omits {phrase!r}")

    def test_it_warns_against_the_monolithic_substitute(self):
        p = _coupling_example_pointer("fsi", "fourc")
        self.assertIn("monolithic", p)

    def test_it_is_a_pointer_not_a_payload(self):
        """The coupling knowledge is ~66k chars. Returning it here would spend
        the context this tool is called early to preserve."""
        p = _coupling_example_pointer("coupling", "fourc")
        self.assertLess(len(p), 4000, f"the pointer grew to {len(p)} chars")


class TestTheFoundSomethingCaseIsTheDangerousOne(unittest.TestCase):
    """The empty-result branch could not reach the case it was built for.

    MEASURED: the filename search comes back empty for fenics but NOT for
    fourc. 4C's test suite matches every coupling keyword —

        fsi          -> fsi_dc3D_part_ait_ga_ost_xwall.4C.yaml
        partitioned  -> elch_1D_line2_multiscale_..._partitioned_macrotomicro
        coupled      -> elch_onewaycoupled_ost.4C.yaml
        coupling     -> beam3r_..._beam_to_beam_point_coupling_elbow.4C.yaml

    and every one of them solves the whole problem INSIDE 4C. They are
    single-code multiphysics, not the two-code partitioned run with
    exports.json / imports.json that a C-series task asks for. Zero 4C decks in
    the tree mention preCICE, so none is a genuine two-code example.

    The case the guard could not reach is the worse one: nothing stops an agent
    looking further like a plausible answer. So the pointer is appended whenever
    the REQUEST is about coupling, found files or not.
    """

    def test_the_fourc_search_does_find_files_so_the_empty_branch_is_not_enough(self):
        from tools.knowledge import discover_test_dirs, resolve_search_keywords
        dirs = discover_test_dirs()
        td = dirs.get("fourc")
        if not (td and Path(td).is_dir()):
            self.skipTest("4C test inputs not present on this machine")
        hits = {}
        for kw in ("coupling", "coupled", "partitioned", "fsi"):
            cands = list(dict.fromkeys(
                [kw] + resolve_search_keywords("fourc", kw)))
            got = [f.name for k in cands
                   for f in sorted(Path(td).rglob("*.4C.yaml"))
                   if k.lower() in f.name.lower()]
            if got:
                hits[kw] = got[:2]
        self.assertTrue(
            hits,
            "4C's test suite no longer matches any coupling keyword. If that "
            "is real, the empty-result branch alone would suffice again — but "
            "verify before simplifying: this measurement is why the pointer is "
            "appended unconditionally.")

    def test_none_of_those_decks_is_a_two_code_example(self):
        """If one ever is, the warning should be narrowed rather than kept."""
        from tools.knowledge import discover_test_dirs
        td = dirs = discover_test_dirs().get("fourc")
        if not (td and Path(td).is_dir()):
            self.skipTest("4C test inputs not present on this machine")
        precice = [f.name for f in Path(td).rglob("*.4C.yaml")
                   if "precice" in f.read_text(errors="ignore").lower()]
        self.assertEqual(
            precice, [],
            f"these 4C decks reference preCICE and may be genuine two-code "
            f"coupled examples worth serving directly: {precice[:5]}")


if __name__ == "__main__":
    unittest.main()
