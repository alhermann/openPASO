"""Does openPASO serve ENOUGH to make 4C run? Answered by writing a deck from it.

THE QUESTION THAT MATTERS. openPASO elides "the solve" by design (Option B), and
for a Python library that is right: the agent should choose its own mesh, space
and weak form. But 4C IS A BINARY THAT CONSUMES A YAML DECK. For it the deck is
the backend's run interface in exactly the way exports.json is the driver's --
you cannot guess it, and no solve happens without it.

MEASURED BEFORE THIS CHANGE, on the payload an agent actually receives from
knowledge(topic='coupling', solver='fourc'):

    PROBLEM TYPE            absent
    MATERIALS               absent
    SCALAR TRANSPORT DYNAMIC absent
    DESIGN LINE DIRICH      absent
    NODE COORDS             absent
    NUMDOF                  absent

It named TRANSPORT ELEMENTS and SYMBOLIC_FUNCTION and nothing else. An agent
could not write a runnable deck from it, and all three openPASO-arm runs of coupled
cell C2 failed on exactly that -- seed71 said so in its own words: "4C scalar
transport module requires specific topology definitions that proved difficult to
generate correctly within time budget".

WHAT IS SERVED NOW is the GRAMMAR, not a solve: the section list with exact
syntax and ARBITRARY placeholder numbers, the four measured ways a deck dies
silently, and the fact that a 2-D volume source term is a SURF Neumann
condition. The agent still derives its own mesh, materials, boundary data and
source term, and still interprets its own results.

THE VERIFICATION IS THE DECK IN tests/fixtures/fourc_running_deck/. I wrote it
acting as the agent, using only the served grammar, for C2 side A -- (0, 0.625)
x (0, 1), k = 1, u = 0 on the outer boundary, h = 1/8, the task's own source
term with `**` rewritten as `^`. 4C runs it to completion: exit 0, "processor 0
finished normally", and it writes out.control, out.mesh.scatra.s0 and
out.result.scatra.s1.

One gap was found by that exercise and is now closed: the source-term mechanism
was not served, and omitting it solves f = 0 -- a run that SUCCEEDS with the
wrong answer, the worst failure mode available.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
import backend_probe  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

DECK = (Path(__file__).parent / "fixtures" / "fourc_running_deck"
        / "C2_sideA_from_served_grammar.4C.yaml")
FOURC = backend_probe.fourc_binary()


def _served():
    from tools.consolidated import _front_load_coupling
    from tools.coupling_knowledge import coupling_knowledge
    return _front_load_coupling(coupling_knowledge(solver="fourc"), "fourc")


class TestTheGrammarIsServed(unittest.TestCase):
    """Every section the running deck uses must be documented, or the deck is
    evidence about me rather than about openPASO."""

    def test_every_section_of_the_running_deck_is_in_the_served_text(self):
        served = _served()
        sections = sorted({
            ln.rstrip(":") for ln in DECK.read_text().splitlines()
            if ln and not ln.startswith((" ", "-")) and ln.endswith(":")})
        missing = [s for s in sections if s not in served]
        self.assertEqual(
            missing, [],
            f"the deck runs but openPASO does not document these sections, so an "
            f"agent could not have written it: {missing}")

    def test_the_source_term_mechanism_is_served(self):
        """Omitting it solves f = 0: a successful run with a wrong answer."""
        served = _served()
        self.assertIn("DESIGN SURF TRANSPORT NEUMANN CONDITIONS", served)
        self.assertIn("DSURF-NODE TOPOLOGY", served)
        self.assertIn("you solve f = 0", served)

    def test_the_placeholders_are_marked_as_arbitrary(self):
        """Option B: the grammar is served, the physics is not."""
        self.assertIn("ARBITRARY PLACEHOLDER", _served())

    def test_the_four_silent_failures_are_served(self):
        served = _served()
        for phrase in ("could not find ':' colon",     # YAML sequence syntax
                       "IS NOT EXPONENTIATION",         # ** vs ^
                       "incorrect size",                # NUMDOF sizing
                       "Dimension of condition"):       # VOL on a 2-D problem
            self.assertIn(phrase, served, f"missing: {phrase}")


@unittest.skipUnless(FOURC.is_file(), "4C binary not present on this machine")
class TestTheDeckActuallyRuns(unittest.TestCase):
    def test_4c_runs_the_deck_to_completion(self):
        env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")
        with TemporaryDirectory() as t:
            d = Path(t) / "A.4C.yaml"
            d.write_text(DECK.read_text())
            r = subprocess.run(
                ["stdbuf", "-oL", str(FOURC), d.name, "out"], cwd=t,
                capture_output=True, text=True, env=env, timeout=600)
            wrote = sorted(f.name for f in Path(t).iterdir()
                           if f.name.startswith("out"))
        self.assertEqual(r.returncode, 0,
                         f"the deck written from the served grammar no longer "
                         f"runs: {r.stdout[-800:]}")
        self.assertIn("finished normally", r.stdout)
        self.assertTrue([w for w in wrote if w.endswith(".control")],
                        f"4C exited 0 but wrote no result files: {wrote}")


if __name__ == "__main__":
    unittest.main()


class TestBothPathsServeTheSameOneCopy(unittest.TestCase):
    """The grammar must reach the SINGLE-CODE cells too, from ONE copy.

    It was served only from the coupling payload. A single-code 4C cell (FC1,
    FC2) never calls knowledge(topic='coupling') -- it has no coupling -- so the
    agents working the LARGEST target (>70% single-code, currently ~40%)
    received nothing about how to write a runnable deck. Same defect that killed
    all three openPASO runs of coupled C2, on the bigger number.

    And it must not become four copies. The interface-probe text existed in four
    places, two of them dead, and a fix applied to the wrong one looked correct
    for hours; the drawn task still emitted the old grid. So both serving paths
    import backends/fourc/deck_grammar.py and this test fails if either grows
    its own.
    """

    def test_the_single_code_path_serves_it(self):
        from backends.fourc.backend import FourcBackend
        import json
        b = FourcBackend()
        for physics in ("scalar_transport", "thermal", "poisson"):
            k = b.get_knowledge(physics)
            self.assertIn("deck_grammar", k or {},
                          f"a single-code {physics} agent gets no deck grammar")
            blob = json.dumps(k, default=str)
            self.assertIn("PROBLEM TYPE", blob)
            self.assertIn("DESIGN SURF TRANSPORT NEUMANN", blob)

    def test_the_coupling_path_serves_it(self):
        self.assertIn("PROBLEM TYPE", _served())
        self.assertIn("DESIGN SURF TRANSPORT NEUMANN", _served())

    def test_both_paths_carry_THE_SAME_text(self):
        """One copy, or the next fix lands on the wrong one."""
        from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR as G
        from backends.fourc.backend import FourcBackend
        import json
        single = json.loads(json.dumps(
            FourcBackend().get_knowledge("scalar_transport"), default=str))
        self.assertEqual(single["deck_grammar"], G,
                         "the single-code path has its own copy of the grammar")
        # the coupling payload re-indents it as a bullet; compare a distinctive
        # line rather than the whole block
        for line in ("DESIGN SURF TRANSPORT NEUMANN CONDITIONS:",
                     'NODE 1 DSURFACE 1'):
            self.assertIn(line, G)
            self.assertIn(line, _served())

    def test_the_grammar_lives_in_exactly_one_module(self):
        import subprocess
        # Match the DOC, not the section name: generators legitimately EMIT
        # `DESIGN SURF TRANSPORT NEUMANN CONDITIONS:` in their deck templates
        # (cardiac_monodomain.py does), and that is not a second copy of the
        # explanation. This test's first version conflated the two and failed
        # on a false positive -- worth keeping in mind, because the same
        # conflation is how a "duplicate" hunt can start deleting real code.
        r = subprocess.run(
            ["grep", "-rl", "you solve f = 0", "--include=*.py", "src/"],
            capture_output=True, text=True, cwd=str(REPO_ROOT))
        files = sorted(f for f in r.stdout.split() if f)
        self.assertEqual(
            files, ["src/backends/fourc/deck_grammar.py"],
            f"the grammar DOC appears in more than one module, which is how "
            f"four copies of the interface-probe rule happened: {files}")
