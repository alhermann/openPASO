"""4C's real diagnostic must reach the agent, not the MPI boilerplate.

WHAT IT COST, MEASURED. Both OASiS-arm runs of coupled cell C2 (seeds 70 and 71)
wrote COULD_NOT_COMPLETE with ZERO deliverables and stopped at 30 and 37 tool
calls, having used 22-27% of their wall budget. Their stated root cause:

    "The 4C binary (/home/alexander/4C/build/4C) fails to execute:
     'Invalid MIT-MAGIC-COOKIE-1 key' followed by 'MPI_ABORT was invoked on
     rank 0' ... This occurs even when using OASiS's run_with_generator tool"

The binary was never broken. The bare arm ran the same binary in the same
minutes and produced a complete three-level submission, and `4C -p` prints the
cookie line and then dumps the whole input grammar successfully.

WHY THEY BELIEVED IT. 4C prints its diagnostic BEFORE the MPI abort boilerplate:

    Invalid MIT-MAGIC-COOKIE-1 key            <- X11 noise, on every run
    ***** 4C version 2026.2.0-dev *****
    Trilinos Version: ...
    PROC 0 ERROR in 4C_io_input_spec_builders.cpp, line 633:
    Could not match this input
    STRUCTURAL DYNAMIC:
      NOT_A_REAL_KEY: 42                      <- the actual defect
    ... ~40 lines of MPI_ABORT boilerplate

and the backend reported `stdout_text[-1000:]`. The last 1000 characters are
entirely boilerplate. The line's own comment said "4C often writes the real
error to stdout, not stderr" and then took the wrong end of it — a mechanism
that existed, was instrumented, and did not reach the case it was built for.

These tests use recorded 4C output rather than invoking the binary, so they keep
meaning on a machine without 4C, and one test runs the real binary when present.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from backends.fourc.backend import _fourc_diagnostic  # noqa: E402

FOURC = Path("/home/alexander/4C/build/4C")

# Recorded verbatim from a real failing run.
REAL_STDOUT = """Invalid MIT-MAGIC-COOKIE-1 key
******************************************************
*                         4C                         *
*                version 2026.2.0-dev                *
******************************************************

Trilinos Version: f4d64271518 (git SHA1)
Total number of MPI ranks: 1

=========================================================================
PROC 0 ERROR in /home/alexander/4C/src/core/io/src/4C_io_input_spec_builders.cpp, line 633:
Could not match this input

STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  DYNAMICTYPE: "Statics"
  NOT_A_REAL_KEY: 42

against the given input specification.
=========================================================================
""" + ("""--------------------------------------------------------------------------
MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
with errorcode 1.

NOTE: invoking MPI_ABORT causes Open MPI to kill all MPI processes.
You may or may not see output from other processes, depending on
exactly when Open MPI kills them.
--------------------------------------------------------------------------
""" * 4)


def _old_behaviour(stdout_text, stderr_text):
    """What the backend used to hand the agent."""
    return (stderr_text[-1000:] + "\n--- stdout tail ---\n"
            + stdout_text[-1000:])[-2000:]


class TestTheDiagnosticIsFoundByContent(unittest.TestCase):
    def test_the_agent_now_receives_the_real_error(self):
        got = _fourc_diagnostic(REAL_STDOUT, "")
        self.assertIn("Could not match this input", got)
        self.assertIn("PROC 0 ERROR", got)

    def test_it_names_the_offending_input_key(self):
        """A diagnostic that does not say WHICH key is not actionable."""
        got = _fourc_diagnostic(REAL_STDOUT, "")
        self.assertIn("NOT_A_REAL_KEY", got)

    def test_the_old_behaviour_really_did_hide_it(self):
        """Proof this test discriminates: the previous code fails it.

        And it was worse than the agents' own account suggests. With the abort
        boilerplate repeated -- Open MPI prints it once per rank plus a wrapper
        copy -- the last 1000 characters contain neither the diagnostic NOR the
        cookie line: they are pure MPI noise. The agents quoted the cookie line
        from their own direct runs, not from what the tool returned. What the
        tool returned explained nothing at all.
        """
        old = _old_behaviour(REAL_STDOUT, "")
        self.assertNotIn("Could not match this input", old)
        self.assertNotIn("PROC 0 ERROR", old)
        self.assertNotIn("NOT_A_REAL_KEY", old)
        self.assertIn("MPI_ABORT", old)
        # nothing but boilerplate: no 4C-specific content survives
        self.assertNotIn("4C", old.replace("MPI_COMM_WORLD", ""))

    def test_a_crash_with_no_marker_still_reports_something(self):
        """A signal or MPI-level failure prints no 4C marker; the fallback must
        still hand back the tails rather than an empty string."""
        got = _fourc_diagnostic("Invalid MIT-MAGIC-COOKIE-1 key\n", "killed\n")
        self.assertTrue(got.strip())
        self.assertIn("killed", got)

    def test_a_marker_free_abort_says_the_cookie_is_not_the_cause(self):
        """The exact situation the two runs were in: nothing but noise. They
        must be told the cookie line is not an explanation."""
        got = _fourc_diagnostic("Invalid MIT-MAGIC-COOKIE-1 key\n", "")
        self.assertIn("X11", got)
        self.assertIn("SUCCESSFUL", got)
        self.assertIn("from the TOP", got)

    def test_stderr_markers_are_also_found(self):
        got = _fourc_diagnostic("", "terminate called after throwing an "
                                    "instance of 'std::runtime_error'\n")
        self.assertIn("terminate called", got)


class TestTheServedKnowledgeSaysIt(unittest.TestCase):
    """The backend fix helps an agent using OASiS's runner. An agent running 4C
    itself needs to be told, or it draws the same wrong conclusion."""

    def test_the_universal_block_warns_about_the_cookie_line(self):
        import tools.knowledge as K
        for phrase in ("MIT-MAGIC-COOKIE", "X11", "4C -p"):
            self.assertIn(phrase, K._UNIVERSAL,
                          f"the served text does not mention {phrase!r}")

    def test_it_says_to_read_the_log_from_the_top(self):
        import tools.knowledge as K
        self.assertIn("FROM THE TOP", K._UNIVERSAL.upper())
        self.assertIn("tail", K._UNIVERSAL)

    def test_it_tells_the_agent_to_self_test_before_blaming_the_tool(self):
        import tools.knowledge as K
        self.assertIn("--help", K._UNIVERSAL)
        self.assertIn("could-not-finish", K._UNIVERSAL)


@unittest.skipUnless(FOURC.is_file(), "4C binary not present on this machine")
class TestAgainstTheRealBinary(unittest.TestCase):
    def test_a_malformed_deck_yields_an_actionable_message(self):
        env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")
        with TemporaryDirectory() as t:
            deck = Path(t) / "bad.4C.yaml"
            deck.write_text('PROBLEM TYPE:\n  PROBLEMTYPE: "Structure"\n'
                            'STRUCTURAL DYNAMIC:\n  NOT_A_REAL_KEY: 42\n')
            r = subprocess.run([str(FOURC), deck.name, "out"], cwd=t,
                               capture_output=True, text=True, env=env,
                               timeout=180)
        self.assertNotEqual(r.returncode, 0, "the bad deck was accepted")
        got = _fourc_diagnostic(r.stdout, r.stderr)
        self.assertIn("NOT_A_REAL_KEY", got,
                      "the real binary's diagnostic does not reach the agent")

    def test_the_cookie_line_appears_on_a_SUCCESSFUL_run(self):
        """The fact that settles it: the line is not a failure signal."""
        env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")
        r = subprocess.run([str(FOURC), "-p"], capture_output=True, text=True,
                           env=env, timeout=180)
        self.assertEqual(r.returncode, 0, "4C -p failed; the binary IS broken")
        self.assertIn("MIT-MAGIC-COOKIE", r.stdout + r.stderr,
                      "the cookie line no longer appears, so the warning in "
                      "the served text may be stale on this machine")


if __name__ == "__main__":
    unittest.main()


class TestTheDeckSyntaxPrimitivesAreServed(unittest.TestCase):
    """Two blockers found by running the OASiS arm's own failing C2 deck, each
    verified by execution and each of which alone kills the run.

    LAYER 1 -- the topology block was written as bare quoted strings rather than
    a YAML sequence, so 4C died during input parsing with

        ERROR: could not find ':' colon after key
        53:17: "NODE 7 DLINE 1"  (size=16)

    BEFORE printing its own banner. Measured: without line-buffered output the
    whole run produced 8 lines -- the X11 cookie warning and MPI boilerplate --
    and with `stdbuf -oL` it produced 43 lines including that message. The bare
    arm's deck, in the same directory with the same binary, exits 0 and reports
    "finished normally"; the only difference in that section is the `- `.

    LAYER 2 -- with the topology repaired, the deck reached the source term and
    4C rejected `-12*x**3*y/5` from
    4C_utils_symbolic_expression.cpp. The task states its source term in Python
    notation, so an agent that copies it hits this on every 4C cell. `^` parses.
    The identical trap was measured in FEBio's math loads earlier
    (`-1*X**2` -> "Token expected (position 6)"; `-1*X^2` accepted), which is
    why the rule is stated generally rather than as a 4C quirk.
    """

    def test_the_double_star_trap_is_served(self):
        import tools.knowledge as K
        self.assertIn("`**` IS NOT EXPONENTIATION", K._UNIVERSAL)
        self.assertIn("USE `^`", K._UNIVERSAL)

    def test_it_names_both_solvers_that_reject_it(self):
        """Stated for one code it reads as a quirk; for two it is a rule."""
        import tools.knowledge as K
        i = K._UNIVERSAL.index("`**` IS NOT EXPONENTIATION")
        block = K._UNIVERSAL[i:i + 1200]
        self.assertIn("SYMBOLIC_FUNCTION_OF_SPACE_TIME", block)
        self.assertIn("FEBio", block)
        self.assertIn("Token expected", block)

    def test_it_says_to_rewrite_the_whole_expression(self):
        """One surviving `**` fails the parse just as completely."""
        import tools.knowledge as K
        self.assertIn("not the first term", K._UNIVERSAL)

    def test_the_yaml_sequence_rule_is_served_with_both_forms(self):
        import tools.knowledge as K
        self.assertIn('- "NODE 1 DLINE 1"', K._UNIVERSAL)
        self.assertIn("could not find ':' colon after key", K._UNIVERSAL)

    def test_it_warns_the_abort_precedes_the_banner(self):
        """This is what made the agents conclude the binary was broken."""
        import tools.knowledge as K
        self.assertIn("BEFORE PRINTING ITS OWN BANNER", K._UNIVERSAL)
        self.assertIn("broken binary", K._UNIVERSAL)

    def test_the_diagnostic_extractor_finds_a_yaml_parse_error(self):
        """The extractor initially knew only `PROC n ERROR` and missed exactly
        this failure -- the case it was written for."""
        log = ("Invalid MIT-MAGIC-COOKIE-1 key\n"
               "ERROR: could not find ':' colon after key\n"
               "53:17: \"NODE 7 DLINE 1\"  (size=16)\n"
               + "MPI_ABORT was invoked on rank 0\n" * 40)
        got = _fourc_diagnostic(log, "")
        self.assertIn("could not find", got)
        self.assertIn("53:17", got)
        self.assertIn("NODE 7 DLINE 1", got)


FAILURE_LOGS = Path(__file__).parent / "fixtures" / "fourc_failure_logs"

# Four real 4C failures, peeled one at a time from the OASiS arm's own failing
# C2 deck by repairing what the previous layer revealed and re-running. Each is
# fatal on its own; each is diagnosed precisely by 4C; none survives a tail.
LAYERS = [
    ("L1_yaml_sequence.log",        "could not find",
     "topology written as bare strings instead of a YAML sequence"),
    ("L2_double_star_exponent.log", "Error while parsing",
     "Python `**` in a SYMBOLIC_FUNCTION_OF_SPACE_TIME"),
    ("L3_val_array_size.log",       "incorrect size",
     "VAL had 7 entries against NUMDOF: 1"),
    ("L4_condition_dimension.log",  "larger than the problem dimension",
     "a DESIGN VOL condition on a 2-D problem"),
]


class TestAllFourRealLayersReachTheAgent(unittest.TestCase):
    """The measurement that settles what OASiS owed this agent.

    Layers 1 and 2 are knowledge OASiS should have served, and now does. Layers
    3 and 4 are the agent's OWN modelling errors -- and 4C names both of them
    exactly ("Candidate parameter 'VAL' has incorrect size", "Dimension of
    condition is larger than the problem dimension"). They needed no new
    knowledge whatsoever. They needed the diagnostic to be VISIBLE.

    Measured over these four logs: the extractor surfaces 4 of 4, the previous
    stdout-tail surfaced 0 of 4. That is the difference between four one-run
    fixes and three runs concluding the binary was broken and submitting
    nothing.
    """

    def test_the_extractor_surfaces_every_layer(self):
        missed = []
        for fname, needle, what in LAYERS:
            log = (FAILURE_LOGS / fname).read_text(errors="ignore")
            if needle not in _fourc_diagnostic(log, ""):
                missed.append(f"{fname}: {what}")
        self.assertEqual(missed, [], f"layers hidden from the agent: {missed}")

    def test_the_old_tail_surfaced_none_of_them(self):
        """Proof of the discriminating power, not a rhetorical flourish."""
        surfaced = []
        for fname, needle, _ in LAYERS:
            log = (FAILURE_LOGS / fname).read_text(errors="ignore")
            if needle in _old_behaviour(log, ""):
                surfaced.append(fname)
        self.assertEqual(
            surfaced, [],
            f"the old stdout-tail already showed {surfaced}; this suite is no "
            f"longer measuring the defect it was written for")

    def test_without_line_buffering_there_is_nothing_to_extract(self):
        """L0: the same failing deck run WITHOUT stdbuf produced eight lines --
        the X11 cookie warning and MPI boilerplate, no banner, no error. 4C
        buffers stdout and MPI_Abort kills the process before the flush. This is
        why the served text tells agents to keep the run line-buffered, and why
        the backend already wraps 4C in `stdbuf -oL`."""
        log = (FAILURE_LOGS / "L0_no_stdbuf_eight_lines.log").read_text()
        self.assertLess(len(log.splitlines()), 12)
        self.assertNotIn("4C", log.replace("MPI_COMM_WORLD", ""))
        got = _fourc_diagnostic(log, "")
        # nothing to find, so it must say so rather than imply a cause
        self.assertIn("X11", got)
        self.assertIn("from the TOP", got)
