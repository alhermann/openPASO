"""The execution-log contract must reach the AGENT, in the rendered task text.

This is the fifth time a mechanism existed, was instrumented, and did not reach
the case it was built for (delivery wrote to the source tree; Option B had a
second door; the agent was never told the time; the audit refused coupled
submissions). So this file does not check that a template CONTAINS the clause —
it renders real tasks through the real builders and reads what the agent would
read.

WHAT THE CONTRACT HAD TO FIX. The old clause required only

    NDOF = <integer>

a line the agent types. The single-code text went further and told the agent
outright that this line "is how the assessment recognises that a solver actually
ran" — so the benchmark itself certified that one hand-written line proves a run.
C2_27b_MCP_seed15 took the invitation: it solved both subdomains with numpy and
scipy.sparse while the task named 4C and Kratos, and its whole execution
evidence was fourteen 10-byte files reading `NDOF = <n>`, credited to both named
codes at once. Its convergence order was genuinely second-order.

The grader cannot repair that after the fact — punishing an agent for meeting
the contract as written is not available to us, and 102 runs (67 bare, 35 openPASO)
had been charged with forgery for exactly that. So the contract now asks for the
one artefact that settles attribution: the solver's OWN captured output. Every
real run produces it by redirection; a numpy monolith cannot produce 4C's
banner.

The canonical `NDOF` line stays REQUIRED alongside it, and these tests pin that
too: it exists so a quiet honest dolfinx run is not condemned for its print
style, and removing it would re-open the bias it was introduced to close.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import campaign3_blind.build_balanced as B  # noqa: E402


# The phrases an agent must be able to act on. Kept as substrings of the
# rendered text so a reworded clause that drops the REQUIREMENT fails, while a
# reworded clause that keeps it can be updated here deliberately.
MUST_SAY = [
    "CONSOLE OUTPUT",        # the artefact demanded
    "2>&1",                  # a concrete, copyable way to obtain it
    "Do not retype",         # the loophole closed explicitly
    "NDOF = <integer>",      # the code-agnostic number kept
]


def _coupled_tasks(limit: int = 4):
    """Render real coupled tasks through the real builder."""
    out = []
    for fn in B.BUILDERS[:limit]:
        r = B.build_one(fn, 20260831)
        out.append((fn.__name__, r["task"]))
    return out


class TestCoupledTaskText(unittest.TestCase):
    def test_the_rendered_task_demands_the_solvers_own_output(self):
        for name, task in _coupled_tasks():
            for phrase in MUST_SAY:
                self.assertIn(
                    phrase, task,
                    f"{name}: the rendered task text an agent receives does "
                    f"not contain {phrase!r}. The clause may be in the "
                    f"template and not in the output.")

    def test_it_says_the_two_sides_must_carry_two_different_codes(self):
        """The specific thing C2_MCP_seed15 violated."""
        for name, task in _coupled_tasks():
            self.assertIn(
                "two DIFFERENT codes", task,
                f"{name}: nothing tells the agent the two logs must come from "
                f"two different codes — the exact hole a monolith walks "
                f"through")

    def test_no_unrendered_placeholder_survives(self):
        """A .format placeholder left in the text is a broken task, and the
        clause was edited by hand into a .format-ed template."""
        for name, task in _coupled_tasks():
            for bad in ("{run command", "{side}", "{ndof_desc}", "{integer}"):
                self.assertNotIn(bad, task, f"{name}: unrendered {bad!r}")

    def test_the_contract_is_still_detected_by_the_grader_gate(self):
        """evidence2.task_prescribes_run_logs decides whether a missing log is
        chargeable. The rewrite must not make the task unrecognisable to it."""
        from campaign3_blind.grading.evidence2 import task_prescribes_run_logs
        for name, task in _coupled_tasks():
            self.assertTrue(
                task_prescribes_run_logs(task),
                f"{name}: the grader no longer recognises this task as "
                f"prescribing run logs, so a missing log stopped being "
                f"chargeable — the rewrite silently disarmed the gate")


class TestSingleCodeTaskText(unittest.TestCase):
    """The single-code half carried the worse wording and must be fixed too:
    the substitution claim (27B+openPASO >= 397B bare) rests on single-code
    correctness being real."""

    def test_the_template_no_longer_certifies_a_typed_line_as_proof(self):
        import campaign3_blind.build_single_v2 as S
        src = Path(S.__file__).read_text()
        self.assertNotIn(
            "This line\nis how the assessment recognises that a solver "
            "actually ran", src,
            "the single-code task still tells the agent that one hand-written "
            "line proves a run")

    def test_the_template_demands_the_solvers_own_output(self):
        import campaign3_blind.build_single_v2 as S
        src = Path(S.__file__).read_text()
        for phrase in ("CONSOLE OUTPUT YOUR SOLVER ITSELF PRODUCED", "2>&1",
                       "Do not retype", "NDOF = <integer>"):
            self.assertIn(phrase, src, f"single-code contract lacks {phrase!r}")


if __name__ == "__main__":
    unittest.main()
