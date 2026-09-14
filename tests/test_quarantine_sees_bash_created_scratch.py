"""An agent writes with bash, not only with write_file.

MEASURED. C2_27b_MCP_seed72's agent built /tmp/coupled_heat_final -- 12
two-sided interface levels and 18 native solver artefacts, a complete worked
coupled answer -- using shell commands. Its transcript names paths under that
directory TEN times, and not once in the `wrote N chars to /path` form that
`_WROTE_RE` matches. So the stray-scratch quarantine could not see it, the next
seed's preflight refused to start (correctly, via the readable-answers gate), and
the development loop stalled instead of healing itself.

That is cross-seed leakage: one seed's agent hands the next seed a worked answer,
in a campaign whose headline claim is that the bare arm completes no coupled
problem.

TWO DEFECTS, both found by running the thing rather than reading it:

  1. Widening the path pattern was not enough. The pre-existing code walks UP
     from a matched path to find the directory to move, and stops at the first
     ancestor that is a scatter root. When the match is ALREADY top-level --
     /tmp/coupled_heat_final -- the first parent IS /tmp, so it breaks
     immediately, `top` stays None, and the tree is silently skipped. The
     widened scan still quarantined 0 trees while the gate was flagging exactly
     that directory.

  2. Probing the filesystem inside the per-match loop walked the disk once per
     regex match per transcript -- thousands of tree walks over ~2,000
     transcripts, minutes per call, in a function that runs at EVERY seed's
     preflight. Candidates are now collected as strings and each unique
     directory is probed once: 1.4 seconds, 8 trees moved, gate clean
     afterwards.

PRECISION comes from the materiality test, not from the pattern. The path regex
is deliberately loose; a candidate is quarantined only if it actually holds a
worked coupled answer (see host_hygiene._materially_useful /
_worked_handshake). That is what allows a loose pattern without repeating the
sweep that once moved a 37,445-file backend build tree.
"""

from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "campaign3_blind"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import host_hygiene as H  # noqa: E402


def _worked_answer(root: Path):
    """The shape seed72's agent left behind: per-level subdirectories, a
    two-sided interface set and native solver output, and NO top-level files
    beyond RESULT.txt."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "RESULT.txt").write_text("LEVELS = 3\n")
    for k in (1, 2, 3):
        for side in ("A", "B"):
            d = root / f"level_{k}" / f"subdomain_{side}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "exports.json").write_text('{"field_name": "temperature"}')
            (d / "imports.json").write_text("{}")
            (d / f"interface_level{k}_{side}.csv").write_text("x,y,u,qn\n")
        (root / f"level_{k}" / "subdomain_A" / "out.control").write_text("x")
        (root / f"level_{k}" / "subdomain_A" / "scatra-00001.pvtu").write_text("x")
    return root


class TestTheTopLevelCase(unittest.TestCase):
    """The bug that survived widening the pattern."""

    def test_a_top_level_directory_is_itself_a_candidate(self):
        """The matched path IS the directory to move, with no deeper file to
        walk up from. This is the case that returned None and skipped."""
        with TemporaryDirectory() as t:
            root = _worked_answer(Path(t) / "coupled_heat_final")
            self.assertIsNotNone(
                H._worked_handshake(root),
                "a complete two-code answer with per-level subdirectories was "
                "not recognised at its top level")

    def test_it_is_not_recognised_from_top_level_files_alone(self):
        """Why the first shape missed it: there are no interface_level files in
        the top directory, only in the per-level subdirectories."""
        with TemporaryDirectory() as t:
            root = _worked_answer(Path(t) / "coupled_heat_final")
            names = [f.name for f in root.iterdir() if f.is_file()]
            self.assertEqual(names, ["RESULT.txt"])
            self.assertIsNone(
                H._materially_useful(root, names),
                "the filename shape claims to recognise this, which would make "
                "the tree-level check redundant")


class TestPrecisionIsFromMateriality(unittest.TestCase):
    """The loose path pattern is only safe because of what follows it."""

    def test_an_ordinary_scratch_directory_is_not_a_candidate(self):
        with TemporaryDirectory() as t:
            d = Path(t) / "some_agent_scratch"
            (d / "sub").mkdir(parents=True)
            (d / "notes.txt").write_text("thinking")
            (d / "sub" / "solve.py").write_text("import numpy\n")
            self.assertIsNone(H._worked_handshake(d))
            self.assertIsNone(H._materially_useful(
                d, [f.name for f in d.iterdir() if f.is_file()]))

    def test_a_handshake_without_a_real_run_is_not_a_candidate(self):
        """2,300+ directories on this machine hold an exports/imports pair from
        driver sweeps; nearly all are one-point toys that teach nothing."""
        with TemporaryDirectory() as t:
            d = Path(t) / "toy_exchange"
            d.mkdir()
            (d / "exports.json").write_text('{"n_points": 1}')
            (d / "imports.json").write_text("{}")
            (d / "run.py").write_text("print(1)\n")
            self.assertIsNone(
                H._worked_handshake(d),
                "a schema demo with no solver output was treated as a worked "
                "answer")

    def test_a_workspace_is_never_a_candidate(self):
        """Without the size bound this matched /home/user/Schreibtisch --
        14,597 native artefacts -- because the walk descends through every
        checkout. That is how a 37,445-file backend build tree once got moved."""
        with TemporaryDirectory() as t:
            d = Path(t) / "workspace"
            (d / "repo").mkdir(parents=True)
            (d / "exports.json").write_text("{}")
            (d / "imports.json").write_text("{}")
            for i in range(260):
                (d / "repo" / f"mesh_{i}.vtu").write_text("x")
            self.assertIsNone(
                H._worked_handshake(d),
                "a large tree was flagged; the size bound is gone")

    def test_a_git_worktree_is_never_quarantinable(self):
        with TemporaryDirectory() as t:
            checkout = _worked_answer(Path(t) / "ofa-v2")
            (checkout / ".git").write_text("gitdir: /some/common/worktree\n")
            self.assertFalse(H.is_quarantinable_scratch(checkout, set()))

    def test_a_transcript_path_does_not_make_source_material(self):
        with TemporaryDirectory() as t:
            source = Path(t) / "source-tree"
            (source / "src").mkdir(parents=True)
            (source / "src" / "solver.py").write_text("print('solver')\n")
            self.assertFalse(H.is_quarantinable_scratch(source, set()))

    def test_a_worked_answer_remains_quarantinable(self):
        with TemporaryDirectory() as t:
            scratch = _worked_answer(Path(t) / "coupled_heat_final")
            self.assertTrue(H.is_quarantinable_scratch(scratch, set()))


class TestTheRunnerWiresItUp(unittest.TestCase):
    """The check must be reachable from the runner, not merely defined."""

    def test_the_runner_imports_both_shapes(self):
        src = (REPO_ROOT / "campaign3_blind" / "run_blind.py").read_text()
        self.assertIn("_materially_useful", src)
        self.assertIn("_worked_handshake", src)

    def test_it_collects_paths_beyond_the_write_file_confirmation(self):
        src = (REPO_ROOT / "campaign3_blind" / "run_blind.py").read_text()
        self.assertIn("_PATH_RE", src,
                      "only write_file confirmations are collected again, so "
                      "bash-created scratch is invisible")

    def test_it_probes_each_candidate_once(self):
        """Guard the performance fix: probing inside the per-match loop took
        minutes in a function that runs at every preflight."""
        src = (REPO_ROOT / "campaign3_blind" / "run_blind.py").read_text()
        self.assertIn("bash_candidates", src)
        i = src.index("for m in _PATH_RE.finditer(text):")
        window = src[i:i + 700]
        self.assertNotIn(
            "_worked_handshake(", window,
            "the filesystem is probed inside the per-match loop again")


if __name__ == "__main__":
    unittest.main()
