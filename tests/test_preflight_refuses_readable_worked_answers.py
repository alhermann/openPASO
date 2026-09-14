"""The runner refused to start while the KEYS were readable, and said nothing
about worked ANSWERS lying around the host.

WHAT WAS MEASURED. On this machine, during development:

  * a COMPLETED 4C+Kratos two-slab coupled heat-transfer run left behind by
    this repo's own test suite — a real 4C deck, 4C's native output
    (out.control, .pvtu, result files), Kratos params, and exports.json plus
    imports.json on BOTH sides, i.e. the whole interface handshake;
  * seven trees of prior agent scatter in $HOME carrying 111 files with the
    campaign's exact deliverable names (solution_level*.csv,
    interface_level*.csv, residual_level*.csv, run_level*.log), including a
    full three-level set and one tree that also held the agent's working
    coupling scripts.

A bare agent working a coupled cell READ one of them — visible in its
trajectory — in a campaign whose headline claim is that the bare arm completes
no coupled problem. That claim cannot be measured on a machine that hands the
bare arm a worked example.

WHY THE EXISTING QUARANTINE CANNOT COVER IT. `_quarantine_stray_scratch` moves
only what a prior RUN's transcript names, deliberately: an earlier
filesystem-sweep version would have moved 1043 directories including unrelated
pytest trees. A developer or test artefact that no run created is invisible to
it.

WHY THE RULE IS NARROW. A sweep on the deliverable FILENAMES alone matched 357
directories and 2,314 files here, almost all of them this repo's own grader test
fixtures, whose contents are synthetic data written to exercise the grader and
which teach an agent nothing. A gate that fires on those is a gate nobody leaves
switched on. Requiring BOTH a two-sided multi-level submission AND something
that actually ran (native solver output) or actually runs (a script importing a
prescribed backend) matched exactly ONE directory on the same machine.

The preflight REFUSES rather than moving: what to do with a developer's own
files is the developer's call, not the runner's.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "campaign3_blind"))


def _mat(d: Path):
    """The decision function, called the way preflight calls it.

    Imported from host_hygiene, not run_blind: the check was extracted precisely
    so it carries no langchain dependency and can be exercised by the repo's
    normal test run rather than only from the one virtualenv that has langchain.
    """
    import host_hygiene as H
    return H._materially_useful(d, [f.name for f in d.iterdir() if f.is_file()])


def _write(d: Path, names, body="x\n"):
    for n in names:
        (d / n).write_text(body)


def _two_sided(levels=3):
    out = []
    for k in range(1, levels + 1):
        out += [f"interface_level{k}_A.csv", f"interface_level{k}_B.csv",
                f"solution_level{k}_A.csv", f"solution_level{k}_B.csv",
                f"residual_level{k}.csv"]
    return out


class TestItFiresOnAWorkedAnswer(unittest.TestCase):
    def test_a_coupled_set_with_native_solver_output_is_flagged(self):
        """The shape of /tmp/coupled_sim_final: 3 two-sided levels plus real
        solver output."""
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, _two_sided(3))
            _write(d, ["out.control", "thermo-00000.pvtu", "out-thermo.pvd"])
            got = _mat(d)
        self.assertIsNotNone(got, "a worked coupled answer was not flagged")
        levels, native, scripts = got
        self.assertEqual(levels, 3)
        self.assertGreaterEqual(native, 3)

    def test_a_coupled_set_with_a_backend_script_is_flagged(self):
        """The shape of $HOME/workspace_coupling: deliverables plus the agent's
        own working solver scripts, which are the adaptable part."""
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, _two_sided(3))
            (d / "fenicsx_solver.py").write_text(
                "import dolfinx\nfrom mpi4py import MPI\n")
            got = _mat(d)
        self.assertIsNotNone(got)
        self.assertIn("fenicsx_solver.py", got[2])

    def test_two_levels_is_enough(self):
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, _two_sided(2))
            _write(d, ["out.control"])
            self.assertIsNotNone(_mat(d))


class TestItStaysQuietOnHarmlessThings(unittest.TestCase):
    """Every one of these exists on this machine in quantity. If the gate fires
    on them it gets switched off, and then it protects nothing."""

    def test_deliverable_names_alone_are_not_enough(self):
        """This repo's grader fixtures: the right filenames, synthetic data, no
        solver anywhere. 357 directories matched this shape here."""
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, _two_sided(3))
            self.assertIsNone(
                _mat(d),
                "the gate fires on filenames alone, which matched 357 "
                "directories of this repo's own test fixtures")

    def test_one_sided_interface_files_are_not_a_submission(self):
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, [f"interface_level{k}_A.csv" for k in (1, 2, 3)])
            _write(d, ["out.control"])
            self.assertIsNone(_mat(d))

    def test_a_single_level_is_not_a_convergence_study(self):
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, _two_sided(1))
            _write(d, ["out.control"])
            self.assertIsNone(_mat(d))

    def test_solver_output_alone_is_not_flagged(self):
        """A backend's own build or example tree — /home/user/dealii has
        37,445 files and is a REQUIRED dependency. Flagging it is how I broke
        the toolchain once already."""
        with TemporaryDirectory() as t:
            d = Path(t)
            _write(d, ["out.control", "step-3.vtu", "solve.py"],
                   body="import dolfinx\n")
            self.assertIsNone(_mat(d))

    def test_a_script_without_deliverables_is_not_flagged(self):
        with TemporaryDirectory() as t:
            d = Path(t)
            (d / "run.py").write_text("import ngsolve\n")
            self.assertIsNone(_mat(d))


class TestTheHostIsActuallyClean(unittest.TestCase):
    def test_no_worked_answer_is_readable_right_now(self):
        """The live check, on the real machine. This is the one that asks what
        an AGENT would find rather than what the function does."""
        import host_hygiene as H
        found = H.readable_worked_answers()
        self.assertEqual(
            [str(d) for d, _ in found], [],
            "worked coupled answers are readable outside the campaign; a bare "
            "coupled run will find them and the 'bare completes none' claim "
            "cannot be measured until they are moved or sealed")


if __name__ == "__main__":
    unittest.main()
