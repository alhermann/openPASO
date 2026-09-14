"""Three ways the grader called an honest run a forgery, each measured.

The paper's claim is that fabrication is near zero in the openPASO arm. A false
accusation costs that claim more than a miss does, and every defect below fell
mostly on the BARE arm — so each one was inflating the measured uplift.

1. SPARTA WRITES `log.sparta`, AND THE GRADER COULD NOT OPEN IT. Its suffix is
   `.sparta`, which READABLE_SUFFIXES did not list, so the file was skipped
   unopened and the three measured SPARTA signatures could never fire: the only
   file containing them was unreadable by construction. 67 runs in the tree
   (35 bare, 32 openPASO) hold real SPARTA output — "SPARTA (24 Sep 2025)",
   "Loop time of ...", "Created ... child grid cells" — in such a file. This is
   the same defect this module already records as fixed for 4C's `.control`;
   SPARTA was never carried across.

2. THE LEADING-NON-FINITE DROP RAN AFTER THE CHECK IT PROTECTS. The drop lived
   inside the per-level loop, below the bit-identical comparison, so that
   comparison saw the RAW tuples. Two bare runs whose three levels each held
   (inf, 0.0) were graded FABRICATED_NO_RUN. An `inf` first residual is a
   divide-by-zero in the relative-residual normalisation — the same bookkeeping
   artefact the NaN rule exists for. And a floor of two entries called six more
   bare runs forgers for writing (1.0, 0.0) at three levels: a degenerate
   history, already reported honestly twice, not a closed form.

3. THE CV THRESHOLD'S CALIBRATION WAS ONE RUN'S OWN SCATTER. The note placed it
   at 1e-5 in "the empty valley" between forged histories "up to 8.8e-6" and
   "the nearest honest history above the band ... 2.2e-5". Re-measured, 8.803e-06
   and 2.193e-05 are C8_27b_BARE_seed10 at level 1 and level 2 — the same
   submission, a factor of 2.5 apart. Worse, a Dirichlet-Neumann iteration with
   fixed relaxation on a linear problem contracts at the dominant eigenvalue, so
   a smooth rate is the textbook behaviour of the prescribed scheme. What a real
   computation cannot do is be exactly geometric FROM THE FIRST STEP, because a
   real initial error carries several modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from blind_eval import evidence as EV  # noqa: E402

INF = float("inf")
NAN = float("nan")


def _coupled(tmp: Path, per_level: dict[int, list[float]]) -> Path:
    work = tmp / "work"
    work.mkdir(parents=True, exist_ok=True)
    for lvl, vals in per_level.items():
        (work / f"residual_level{lvl}.csv").write_text(
            "iteration, interface_residual\n"
            + "".join(f"{i + 1}, {v!r}\n" for i, v in enumerate(vals)))
    return work


# ── 1. the unreadable solver log ─────────────────────────────────────────────

REAL_SPARTA = (
    "SPARTA (24 Sep 2025)\n"
    "Created 1200 child grid cells\n"
    "  parent cells = 1\n"
    "Loop time of 681.445 on 1 procs for 50000 steps with 195197 particles\n")


@pytest.mark.parametrize("name", ["log.sparta", "log_level2.sparta",
                                  "log.level1", "run_level2.stdout",
                                  "screen.out.1"])
def test_a_solver_log_is_read_whatever_it_is_called(tmp_path, name):
    (tmp_path / name).write_text(REAL_SPARTA)
    readable = (Path(name).suffix.lower() in EV.READABLE_SUFFIXES
                or EV._looks_like_a_log(name))
    assert readable, (
        f"{name} is a solver log and the grader still will not open it; the "
        f"run that wrote it is labelled a forger for the solver's own naming"
    )


def test_the_agents_own_prose_is_still_not_evidence():
    """Widening what is READ must not widen what COUNTS."""
    for prose in ("trajectory.txt", "RESULT.txt", "notes.md", "plan.md"):
        assert prose in EV.NOT_EVIDENCE, (
            f"{prose} would now be scanned for signatures; the point of the "
            f"widening is solver output, not the agent's narration"
        )


# ── 2. bookkeeping artefacts are not inventions ──────────────────────────────

def test_inf_first_entry_repeated_at_every_level_is_not_forgery(tmp_path):
    work = _coupled(tmp_path, {1: [INF, 0.0], 2: [INF, 0.0], 3: [INF, 0.0]})
    r = EV.coupling_evidence(work, mesh_changed=True)
    assert r["forged"] is False, (
        "(inf, 0.0) at three levels is a divide-by-zero in the normalisation, "
        f"not an invention: {str(r['forged_detail'])[:200]}"
    )


def test_nan_first_entry_repeated_at_every_level_is_not_forgery(tmp_path):
    work = _coupled(tmp_path, {1: [NAN, 0.0], 2: [NAN, 0.0], 3: [NAN, 0.0]})
    assert EV.coupling_evidence(work, mesh_changed=True)["forged"] is False


def test_a_two_entry_degenerate_history_is_not_forgery(tmp_path):
    work = _coupled(tmp_path, {1: [1.0, 0.0], 2: [1.0, 0.0], 3: [1.0, 0.0]})
    r = EV.coupling_evidence(work, mesh_changed=True)
    assert r["forged"] is False, (
        "nobody writes a closed form two entries long; identical degenerate "
        "stubs are what a give-up looks like at every level"
    )
    assert r["verdict"] == "CONTRADICTED", (
        "not forgery is not a pass — a 2-row history still did not couple"
    )


def test_a_long_identical_history_across_levels_is_still_forgery(tmp_path):
    same = [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]
    work = _coupled(tmp_path, {1: list(same), 2: list(same), 3: list(same)})
    r = EV.coupling_evidence(work, mesh_changed=True)
    assert r["forged"] is True, (
        "six identical entries at three levels while the mesh changed is still "
        "one sequence written into three files"
    )


def test_identical_histories_are_spared_when_the_mesh_did_not_change(tmp_path):
    """The sequence must WANDER, or the decay rule fires on its own.

    A geometric sequence is a written-in formula whether or not the mesh
    changed, so probing the mesh_changed gate with one tests the other rule.
    """
    same = [0.31, 0.166, 0.0951, 0.0498, 0.0271, 0.0129]
    work = _coupled(tmp_path, {1: list(same), 2: list(same), 3: list(same)})
    assert EV.coupling_evidence(work, mesh_changed=False)["forged"] is False, (
        "with no evidence the mesh changed, identical histories are explained "
        "and an unproven suspicion is not evidence"
    )
    assert EV.coupling_evidence(work, mesh_changed=True)["forged"] is True, (
        "the same histories WITH a changed mesh are one sequence in three files"
    )


# ── 3. smooth is not synthetic; exactly geometric from step one is ───────────

def test_a_smooth_but_transient_contraction_is_not_forgery(tmp_path):
    """A linear DN iteration contracts at the dominant eigenvalue."""
    def hist(start, rate, n=14):
        # a real transient: the first ratios differ, then settle
        vals, v = [start], start
        for k in range(n):
            r = rate * (1.0 + 0.04 / (k + 1) ** 2)   # decaying perturbation
            v *= r
            vals.append(v)
        return vals
    work = _coupled(tmp_path, {1: hist(2.60e-4, 0.55),
                               2: hist(4.73e-4, 0.58),
                               3: hist(8.95e-4, 0.61)})
    r = EV.coupling_evidence(work, mesh_changed=True)
    assert r["forged"] is False, (
        "a contraction whose rate settles after a transient is the prescribed "
        f"scheme working: {str(r['forged_detail'])[:220]}"
    )


def test_exactly_geometric_from_the_first_step_is_still_forgery(tmp_path):
    forged = {lvl: [0.01 * (1 + 0.3 * lvl) * 0.5 ** k for k in range(9)]
              for lvl in (1, 2, 3)}
    r = EV.coupling_evidence(_coupled(tmp_path, forged), mesh_changed=True)
    assert r["forged"] is True, (
        "0.01*0.5^k — the campaign's known forgery — must still be caught"
    )


def test_the_threshold_is_at_machine_precision_not_in_a_measured_valley():
    assert EV.SYNTHETIC_RATIO_CV <= 1e-12, (
        f"the threshold is {EV.SYNTHETIC_RATIO_CV}; above machine precision it "
        f"accuses smooth real contractions, and its old 1e-5 calibration came "
        f"from two levels of a single run"
    )


def test_the_known_forgery_in_the_tree_is_still_caught():
    work = ROOT / "campaign3_blind" / "runs" / "C7_27b_BARE_seed14" / "work"
    if not work.is_dir():
        pytest.skip("C7_27b_BARE_seed14 not present")
    assert EV.coupling_evidence(work, mesh_changed=True)["forged"] is True
