"""ONE problem distribution. Development samples it; evaluation resamples it.

THE DESIGN
----------
There is a single distribution of problems — the operator families, their
parameter ranges, the blind protocol, the output contract. Both phases draw from
it. The difference is only WHEN and WITH WHICH SEED:

  * development draws are the instances that exist now. They are used to refine
    the knowledge, run post-mortems and judge whether it has converged.
  * the evaluation draw is a FRESH sample from the same distribution, taken
    after the freeze.

Same distribution matters: it is what makes the evaluation measure the same
capability the development phase measured, rather than a different task that
happens to be harder or easier. New sample matters just as much: it is what makes
the number mean something about the tool instead of about the tuning.

THE GAP THIS CLOSES
-------------------
Nothing distinguished the two. Every instance sat in `problems/` and the runner
had no notion of phase, so an evaluation could be graded on the very instances
the knowledge was shaped against. That is contamination of the most damaging
kind — not a leaked answer but a leaked problem — and it is invisible in the
results, because every number still looks fine.

It also cannot be repaired afterwards by intending to be careful. Once an agent
has been run on an instance and a pitfall has been written or corrected because
of what happened, that instance is spent for grading. So the distinction has to
be enforced by the code that runs the campaign, before development proceeds.

WHAT IS ENFORCED
----------------
`assert_evaluation_is_clean()` refuses an evaluation run unless the knowledge is
frozen and no instance being graded coincides with a spent one — by id, or by
the fingerprint of its task text, so copying a development problem under a new
name does not launder it. A rule that relies on somebody remembering it is not a
rule.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FREEZE_MARKER = HERE / "FROZEN.json"

# WHY RESAMPLING ALONE IS NOT ENOUGH
#
# A fresh draw protects against grading an instance the knowledge was tuned on.
# It does NOT protect against the knowledge containing the answer to instances in
# general — and that is a separate leak with the same effect. If a development
# post-mortem writes "for this coefficient the order comes out 1.997" or ships an
# exact solution or a tuned relaxation parameter, then every future draw from the
# distribution is compromised, because the agent can read the answer out of the
# tool being evaluated rather than computing it.
#
# So the knowledge gate must carry only GENERAL facts about the codes: what a
# keyword means in the installed version, which element locks, what a failure
# looks like, which default changed. Never a solution, never a measured result
# from one of our own runs, never a parameter tuned for one geometry.
#
# That is enforced separately and continuously, not here: openPASO's own
# `tests/test_knowledge_not_contaminated.py` is a merge gate, and it caught
# exactly these three shapes during development — measured convergence tables
# served through `prepare_simulation`, an exact solution `T(x) = 100*(1-x)`
# shipped in the coupling knowledge, and a manufactured field with its own EOCs
# in a DUNE payload. Resampling and decontamination are independent
# requirements; passing one says nothing about the other.

# Instances already drawn and used for development. Spent for grading.
DEVELOPMENT: dict[str, str] = {
    "B1": "NGSolve, anisotropic diffusion, constant SPD tensor, 2D",
    "B2": "deal.II, variable-coefficient diffusion, 2D",
    "B3": "FEniCSx, nonlinear diffusion a(u) = 1 + u^2/2, 2D",
    "B4": "3D single-code extension",
    "B5": "vector elasticity extension",
    "B6": "3D extension",
    "B7": "single-code extension",
    "D1": "FEniCSx + NGSolve, domain-decomposed diffusion, 2D",
    "D2": "FEniCSx + deal.II, domain-decomposed diffusion, 3D",
    "D3": "FEniCSx + Kratos, two-material conduction, conductivity jump 1:4, 2D",
    "D4": "FEniCSx + deal.II, domain-decomposed linear elasticity (vector), 2D",
    # D5-D8 WERE MISSING FROM THIS LIST, AND THEY ARE SPENT.
    #
    # path_readiness.json records all FIFTEEN instances as walked non-blind with
    # the results discarded, and the walks changed the tool under test — a
    # Kratos Neumann participant, a 3-D deal.II participant, transient
    # participants, a non-rectangular-subdomain mesher, a bent-interface
    # arclength exchange, and a heap-corruption fix in the 3-D flux
    # post-processing that was silent in 2-D and fatal in 3-D. Those are
    # instance-caused changes to the thing being measured.
    #
    # With them absent, assert_evaluation_is_clean would have let D5-D8 be
    # graded as held-out. This list must be derivable from the walk record, not
    # maintained by hand — see the assertion below, which now enforces that.
    "D5": "FEniCSx + scikit-fem, notched non-convex subdomain, bent "
          "L-polyline interface, four materials, 2D",
    "D6": "NGSolve + Kratos, conductivity contrast 1:1000, role assignment "
          "forced, 2D",
    "D7": "FEniCSx + NGSolve, genuinely different operators either side, 2D",
    "D8": "FEniCSx + deal.II, transient two-material conduction, waveform "
          "relaxation over the space-time trace",
    # ── Round 1 of the balanced matrix, run 2026-08-15 on the 27B, both arms.
    # Every one of these is SPENT: the round provoked committed changes to the
    # thing under test (write_file sandbox confinement, tool errors becoming
    # observations, probe-grid set matching, and the knowledge fixes the
    # post-mortem produces). They were absent from this list until an audit
    # found it — with them missing, assert_evaluation_is_clean would have
    # waved all 32 through as held-out.
    "FE1": "FEniCSx, nonlinear diffusion, 2D",
    "FE2": "FEniCSx, near-incompressible linear elasticity (nu = 0.49999), 2D",
    "DL1": "deal.II, variable-coefficient diffusion, 2D",
    "DL2": "deal.II, linear elasticity, 3D",
    "NG1": "NGSolve, anisotropic diffusion, 2D",
    "NG2": "NGSolve, steady incompressible Navier-Stokes, 2D",
    "SK1": "scikit-fem, Stokes flow, mixed velocity-pressure, 2D",
    "SK2": "scikit-fem, biharmonic (clamped plate), 2D",
    "KR1": "Kratos, linear elasticity, 3D",
    "KR2": "Kratos, steady diffusion on a curved (circular) domain, 2D",
    "DU1": "DUNE, orthotropic diffusion, continuous Galerkin, 2D",
    "DU2": "DUNE, advection-diffusion, discontinuous Galerkin, 2D",
    "FB1": "FEBio, linear elasticity, plane strain, 2D",
    "FB2": "FEBio, quasi-static viscoelasticity, transient load, 2D",
    "FC1": "4C, transient heat conduction, 2D",
    "FC2": "4C, near-incompressible linear elasticity (nu = 0.4999), 2D",
    "SP1": "SPARTA, plane Couette flow, wall shear stress (DSMC), band-only",
    "SP2": "SPARTA, parallel-wall heat conduction, wall heat flux (DSMC), "
           "band-only",
    "C1": "4C + FEniCSx, steady thermoelasticity, temperature and "
          "displacement transmitted together, 2D",
    "C2": "4C + Kratos, two-material conduction, severe contrast, 2D",
    "C3": "DUNE + 4C, reaction-diffusion coupled to diffusion, 2D",
    "C4": "FEniCSx + deal.II, transient two-material heat conduction, 2D",
    "C5": "scikit-fem + FEniCSx, four-material conduction, notched domain, "
          "bent interface, 2D",
    "C6": "deal.II + NGSolve, conjugate heat transfer, anisotropic tensor "
          "jump, 2D",
    "C7": "FEBio + deal.II, two-material elasticity, shear-modulus jump, 2D",
    "C8": "NGSolve + Kratos, two-material conduction, severe contrast, 2D",
    "C9": "NGSolve + scikit-fem, two-material linear elasticity, 2D",
    "C10": "Kratos + DUNE, two-material conduction, planar interface, 3D",
    "C11": "FEBio + scikit-fem, two-material elasticity, shear jump, 2D",
    "C12": "DUNE + FEBio, two-material elasticity, shear jump, 2D",
    "C13": "FEniCSx + SPARTA, conjugate heat transfer, conducting solid and "
           "rarefied argon (FEM-DSMC), band-only, outside the balance pool",
    "C14": "4C + FEniCSx, steady fluid-structure interaction, graded against "
           "a monolithic reference, outside the balance pool",
}


def _walked_but_unlisted() -> list[str]:
    """Instances the walk record says are spent but DEVELOPMENT omits.

    The hand-maintained list drifted once already, by four coupled instances,
    which is the failure this catches: an omission here does not raise, it
    silently reclassifies a burnt problem as held-out.
    """
    import json
    from pathlib import Path as _P
    rec = _P(__file__).resolve().parent / "path_readiness.json"
    if not rec.is_file():
        return []
    try:
        walked = json.loads(rec.read_text()).get("path_verified", {})
    except (json.JSONDecodeError, OSError):
        return []
    return sorted(k for k, v in walked.items() if v and k not in DEVELOPMENT)


class SpentListDriftError(RuntimeError):
    """DEVELOPMENT omits an instance the walk record says is spent."""


def assert_spent_list_current() -> None:
    """Fail loudly if a walked instance is missing from DEVELOPMENT.

    `_walked_but_unlisted` existed for exactly this and was never called by
    anything — dead code whose docstring claimed a guarantee the module did
    not provide, while the comment above claimed 'see the assertion below,
    which now enforces that' and there was no assertion. An audit found all
    32 instances of the balanced round unlisted; every one would have been
    graded as held-out.

    Called at import time, so nothing can use this module's phase logic
    against a stale list.
    """
    drift = _walked_but_unlisted()
    if drift:
        raise SpentListDriftError(
            f"path_readiness.json records {len(drift)} walked instance(s) "
            f"that DEVELOPMENT does not list as spent: {drift}. A walked "
            f"instance is burnt — add it to DEVELOPMENT (with a description) "
            f"before any evaluation draw, or the evaluation gate will treat "
            f"it as held-out.")


assert_spent_list_current()


def held_out_spec(seed: int) -> dict:
    """What the evaluation draw must vary, and what it must not.

    VARY, so the instance is genuinely new: the manufactured field, the
    coefficients, the mesh sequence, the interface position, the code pairing,
    and the subdomain extents.

    HOLD FIXED, so the evaluation measures the same capability the development
    phase measured rather than a different task: the operator families, the
    blind protocol (no exact solution anywhere in the prompt), the probe-grid
    output contract, and the grading rule.

    The seed is recorded in the freeze marker, so the draw is reproducible by a
    reader and cannot be re-rolled quietly until it flatters the result.
    """
    return {
        "seed": seed,
        "families": {
            "single_code": ["anisotropic diffusion", "variable-coefficient "
                            "diffusion", "nonlinear diffusion",
                            "convection-diffusion", "reaction-diffusion",
                            "helmholtz", "vector elasticity"],
            "coupled": ["domain-decomposed diffusion",
                        "two-material conduction with a conductivity jump",
                        "domain-decomposed linear elasticity"],
        },
        "must_vary": ["manufactured field", "coefficients", "mesh sequence",
                      "interface position", "code pairing", "subdomain extents"],
        "must_hold": ["operator families", "blind protocol",
                      "probe-grid output contract", "grading rule"],
        "must_not_reuse": sorted(DEVELOPMENT),
        # The geometry constraint that already applies to the coupled set: it
        # must not be a case openPASO ships a pre-built solver for, or the
        # comparison measures whether the tool contains the test.
        "coupled_geometry": "not the unit square split at x = 1/2, which is "
                            "what `coupled_solve` hard-codes",
    }


def instance_fingerprint(problem_dir: Path) -> str:
    """Identity of a built instance, from its task text.

    Used to prove an evaluation instance is not a development one even if it has
    been renamed — comparing ids alone would miss a copy under a new name.
    """
    task = problem_dir / "task.txt"
    if not task.is_file():
        return ""
    return hashlib.sha256(task.read_bytes()).hexdigest()


def development_fingerprints(problems_root: Path | None = None) -> dict[str, str]:
    root = problems_root or (HERE / "problems")
    out = {}
    for pid in DEVELOPMENT:
        d = root / pid
        if d.is_dir():
            fp = instance_fingerprint(d)
            if fp:
                out[pid] = fp
    return out


class EvaluationNotCleanError(RuntimeError):
    """The evaluation cannot proceed without invalidating its own result."""


def assert_evaluation_is_clean(eval_root: Path,
                              problems_root: Path | None = None) -> list[str]:
    """Refuse an evaluation that would grade development instances.

    Two checks, both structural rather than advisory:

      1. The knowledge must be FROZEN. Without a freeze marker there is no
         moment after which development stopped, so "held out" means nothing —
         the knowledge could have been changed in response to these very
         problems.
      2. No evaluation instance may match a development instance, by id OR by
         the fingerprint of its task text, so a rename does not launder it.

    Returns the list of evaluation instance ids on success. Raises rather than
    warning, because a warning in a long pipeline is a thing nobody reads.
    """
    if not FREEZE_MARKER.is_file():
        raise EvaluationNotCleanError(
            f"no freeze marker at {FREEZE_MARKER.name}: the knowledge has not "
            f"been frozen, so nothing is held out from it. Freeze first, "
            f"recording the commit and the draw seed, then draw the evaluation "
            f"set.")

    marker = json.loads(FREEZE_MARKER.read_text())
    dev_fps = set(development_fingerprints(problems_root).values())
    dev_ids = set(DEVELOPMENT)

    ids: list[str] = []
    problems: list[str] = []
    for d in sorted(p for p in eval_root.iterdir() if p.is_dir()):
        ids.append(d.name)
        if d.name in dev_ids:
            problems.append(f"{d.name}: reuses a development instance id")
        fp = instance_fingerprint(d)
        if fp and fp in dev_fps:
            problems.append(
                f"{d.name}: its task text is byte-identical to a development "
                f"instance, so renaming it did not make it new")

    if problems:
        raise EvaluationNotCleanError(
            "the evaluation set overlaps the development set, so grading it "
            "would measure knowledge tuned on these very problems:\n  "
            + "\n  ".join(problems))

    if not ids:
        raise EvaluationNotCleanError(
            f"no evaluation instances found under {eval_root}")

    if marker.get("draw_seed") is None:
        raise EvaluationNotCleanError(
            "the freeze marker records no draw seed, so the evaluation draw is "
            "not reproducible and cannot be shown not to have been re-rolled")

    return ids
