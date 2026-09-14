"""Execution evidence: the canonical contract line, and the growth it must show.

Per-code evidence and the coupled residual-history gate are IMPORTED from
`src/blind_eval/evidence.py` of THIS checkout (see loading.py for why the
checkout pinning is load-bearing). That module owns the canonical line

    NDOF = <integer>

accepted as the numeric run signature for EVERY code, so no honest run can be
labelled fabricated for its solver's print style again.

What is NEW here is that the contract line is finally read PER LEVEL and its
numbers are required to make sense together. `fit_order` never sees a mesh: an
agent that refines by 4x per "level" produces error ratios that read as
superconvergence, and nothing upstream could catch it. The NDOF sequence can:
under the prescribed halving it must grow like 2**dim per level (bounds and
their derivation in constants.py). No growth means the same mesh was submitted
as a sequence; 2**(2*dim) growth means the sequence is not the prescribed one.

Task-text honesty: the run-log contract is stated in the single-code and
C-series task texts. The older D-series task texts predate it and do not ask
for run logs, so their absence there cannot be charged to the agent — for
those cells a missing log is reported as `NDOF growth NOT CHECKED`, loudly,
and the per-code signatures still decide the evidence. A present log is
checked in either case.
"""
from __future__ import annotations

from pathlib import Path

from . import constants as C
from .loading import evidence_mod
from .submission import RUN_LOG_FILE


def task_prescribes_run_logs(task_txt: str) -> bool:
    return "run_level" in task_txt and "NDOF" in task_txt


def task_demands_own_solver_output(task_txt: str) -> bool:
    """Did THIS task tell the agent to capture its solver's own output?

    The per-code evidence gate below rejects a coupled submission whose only
    proof of execution is the code-agnostic `NDOF = <integer>` contract line.
    That is a contract breach ONLY where the task asked for more. Measured over
    the three problem roots in this tree:

        problems/       0 of 47 task files demand captured solver output
        problems_dev2/  1 of 1
        problems_dev3/  1 of 1

    So for the older draw — which is most of the graded corpus — the gate
    rejected submissions for doing exactly what they were told, and it did so
    unevenly: 29.7% of bare grade-1 coupled runs against 15.7% of openPASO ones,
    which inflates the measured uplift.

    The rule this restores is the one already written into the comment on that
    gate: "Until the task asks, the grader may not punish." The newer task
    builder does ask, in as many words, so the gate keeps its teeth exactly
    where the agent was given the instruction.
    """
    low = (task_txt or "").lower()
    return any(k in low for k in (
        "console output that", "captured verbatim", "verbatim",
        "own console output", "captured output",
        "redirect the run into the file", "2>&1"))


def run_log_ndofs(work: Path) -> dict:
    """side -> {level: ndof} from run_level<k>[_<side>].log, using the
    canonical contract regex from blind_eval.evidence — one authority."""
    EV = evidence_mod()
    out: dict[str, dict[int, int]] = {}
    for f in sorted(work.rglob("run_level*.log")):
        m = RUN_LOG_FILE.match(f.name)
        if not m:
            continue
        try:
            text = f.read_text(errors="ignore")
        except OSError:
            continue
        # strip what the TERMINAL wrote before matching what the AGENT wrote
        cm = EV.CANONICAL_NDOF.search(EV.strip_terminal_noise(text))
        if not cm:
            continue
        side = (m.group(2) or "-").upper()
        out.setdefault(side, {})[int(m.group(1))] = int(cm.group(1))
    return out


def ndof_growth(ndofs: dict, mesh_N, dim: int, required: bool):
    """Check the per-level NDOF sequence grows like the prescribed halving.

    Returns (fatal_missing, sequence_violations, notes, table):
      * fatal_missing — levels the contract requires a log for and none exists
        ("a level without it counts as not run at all");
      * sequence_violations — ratios outside the 2**dim band, or shrinkage:
        the submitted levels are not the prescribed mesh sequence.
    """
    lo, hi = C.ndof_ratio_bounds(dim)
    nlevels = len(mesh_N or [])
    fatal_missing, violations, notes = [], [], []
    table = {}
    if not ndofs:
        if required:
            fatal_missing.append(
                "no run_level<k>.log with an `NDOF = <integer>` line exists "
                "for any level; the task states a level without it counts as "
                "not run at all")
        else:
            notes.append(
                "NDOF growth NOT CHECKED: this task text predates the "
                "run-log contract and no run_level logs were written; "
                "execution evidence rests on the per-code signatures alone")
        return fatal_missing, violations, notes, table

    for side, per_level in sorted(ndofs.items()):
        table[side] = dict(sorted(per_level.items()))
        if required and nlevels:
            missing = [k for k in range(1, nlevels + 1) if k not in per_level]
            if missing:
                fatal_missing.append(
                    f"side {side}: no NDOF contract line for level(s) "
                    f"{missing}; the task states a level without it counts "
                    f"as not run at all")
        lv = sorted(per_level)
        for a, b in zip(lv, lv[1:]):
            if b != a + 1:
                continue
            n0, n1 = per_level[a], per_level[b]
            if n0 <= 0:
                violations.append(f"side {side}: NDOF {n0} at level {a} is "
                                  f"not a run")
                continue
            r = n1 / n0
            if not (lo <= r <= hi):
                violations.append(
                    f"side {side}: NDOF grows {n0} -> {n1} (x{r:.2f}) from "
                    f"level {a} to {b}; the prescribed halving must grow it "
                    f"like 2**{dim} (accepted {lo:.1f}..{hi:.1f}). The "
                    f"submitted levels are not the prescribed mesh sequence — "
                    f"this is how a 4x refinement reads as superconvergence "
                    f"and how an unrefined mesh reads as a sequence.")
    return fatal_missing, violations, notes, table


def assess_execution(work: Path, codes: list, coupled: bool, task_txt: str,
                     mesh_N, dim: int, iface_tol: float,
                     claimed_iters) -> dict:
    """The whole execution-evidence question for one run.

    Returns a dict with:
      fatal    — None, or the outcome the evidence forces
                 ("FABRICATED_NO_RUN" | "MALFORMED_SUBMISSION")
      reasons  — machine-readable reasons when fatal
      per_code / coupling / ndof / notes — the evidence trail
    """
    EV = evidence_mod()
    # THE MESH FACT MUST BE ESTABLISHED BEFORE THE HISTORY IS JUDGED.
    #
    # A residual history that is bit-identical across levels is a forgery only
    # if the mesh actually changed between those levels; if the agent submitted
    # one mesh three times, identical histories are honest and the defect is the
    # mesh sequence. So the NDOF sequence is read FIRST and passed in. `None`
    # (no readable run logs) means we cannot establish it, and then no forgery
    # is claimed.
    ndofs = run_log_ndofs(work)
    mesh_changed = None
    if ndofs:
        per_side = [sorted(v.items()) for v in ndofs.values()]
        seqs = [[n for _, n in s] for s in per_side if len(s) > 1]
        if seqs:
            mesh_changed = any(s[-1] > s[0] * 1.5 for s in seqs)
    rep = EV.assess(work, codes, coupled=coupled, iface_tol=iface_tol,
                    claimed_iterations=claimed_iters,
                    mesh_changed=mesh_changed)
    out = {
        "verdict": rep.verdict,
        "per_code": [{"code": e.code, "verdict": e.verdict,
                      "files": e.files[:4], "detail": e.detail[:300]}
                     for e in rep.per_code],
        "coupling": rep.coupling,
        "notes": list(rep.notes),
        "fatal": None,
        "reasons": [],
    }

    # POSITIVE EVIDENCE OF INVENTION IS CHECKED FIRST, AND ONLY IT EARNS THE
    # FABRICATION LABEL ON A COUPLED CELL.
    #
    # This branch must precede every contract check below. Otherwise the
    # relabelling done further down would hand the campaign's one ADMITTED
    # monolith a paperwork verdict: C7_27b_BARE_seed2, whose own
    # IMPLEMENTATION_NOTES.txt says "The coupling iterations shown are simulated
    # based on the monolithic solution, rather than actual separate solves",
    # also has canonical-only run logs, so it would exit at the
    # shared-evidence branch and never reach its forged history. A forgery must
    # not be rescued by a second, milder defect.
    if coupled and rep.coupling.get("forged"):
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = ["SYNTHETIC_RESIDUAL_HISTORY"]
        out["notes"].append(rep.coupling.get("forged_detail", ""))
        # "NO_RUN" IS OFTEN THE WRONG WORD, AND THE PAPER REPORTS A RATE.
        #
        # Measured over the 50 coupled runs in this tree whose residual
        # histories are bit-identical across three or more levels -- the
        # condition that lands here -- about 19 submitted identically-zero
        # fields or none at all, and about 21 submitted fields that are
        # NONZERO AND DIFFERENT AT EVERY LEVEL. The second group ran a solver
        # and refined it; what they invented was one required file.
        #
        # C2_27b_MCP_seed502 and C2_27b_BARE_seed502 are the pair that forced
        # this. Both wrote fifty rows of exactly 1.0 at all three levels and
        # both were graded FABRICATED_NO_RUN. The bare one is zero everywhere.
        # The openPASO one's side A peaks at 1.265e-01 / 1.320e-01 / 1.323e-01
        # and its side B at 2.280e-03 / 2.334e-03 / 2.342e-03, within a few
        # percent of an independently computed reference.
        #
        # The outcome is NOT softened -- inventing a deliverable is an
        # integrity violation whatever else is true, and this stays fatal and
        # stays in every denominator. What changes is that the two are now
        # COUNTABLE apart, so a reported fabrication rate can say which it
        # means instead of pooling them.
        try:
            field_state = EV.per_level_field_state(work)
        except Exception:                        # noqa: BLE001
            field_state = {"verdict": "NOT_ASSESSED"}
        out["field_state"] = field_state
        out["invention_scope"] = (
            "ARTEFACT_ONLY" if field_state.get("verdict") == "REAL_AND_REFINED"
            else "WHOLE_SUBMISSION"
            if field_state.get("verdict") in ("ALL_ZERO", "NO_FIELD_FILES")
            else "UNDETERMINED")
        if out["invention_scope"] == "ARTEFACT_ONLY":
            out["notes"].append(
                "INVENTION IS CONFINED TO THE COUPLING HISTORY: the per-level "
                "fields are nonzero and different at every level (" +
                ", ".join(f"L{k}={v:.4e}" for k, v in
                          sorted(field_state.get("per_level_peak", {}).items()))
                + "), so a solver ran and refined. The run still fails -- a "
                "written-in deliverable is an integrity violation -- but any "
                "fabrication rate quoting this run must not read it as "
                "'nothing ran'.")
        return out

    unproven = [e.code for e in rep.per_code if e.verdict != "PROVEN"]
    if unproven:
        # THE COUPLED PRECEDENT, APPLIED HERE: UNPROVEN IS NOT INVENTED.
        # Where the task DEMANDS the named code's own captured output, a
        # submission whose numbers may be real but whose provenance is not
        # the named code's is a contract failure, not a forgery -- measured:
        # FB2_27b_MCP_seed3902 solved the FEBio task with NGSolve, honestly
        # and wrongly, and "no run happened" is simply false of it. The
        # FABRICATED label stays for gradings under the older task text,
        # where this branch fires only when NO code-specific signature exists
        # anywhere (the numpy/scipy stand-in class) -- unchanged, so as-run
        # regrades of the historical corpus are bit-identical.
        if task_demands_own_solver_output(task_txt):
            out["fatal"] = "MALFORMED_SUBMISSION"
            out["reasons"] = [f"NO_EXECUTION_EVIDENCE({c})" for c in unproven]
            out["notes"].append(
                "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: "
                "the task required the named code's own captured output and "
                "none of its signatures appear, so the numbers cannot be "
                "credited to that code. Wrong or missing provenance is not "
                "proof of invention.")
            return out
        out["fatal"] = "FABRICATED_NO_RUN"
        out["reasons"] = [f"NO_EXECUTION_EVIDENCE({c})" for c in unproven]
        return out
    # ONE FILE CANNOT BE TWO CODES' OUTPUT.
    #
    # Each per-code verdict can be PROVEN individually while the SAME file
    # proves both, because the canonical `NDOF = <n>` line is code-agnostic by
    # design. So `unproven` is empty and this branch is the only place the
    # collision can be caught. `assess` sets the flag only when EVERY match is
    # that shared contract line and no code-specific signature exists anywhere
    # in the run, transcript included.
    #
    # C2_27b_MCP_seed15 is why this exists. Its participants import numpy and
    # scipy.sparse — their own docstrings say "direct FEM assembly" — and no
    # file in the run contains a 4C or Kratos token. Its whole execution
    # evidence is fourteen 10-byte files reading `NDOF = <n>`, credited to both
    # prescribed codes at once. Its numbers are genuinely second-order, which
    # is precisely why the previous note was not enough: everything except
    # "did the prescribed codes run" looked right, and that is the one thing a
    # coupled cell exists to measure. I graded it CORRECT earlier today on the
    # strength of its history and interface behaviour; this is the correction.
    #
    # C8_27b_MCP_seed4 is the control: its run logs are also canonical-only,
    # but Kratos's own telemetry ("ResidualBasedLinearStrategy: Setup Dofs
    # Time: 0.00222747 [s]") appears in the run, so the flag is not set and it
    # stays CORRECT — independently confirmed by reproducing its NGSolve DOF
    # sequence 60/212/795 from scratch with netgen.
    # THE AGENT COMPLIED WITH THE CONTRACT AS WRITTEN. THAT IS NOT FORGERY.
    #
    # Labelling this FABRICATED_NO_RUN charges the agent with inventing numbers
    # for doing exactly what the task asked. Measured: no C-series task text
    # ever required a participant to write its solver's OWN output — the stated
    # requirement is the code-agnostic line `NDOF = <integer>` in
    # run_level<k>_<side>.log, and nothing more. 102 runs hit this branch, 67
    # bare and 35 openPASO, so the mislabel inflates the reported fabrication rate
    # of BOTH arms with a requirement that was never communicated.
    #
    # It stays FATAL, and the cell is not a success: on a coupled cell the
    # canonical line genuinely cannot show that two DIFFERENT codes ran, so the
    # claim is unproven. But unproven is not invented. MALFORMED_SUBMISSION is
    # the bucket this grader already uses for a deliverable that does not carry
    # what the verdict needs (RUN_LOG_CONTRACT_UNMET, three lines down, was
    # moved here for the identical reason), and it stays in every denominator —
    # `aggregate` counts it — so nothing is hidden by the move.
    #
    # The real repair is UPSTREAM and is not in this file: the coupled task must
    # require each participant to capture its solver's own output. Then a
    # missing signature is a genuine contract breach and attribution becomes
    # possible. Until the task asks, the grader may not punish.
    if coupled and getattr(rep, "shared_evidence_fatal", False):
        # NOW CONDITIONAL ON WHAT THIS TASK ACTUALLY ASKED FOR.
        #
        # The note below already said the repair was upstream and that "until
        # the task asks, the grader may not punish" — and then punished. The
        # upstream repair has since landed in the task builder, so the gate can
        # finally be what the note describes: fatal where the task demanded the
        # solver's own captured output, a recorded finding where it did not.
        #
        # Measured: 0 of 47 tasks in the older draw demand it, against 1 of 1
        # in each newer draw. This was the largest rejection class in the
        # corpus (86 decided) and fell 29.7% bare against 15.7% openPASO, so
        # correcting it lowers the measured uplift rather than flattering it.
        if task_demands_own_solver_output(task_txt):
            out["fatal"] = "MALFORMED_SUBMISSION"
            out["reasons"] = ["NO_PER_CODE_EXECUTION_EVIDENCE"]
            out["notes"].append(
                "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: "
                "the task required each participant to capture its own "
                "solver's output and the submission carries only the "
                "code-agnostic contract line, so which code produced it "
                "cannot be established. The coupled claim is unproven, not "
                "shown to be invented.")
            return out
        # MACHINE-READABLE, NOT ONLY PROSE.
        #
        # A note in `notes` is invisible to anything that counts outcomes, and
        # the test that guarded this gate said exactly why that matters: "an
        # unattributable coupled run would now be gradeable as CORRECT". It
        # would. The repair is not to punish compliance again — it is to make
        # the hole countable, so any coupling claim can exclude these runs and
        # a reader can see how many there were.
        out["per_code_attribution"] = "UNPROVEN"
        out["notes"].append(
            "PER-CODE ATTRIBUTION UNPROVEN, NOT PUNISHED: the only execution "
            "evidence is the code-agnostic `NDOF = <integer>` line, which is "
            "exactly and only what this task asked for — it never required "
            "each participant to capture its own solver's output. The coupled "
            "claim therefore cannot be attributed to the two named codes, and "
            "that is the harness's gap, not the submission's. Grading "
            "continues on the numbers.")
    if coupled and rep.coupling.get("verdict") != "PROVEN":
        # A DIVERGING ITERATION IS A WRONG ANSWER, NOT A LIE.
        #
        # Reached only when `forged` above did NOT fire, so every remaining
        # complaint is a real numerical failure: too few iterations, a residual
        # above the prescribed tolerance, an insufficient decrease, a constant
        # residual, a mid-history NaN, or no history file at all. An earlier
        # kind of thing — under-converged, not invented — and this is where
        # they came from. MEASURED MYSELF over all 396 coupled runs in the
        # graded tree: 329 have a coupling history that fails, and 299 of those
        # (91%) carry NO positive sign of invention. Only 30 do (22 BARE,
        # 8 MCP). The run still fails fatally; only the accusation is dropped.
        #
        # An earlier audit reported this as "27 of 71". That pair is NOT
        # reproducible from any grade artefact in the repo and is not cited
        # here; the numbers above are re-derived from the run trees.
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["COUPLING_EVIDENCE_" +
                          str(rep.coupling.get("verdict", "ABSENT"))]
        out["notes"].append(
            "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: the "
            "coupling evidence is deficient but carries no positive sign of "
            "invention (no closed-form residual decay)")
        return out

    out.setdefault("per_code_attribution", "PROVEN")
    required = task_prescribes_run_logs(task_txt)
    fatal_missing, violations, notes, table = ndof_growth(
        ndofs, mesh_N, dim, required)         # read once, above
    # A CELL WHOSE TASK SAYS THE GRID DOES NOT REFINE MUST NOT BE FAILED FOR
    # NOT REFINING ITS GRID.
    #
    # ndof_growth demands the reported count grow like 2**dim per level, which
    # is right wherever a mesh is halved. The DSMC cells prescribe the
    # opposite, in as many words: "THE GRID IS FIXED AND THE PARTICLE COUNT
    # REFINES ... do NOT refine the grid instead", with the particle count
    # quadrupling per level, and the task itself says "particle count plays the
    # role the degree-of-freedom count plays elsewhere". Their EXECUTION LOG
    # clause still asks for an NDOF line, so the growth rule ran on them.
    #
    # An agent that reports the grid cell count it was told to hold fixed was
    # therefore charged MESH_SEQUENCE_NOT_PRESCRIBED for obeying the task, and
    # one that reports particles passed only because quadrupling happens to
    # land inside the 2.2-7.2 band for dim=2. Which quantity the line carries
    # cannot be told apart from the number alone, so on these cells the
    # observation is RECORDED and not charged — the same rule as everywhere
    # else here: the grader may not punish what the task did not ask.
    if "GRID IS FIXED" in (task_txt or "").upper() and violations:
        notes.append(
            "NDOF GROWTH NOT CHARGED: this task prescribes a FIXED grid with a "
            "refining particle count, so a constant or non-2**dim NDOF "
            "sequence is what it asked for. Recorded, not charged: "
            + "; ".join(str(v)[:160] for v in violations))
        out.setdefault("notes", []).extend(notes)
        violations = []
    out["ndof"] = table
    out["notes"].extend(notes)
    if fatal_missing:
        # A MISSING LOG LINE IS A CONTRACT FAILURE, NOT AN ACCUSATION.
        #
        # This branch is only reached AFTER every named code's execution
        # evidence came back PROVEN and, for a coupled cell, after the
        # partitioned-iteration history came back PROVEN. Labelling it
        # FABRICATED_NO_RUN then contradicts the two checks immediately above:
        # the record says "3 level(s) with a well-formed, converging
        # partitioned-iteration residual history" and the outcome says the
        # numbers were invented.
        #
        # Five runs sit in exactly that state — C5_MCP_s14, C4_MCP_s15,
        # C12_MCP_s15, C3_BARE_s4, C7_BARE_s9 — three openPASO and two bare, so
        # the conflation inflates the reported fabrication rate of BOTH arms
        # with paperwork. The campaign reports that rate as a headline.
        #
        # The run still FAILS, and fails fatally: the task states that a level
        # without its log counts as not run, and nothing here softens that.
        # Only the name changes, to the one the very next branch already uses
        # for a sibling contract failure (MESH_SEQUENCE_NOT_PRESCRIBED). What
        # is lost is the claim that the agent invented its numbers, which in
        # these five cases the evidence positively contradicts.
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["RUN_LOG_CONTRACT_UNMET"]
        out["notes"].extend(fatal_missing)
        out["notes"].append(
            "graded MALFORMED_SUBMISSION rather than FABRICATED_NO_RUN: the "
            "execution and coupling evidence above are PROVEN, so the defect "
            "is the missing contract line, not invented numbers")
        return out
    if violations:
        out["fatal"] = "MALFORMED_SUBMISSION"
        out["reasons"] = ["MESH_SEQUENCE_NOT_PRESCRIBED"]
        out["notes"].extend(violations)
        return out
    return out
