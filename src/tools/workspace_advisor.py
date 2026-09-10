"""Workspace advisor: OASiS's checks on what an agent leaves behind.

EVERY CHECK IN THIS MODULE IS PRODUCT CODE, NOT EVALUATION CODE. The external
harness that drives development runs (langgraph_eval/agent.py) fires these at
its hook points -- a file written, a shell command's output, a result set
delivered -- but defines none of them:
the boundary, set explicitly on 2026-09-03, is that the harness carries no
domain or contract knowledge of its own, because any capability that lives
only in the runner is not OASiS's and cannot be claimed, shipped, or exercised
by a fresh draw. Sibling of tools/result_audit.py, which owns the numeric
self-consistency audit; this module owns the earlier, cheaper moments -- the
script as written, the artefact as it lands, the error as it is read.

Each check states the measured failure it exists for in its own docstring, is
calibrated against the real run that motivated it AND against a reference
result set verified correct against an independent reference, and returns ""
when it has nothing to say.
"""
from __future__ import annotations

import re as _re_mod
from pathlib import Path

# Deliverable discovery is shared with the audit and knows no task's naming
# scheme: level-indexed files are read from the agent's own names and
# classified by content (field / interface / history / run log).
from tools.result_audit import (          # noqa: E402
    _DOF_LINE, _LEVEL_FILE as _A_LEVEL_FILE, _csv_role, _field_files,
    _history_files, _interface_files, _level_of, _side_of)

def _flat(v):
    """Every scalar in a nested list, however the participant shaped it."""
    if isinstance(v, (list, tuple)):
        for x in v:
            yield from _flat(x)
    elif isinstance(v, (int, float)):
        yield v

def _work_on_disk_contradicting_a_give_up(work: Path) -> str:
    """A give-up written on top of a finished run — reported structurally.

    Measured twice. Ten coupled runs drove a coupling to convergence, hit a
    flux-balance finding, and filed a could-not-finish report with a median 69%
    of their budget unspent. The tool's reply was then rewritten to say NOT
    VERIFIED and NOT A RESULT are different verdicts and to write the
    deliverables first — and in the very next probe three of six runs did it
    again anyway, one of them stating in its own words that the coupling
    converged in ~7 iterations before giving up on the balance check.

    Wording the imperative better does not work; the same lesson as the audit
    tool that was called by 1 of 51 runs when merely offered. So this reads the
    agent's OWN files at the moment it writes a give-up and says what is
    already there. It supplies no knowledge, no method and no numbers — it only
    refuses to let a finished run be filed as an unfinished one silently.
    """
    import csv as _csv
    sol = _field_files(work)
    iface = _interface_files(work)
    resid = _history_files(work)
    # A PARTICIPANT'S OWN EXPORT COUNTS AS WORK. One run of a later batch of
    # development runs wrote no CSV at all and still had exports.json for both
    # halves of level 1 — a solve that ran and an interface exchange that
    # completed, filed as could-not-finish. Looking only for the task's
    # deliverables misses exactly the run that did the work and never wrote it
    # down.
    # Non-empty AND not the iteration-1 fallback: a participant that wrote
    # only placeholder zeros has not solved anything, and calling that "work
    # on disk" would be the same crying-wolf that teaches agents to ignore a
    # gate. Checked on that batch's four give-ups: all four carry real numbers
    # (|q| up to 0.98), so none of them is a false positive.
    def _real(q):
        try:
            import json as _json
            d = _json.loads(q.read_text())
        except Exception:                            # noqa: BLE001
            return False
        for key in ("values", "normal_fluxes"):
            for x in _flat(d.get(key) or []):
                if x not in (0, 0.0) and x == x:
                    return True
        return False

    exports = [q for q in sorted(work.rglob("exports.json")) if _real(q)]
    # RAW SOLVER OUTPUT IS WORK TOO. Measured: two runs drove FEBio to NORMAL
    # TERMINATION 8 and 10 times, logged full nodal output through
    # <node_data> into per-step CSV blocks, wrote NO deliverable at all, and
    # filed a could-not-finish report at 32% of their wall budget -- this check
    # stayed silent because its evidence list held only the task's own file
    # names. A solve that terminated normally plus its native output on disk
    # means the ONLY missing step is reading the numbers back at the probe
    # points; that is the largest measured failure class there is, and a
    # give-up over it deserves the same structural contradiction.
    ok_logs, native = [], []
    for q in sorted(work.rglob("*.log")):
        try:
            if _looks_like_captured_output(q.read_text(errors="replace")):
                ok_logs.append(q)
        except OSError:
            pass
    for pat in ("*.csv", "*.xplt", "*.vtu", "*.exo", "*.pvd"):
        for q in sorted(work.rglob(pat)):
            if _A_LEVEL_FILE.match(q.name) and _csv_role(q) != "raw":
                continue
            try:
                if q.suffix == ".csv":
                    head = q.read_text(errors="replace")[:200]
                    if "*Step" in head or "*Data" in head:
                        native.append(q)
                elif q.stat().st_size > 0:
                    native.append(q)
            except OSError:
                pass
    # PARTICIPANT SCRIPTS ARE WORK. Measured: one session wrote both sides'
    # participant scripts (each implementing the imports.json/exports.json
    # exchange), a driver stub whose coupling step was a print saying what
    # it WOULD do, and filed its give-up with no residual history anywhere.
    # Everything above counts artefacts the exchange PRODUCES; a run that
    # stopped one step before producing them read as "nothing on disk" and
    # this gate stayed silent.
    pscripts = []
    for q in sorted(work.rglob("*.py")):
        try:
            c = q.read_text(errors="replace")
        except OSError:
            continue
        if "exports.json" in c and ("imports.json" in c
                                    or "InterfaceData" in c):
            pscripts.append(q.name)

    if not (sol or iface or resid or exports or pscripts
            or (ok_logs and native)):
        return ""                      # nothing on disk: the give-up is honest

    conv = []
    for r in resid:
        try:
            rows = [row for row in _csv.reader(r.open()) if row]
            vals = []
            for row in rows[1:]:
                try:
                    vals.append(float(row[-1]))
                except (ValueError, IndexError):
                    pass
            if vals:
                conv.append((r.name, len(vals), vals[-1]))
        except OSError:
            pass

    bits = []
    if sol:
        bits.append(f"{len(sol)} per-level field file(s)")
    if iface:
        bits.append(f"{len(iface)} per-level interface file(s)")
    if exports and not (sol or iface):
        bits.append(f"{len(exports)} participant exports.json — a solve ran "
                    f"and an interface exchange completed, but none of the "
                    f"task's own output files were written")
    # PARTICIPANTS BUILT, ITERATION NEVER RUN. Structural statement only:
    # what is on disk, what is absent.
    if pscripts and not resid:
        bits.append(
            f"{len(pscripts)} script(s) implementing the "
            f"imports.json/exports.json participant exchange "
            f"({', '.join(pscripts[:4])}) — and NO partitioned-iteration "
            f"residual history anywhere: the participants were built and "
            f"the coupling iteration over them was never run. That "
            f"single remaining step is what stands between the work on "
            f"disk and a result set with coupling evidence.")
    if ok_logs and native and not (sol or iface):
        bits.append(
            f"{len(ok_logs)} solver log(s) with the solver's own successful "
            f"termination and {len(native)} native output file(s) "
            f"(e.g. {native[0].name}) — the solves RAN and their results are "
            f"on disk; the only step missing is reading those values back at "
            f"the task's probe points and writing the deliverable files")
    # SCREEN THE INVENTORY BEFORE ADVERTISING IT. Measured: a run whose three
    # residual histories were written-in (ratio cv 7.0e-14) received this
    # notice listing them as "12 iterations ending at 5.66e-07" — the give-up
    # gate endorsing files the auto-audit was simultaneously calling
    # fabrication, and the agent obeyed the flattering voice and its result
    # was read as fabricated, with no run behind it. The same detector the
    # audit uses screens this list,
    # so the two notices cannot disagree about the same file again.
    flagged = {}
    try:
        from tools.result_audit import residual_findings   # noqa: PLC0415
        for f in residual_findings(work):
            if f.get("sequence") and f.get("finding"):
                flagged[f["sequence"]] = f["finding"].split(":")[0]
    except Exception:                                       # noqa: BLE001
        pass
    # Two classes of finding, two different imperatives. A WRITTEN-IN,
    # CONSTANT or NON-FINITE history is a liability: it reads as fabrication
    # and is worth less than an honest unconverged report — delete it. A TOO-SHORT
    # or BARELY-MOVED history is real but insufficient evidence — extend it,
    # never delete it.
    _fatal = ("WRITTEN-IN", "CONSTANT RESIDUAL", "NON-POSITIVE OR NON-FINITE")
    for name, n, last in conv:
        finding = flagged.get(name, "")
        if any(k in finding for k in _fatal):
            bits.append(
                f"{name}: NOT WORK — {finding}. A written-in, constant or "
                f"non-finite history is a liability in a result set, not "
                f"evidence: it is read as invented, below an honest "
                f"unconverged report. Delete it and either couple for real "
                f"or hand in the honest state.")
        elif finding:
            bits.append(
                f"{name} with {n} iterations ending at {last:.3g} — but "
                f"{finding}. The run behind it is real; the evidence is "
                f"insufficient as it stands, so extend the iteration rather "
                f"than deleting the file.")
        else:
            bits.append(f"{name} with {n} iterations ending at {last:.3g}")
    return (
        "YOU ARE FILING A GIVE-UP ON TOP OF WORK THAT IS ON DISK.\n  "
        + "\n  ".join(bits)
        + "\nA could-not-finish report counts for nothing. A result set built "
          "from the numbers you already have stands on those numbers, PROVIDED "
          "it is complete: every level the task prescribes and, where the "
          "task names two subdomains, both files per level. A result set "
          "missing a level or a side is unusable, worth the same "
          "as no result set, so complete the sequence from what you have "
          "rather than filing part of it. A verification "
          "finding — a flux imbalance, a failed conservation check — is NOT a "
          "reason to withhold a field your solve already produced: those are "
          "different verdicts and only one of them is worth zero. Write the "
          "deliverables the task asks for from what you have, state the "
          "finding alongside them, and keep working on the finding with "
          "whatever time is left."
    )
_REGISTRY_SIG = r"[A-Z][A-Za-z0-9_]*\d+D\d+N"
_REGISTRY_NOT_A_COMPONENT = ("Utility", "Utilities", "Process", "Factory",
                             "Modeler")

def _looks_like_registry_component(name: str) -> bool:
    return not any(w in name for w in _REGISTRY_NOT_A_COMPONENT)

def _REGISTRY_MSG(name: str, where: str) -> str:
    """The one general truth about reaching a Kratos component from Python."""
    return (
        "\n\n[" + where + "]\n"
        "  * `SomeApplication." + name + "` IS NOT HOW A REGISTERED KRATOS "
        "COMPONENT IS REACHED, AND THIS AttributeError IS NOT EVIDENCE THAT "
        + name + " IS MISSING. Kratos elements and conditions live in a C++ "
        "registry and are built BY NAME through a factory on the model part; "
        "none of them is exposed as a Python attribute. Measured on this "
        "install, every one of the three:\n"
        "        LaplacianElement2D3N   python-attribute=False  "
        "factory-by-name=True\n"
        "        ThermalFace2D2N        python-attribute=False  "
        "factory-by-name=True\n"
        "        FluxCondition2D2N      python-attribute=False  "
        "factory-by-name=True\n"
        "    LaplacianElement2D3N is the element your own solve already runs "
        "on, so the AttributeError says nothing whatever about existence. "
        "Build it by name instead:\n"
        "        mp.CreateNewCondition(\"" + name + "\", cid, [n1, n2], prop)\n"
        "        mp.CreateNewElement(\"LaplacianElement2D3N\", eid, "
        "[a, b, c], prop)\n"
        "    DO NOT CHANGE CODES OVER THIS. A previous run on this problem read "
        "this same AttributeError as \"not available in version 10.3.0\", "
        "abandoned the two codes the task prescribes, went looking for a "
        "third, and delivered nothing at all.")

def _registry_attribute_check(written: Path, content: str) -> str:
    """A registered component written as a module attribute never resolves.

    MEASURED, one development run. The served pitfall NAMES the condition, and
    the write-time check hands over the exact factory line, and the run still
    wrote `KM.ConvectionDiffusionApplication.ThermalFace2D2N(condition_id,
    ...)`. Python raised `has no attribute 'ThermalFace2D2N'`; the agent
    concluded the condition does not exist in this Kratos version, considered
    replacing both prescribed codes with FEniCSx, and ran out of clock. The
    condition exists -- so does every other name it would have reached that
    way.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    for m in _re.finditer(r"\.\s*(" + _REGISTRY_SIG + r")\s*\(", content):
        if _looks_like_registry_component(m.group(1)):
            return _REGISTRY_MSG(
                m.group(1),
                "early check of " + written.name + ", read from the script "
                "you just wrote")
    return ""

def _registry_error_check(output: str) -> str:
    """The same truth, keyed on the ERROR the agent actually read.

    This is the channel that matters. The broken constructor in that run was
    not in any file at the end of the run -- all three of its scripts hold
    zero attribute-constructor calls -- so a check that reads what is on disk
    would have missed it. What the agent ended up with was the traceback, and
    the traceback is what it misread.
    """
    import re as _re

    m = _re.search(r"has no attribute ['\"](" + _REGISTRY_SIG + r")['\"]",
                   output)
    if m and _looks_like_registry_component(m.group(1)):
        return _REGISTRY_MSG(
            m.group(1),
            "the command you just ran hit an AttributeError on a Kratos "
            "component name")
    return ""
_LEVEL_FILE = _re_mod.compile(
    r"^(?P<stem>solution)_level(?P<k>\d+)(?P<side>_[AB])?\.csv$")

def _value_column(p: Path) -> list[str] | None:
    """The last column of a probe file, as raw text -- exact by construction."""
    try:
        rows = [r for r in p.read_text(errors="replace").splitlines() if r.strip()]
    except OSError:
        return None
    if len(rows) < 3:
        return None
    if any(c.isalpha() for c in rows[0]):
        rows = rows[1:]
    out = []
    for r in rows:
        parts = r.split(",")
        if len(parts) < 2:
            return None
        out.append(parts[-1].strip())
    return out or None

def _identical_levels_check(workdir: Path, written: Path) -> str:
    """The same field delivered at every level. An order cannot come from it.

    MEASURED, one development run. Its side A is BIT-IDENTICAL at all three
    levels -- max|u_i - u_j| = 0.000e+00 for every pair, peak 0.1332715818041668
    three times -- so it solved subdomain A once and wrote the same 1936 values
    into the side-A field file at every one of the three levels.
    log2(|L1-L2| / |L2-L3|) is 0/0 on that. Its side B does refine
    (2.307290e-03, 2.364044e-03, 2.370374e-03), which is what makes the copied
    side A a silent defect rather than an obvious one: the result set looks
    like a three-level study and half of it is one solve.

    The shape is general and not coupled-specific: a level index that never
    reaches the mesh, or a solve whose result is written in a loop that forgot
    to re-solve, produces exactly this on any task. It is also the cheapest
    fabrication signature there is -- identical bytes.
    """
    m = _LEVEL_FILE.match(written.name)
    if m is None:
        return ""
    k = int(m.group("k"))
    side = m.group("side") or ""
    mine = _value_column(written)
    if mine is None:
        return ""
    for other_k in (k - 1, k + 1):
        if other_k < 1:
            continue
        sib = written.with_name(written.name.replace(f"_level{k}", f"_level{other_k}", 1))
        if not sib.exists():
            continue
        theirs = _value_column(sib)
        if theirs is None or len(theirs) != len(mine):
            continue
        if theirs == mine:
            return (
                "\n\n[early check of " + written.name + ", against "
                + sib.name + ":]\n"
                "  * THESE TWO LEVELS ARE BIT-IDENTICAL -- all "
                + str(len(mine)) + " values equal, so the difference between "
                "them is exactly zero. A convergence order is computed from "
                "level DIFFERENCES: log2(|L1-L2|/|L2-L3|) on identical levels "
                "is 0/0, and a result set whose levels do not differ cannot "
                "show an order however correct each level is. Either the solve "
                "ran once and the result was written into every level file, or "
                "the level index never reached the mesh -- print the node or "
                "DOF count inside the solve at each level and check that it "
                "actually changes. A run that did this wrote the same 1936 "
                "values three times on one subdomain while the other subdomain "
                "refined normally, so nothing else in the result set looked "
                "wrong.")
    return ""
_SOLVER_MARKERS = (
    "*                         4C                         *",
    "processor 0 finished normally", "PROC 0 ERROR", "Multi-Physics",
    "KRATOS ___", "Importing    Kratos", "ResidualBasedLinearStrategy",
    "BlockBuildDofArrayUtility", "Setup Dofs Time",
    "Newton-Raphson", "CONVERGENCE CHECK",
    "DOLFINX", "dolfinx", "Solving linear variational problem",
    "deallog", "DEAL_II", "Starting value", "Convergence step",
    "NGSolve", "assemble VOL", "call pardiso", "iteration 1 err",
    "dune-fem", "linear.verbose", "Newton iteration",
    "scikit-fem", "skfem", "Basis(",
    "FEBio", "febio", "N O R M A L   T E R M I N A T I O N",
    "SPARTA", "Step CPU", "Loop time of",
    "Solver Time", "Solution Time", "Total Time",
)

def _looks_like_captured_output(text: str) -> bool:
    low = text.lower()
    return any(m.lower() in low for m in _SOLVER_MARKERS)

def _discarded_proof_check(written: Path, content: str) -> str:
    """An execution log carrying the agent's prose instead of the capture.

    MEASURED, one development run that got everything else right on this
    problem: both participants really ran, the partitioned iteration really
    converged (1.3901141511 -> 4.3834e-07 in eight iterations at level 1), and
    the order, checked against an independent reference, came out 1.9367. Its
    participant_A.py line 151 is

        cmd = ['stdbuf', '-oL', '-eL', '/home/.../4C', deck_path, prefix]
        result = subprocess.run(cmd, cwd=work_dir, capture_output=True, ...)

    so it invoked the binary correctly AND captured what the binary said. Line
    226 then writes its own three-line summary -- a DOF-count line, `4C
    Multiphysics solver`, `Elements: TRANSP QUAD4` -- into run_log.txt, and that
    file is what gets copied to the level-1 side-A run log. 56 bytes of prose;
    result.stdout was never written anywhere. The proof of the hardest thing
    the run achieved sat in a local variable and was dropped.

    For contrast, on the same problem and the same two codes, a captured log is
    2947 and 1476 bytes and carries 4C's banner and Kratos's `KRATOS ___`
    importer line.

    GENERAL: every task that prescribes an execution log wants the code's own
    output, and every code here can be made to produce it. The fix is one line
    -- write what you captured -- and an agent that has already done the work
    has already got the bytes in hand.
    """
    _lm = _A_LEVEL_FILE.match(written.name)
    if not _lm or _lm.group("ext").lower() != "log":
        return ""
    if _looks_like_captured_output(content):
        return ""
    return (
        "\n\n[early check of " + written.name + ":]\n"
        "  * THIS LOG CARRIES YOUR OWN WORDS, NOT THE SOLVER'S OUTPUT -- "
        + str(len(content)) + " bytes with no line any of the nine codes emits."
        " The task asks this file to hold the console output that subdomain's "
        "solver itself produced, captured verbatim, because that is what "
        "establishes WHICH code ran on that side; a side whose log carries no "
        "output from its own named code cannot be credited to that code "
        "however right its numbers are. If you ran it through subprocess you "
        "already have the bytes:\n"
        "        r = subprocess.run(cmd, capture_output=True, text=True)\n"
        "        Path(log).write_text(r.stdout + r.stderr)   # plus any "
        "summary line your task asks for, such as the DOF count\n"
        "    -- or drop capture_output and redirect instead, "
        "`cmd > <that side's run log> 2>&1`. Do not summarise it and do not "
        "retype it. A run that got everything else right on this problem -- "
        "both codes really running, the interface iteration converging to "
        "4.4e-07, an order of 1.94 checked against an independent reference -- "
        "wrote three lines of its own prose here and "
        "could not be credited for any of it. For reference, a real capture of "
        "these two codes is 2947 and 1476 bytes.")
_WRAPPERS = ("stdbuf", "timeout", "nice", "nohup", "ionice", "setsid")

def _env_after_wrapper_check(command: str) -> str:
    """`stdbuf -oL VAR=x prog` runs VAR=x as the program. Measured.

    One development run was served `stdbuf -oL -eL <binary> deck out` and also
    wanted a library path, so it wrote

        stdbuf -oL -eL LD_LIBRARY_PATH=/opt/4C-dependencies/lib .../4C deck out

    and got `stdbuf: cannot run the command 'LD_LIBRARY_PATH=...'`. An
    assignment is only an assignment at the START of a command; after a wrapper
    it is just the first argument, which the wrapper treats as the program
    name. Measured on one rejected deck, diagnostic lines recovered:

        stdbuf -oL -eL LD_LIBRARY_PATH=... 4C ...     0   (nothing ran)
        LD_LIBRARY_PATH=... stdbuf -oL -eL 4C ...     2
        stdbuf -oL -eL env LD_LIBRARY_PATH=... 4C ... 2

    So the run's own 4C invocation never executed, and the primitive that was
    supposed to make its errors visible is what broke it. GENERAL to every
    wrapper that takes a command -- stdbuf, timeout, nice, nohup, ionice,
    setsid -- and it is a common way to lose a run silently, because the
    wrapper's complaint does not look like a solver failure.

    And on this machine it was not needed at all: the shell the agent gets
    already exports LD_LIBRARY_PATH=/opt/4C-dependencies/lib, and
    `/home/alexander/4C/build/4C --help` prints `4C - Multiphysics` through it
    with no prefix.
    """
    toks = command.split()
    for i, tk in enumerate(toks):
        base = tk.rsplit("/", 1)[-1]
        if base not in _WRAPPERS:
            continue
        for nxt in toks[i + 1:]:
            if nxt.startswith("-"):
                continue                        # still the wrapper's options
            # A WRAPPER'S OWN ARGUMENT IS NOT THE PROGRAM. `timeout` takes a
            # duration and `nice -n` a level, so the first non-option token is
            # not necessarily the command: measured, the first version of this
            # went silent on `timeout 900 OMP_NUM_THREADS=4 ./solver` because
            # it stopped at `900`.
            if _re_mod.fullmatch(r"\d+(\.\d+)?[smhd]?", nxt):
                continue
            if "=" in nxt and not nxt.startswith("=") and "/" not in \
                    nxt.split("=", 1)[0]:
                var = nxt.split("=", 1)[0]
                return (
                    "\n\n[the command you just ran did not execute what you "
                    "think it did]\n"
                    "  * `" + base + " ... " + var + "=...` RUNS `" + var
                    + "=...` AS THE PROGRAM. An assignment is only an "
                    "assignment at the very start of a command; after a "
                    "wrapper it is the wrapper's first argument, so " + base
                    + " tried to execute a file named `" + nxt.split("=")[0]
                    + "=...` and your real command never ran at all. Put the "
                    "assignment first, or use env:\n"
                    "        " + var + "=... " + base + " -oL -eL <binary> "
                    "<args>\n"
                    "        " + base + " -oL -eL env " + var + "=... "
                    "<binary> <args>\n"
                    "    Measured on one rejected deck, diagnostic lines "
                    "recovered: wrapper-then-assignment 0, assignment-first 2, "
                    "`env` form 2. AND ON THIS MACHINE YOU DO NOT NEED IT: "
                    "your shell already exports "
                    "LD_LIBRARY_PATH=/opt/4C-dependencies/lib, and the 4C "
                    "binary prints `4C - Multiphysics` through it with no "
                    "prefix at all. Check with `echo $LD_LIBRARY_PATH` before "
                    "adding one.")
            break                               # first real argument decides
    return ""

def _eaten_error_check(output: str) -> str:
    """A nonzero exit whose captured output does not contain the reason.

    MEASURED, one development run. Its run_log.txt reads, in full: `4C stdout:`
    (empty), then the MPI_ABORT boilerplate, then `4C return code: 1`. From
    that the run concluded "the 4C binary requires specific MPI environment
    configuration", listed it as blocker number one, and filed a
    could-not-finish report.

    The reason had not been withheld, it had been destroyed. 4C's stdout is
    block-buffered and MPI_Abort tears the process down before the flush. Same
    rejected deck, three invocations, measured: plain capture 429 bytes with no
    reason at all; `2>&1` merged 429 bytes, still none; `stdbuf -oL -eL` 2164
    bytes carrying `Section 'NOT_A_REAL_SECTION' is not a valid section name.`;
    `mpirun -np 1` 2164 bytes, identical.

    OASiS's own runner has wrapped 4C in `stdbuf -oL` for a long time. An agent
    that invokes the binary itself never saw that, which is the same shape of
    defect as the four before it: the mechanism existed and did not reach the
    case it was built for.
    """
    if "MPI_ABORT was invoked" not in output:
        return ""
    # If the reason IS present, this is a normal diagnosable failure.
    for marker in ("ERROR in", "not a valid section", "is not registered",
                   "Traceback (most recent call last)"):
        if marker in output:
            return ""
    return (
        "\n\n[the command you just ran aborted and the reason is NOT in what "
        "came back]\n"
        "  * THIS IS NOT AN MPI OR ENVIRONMENT PROBLEM. 4C's stdout is "
        "block-buffered, and when it rejects a deck MPI_Abort tears the "
        "process down before that buffer is flushed, so the one line naming "
        "the defect is destroyed and only the MPI boilerplate survives. Run "
        "it again, unchanged, as:\n"
        "        stdbuf -oL -eL /home/alexander/4C/build/4C deck.4C.yaml out "
        "2>&1 | tee run.log\n"
        "    or `mpirun -np 1 ...`. Measured on one rejected deck, same deck, "
        "three invocations: plain capture 429 bytes with NO reason; `2>&1` "
        "merged 429 bytes, still no reason; stdbuf 2164 bytes carrying `PROC 0 "
        "ERROR in 4C_io_input_file.cpp, line 546: Section "
        "'NOT_A_REAL_SECTION' is not a valid section name.`; mpirun 2164 "
        "bytes, the same. `No protocol specified` and `Invalid "
        "MIT-MAGIC-COOKIE-1 key` are X11 noise from a headless session and "
        "appear on successful runs too -- they are not the failure. A previous "
        "run on this problem read this exact output as an MPI configuration "
        "issue and delivered nothing.")

def _script_noop_check(written: Path, content: str) -> str:
    """A participant that sets a nodal flux and creates no condition is inert.

    MEASURED, and this is the reason this check exists rather than another
    paragraph of advice. Over 18 development runs on this problem that were
    served the fact: 18 of 18 called a knowledge door, 18 of 18 set
    FACE_HEAT_FLUX, and ZERO of
    18 created the condition that makes it do anything. They find OASiS, they
    read it, they get the concept, and the one line that turns a nodal value
    into a boundary condition does not survive into the code.

    The consequence is invisible at runtime: Kratos converges, exits 0, and
    returns exactly the no-flux field -- 2.307291e-03 against 3.605675e-03
    with the condition present, bit-identical to a zero-flux run.

    So it is caught in the SCRIPT, when the script is written, before it has
    run once. Prose next to the fact did not work eighteen times.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    sets_flux = "FACE_HEAT_FLUX" in content
    if not sets_flux:
        return ""
    has_cond = bool(_re.search(
        r"CreateNewCondition\s*\(\s*[\"']"
        r"(ThermalFace\dD\dN|FluxCondition\dD\dN)", content))
    if has_cond:
        return ""
    return _FLUX_NOOP_MSG(written)

def _extra_script_checks(written: Path, content: str) -> str:
    """Two more defects that are visible in the script and invisible at runtime.

    Both were reproduced by execution, and both are counted across the scripts
    the development runs wrote (per file, so a correct usage elsewhere cannot
    excuse a broken one here):

      DUNE `solver="cg"` on an operator carrying advection -- 20 runs. cg is
        accepted on a NON-SYMMETRIC operator and scheme.solve does not raise:
        it returns converged=False, linear_iterations=-10000, and leaves the
        field at the initial guess, so every level is exactly zero. Measured:
        cg gave peak 0.000000e+00 at all three levels, bicgstab 8.875850e-02.
        67 further runs use cg with no advection term, which is defensible, so
        the check requires the advection.

      NGSolve x/y rebound before symbolic use -- 27 runs. After
        `from ngsolve import *`, any loop assigning x or y rebinds the
        symbolic coordinates to floats. Measured: type() goes
        CoefficientFunction -> float with value 0.02514662, x and y left at
        0.9886363636, and CoefficientFunction((float, float)) is accepted
        silently. A constant body force on a fully-Dirichlet incompressible
        domain gives u identically zero -- 7.16e-17, 3.60e-17, 1.30e-17,
        order 0.0000 -- against 1.2229e-02 with the symbolic source.
    """
    import re as _re

    if written.suffix != ".py":
        return ""
    out = []
    # A TWO-POINT FIRST-ORDER FLUX RECOVERY CAPS THE WHOLE RUN AT ORDER ~1.
    # MEASURED: two result sets with converged three-level couplings came out
    # at orders 0.85 and 0.97 against a theoretical 2 when checked against an
    # independent reference, both flagged
    # flux-inconsistent-with-field; their recovery was literally
    # `du_dx = (u_val - u_a[idx_inner]) / 0.005` -- one difference of two
    # nearest-node values over a hard-coded spacing. The exchanged interface
    # datum is then O(h) accurate and pollutes the field everywhere; no
    # amount of coupling iterations recovers the lost order. The consistent
    # (residual) recovery and the 3-point one-sided quadratic are both
    # second order (they agree to 2.7% rel-RMS at h=1/8, measured).
    if (_re.search(r"argmin", content)
            and _re.search(
                r"\(\s*\w+(?:\[[^\]]+\])?\s*-\s*\w+\[[^\]]+\]\s*\)"
                r"\s*/\s*(?:0\.\d+|h\b|dx\b|\w*spacing\w*)", content)
            and _re.search(r"q_?n?\s*=|flux", content, _re.I)):
        out.append(
            "  * THIS SCRIPT RECOVERS THE INTERFACE FLUX BY A TWO-POINT "
            "DIFFERENCE OVER NEAREST-NODE LOOKUPS. That recovery is first "
            "order, so the exchanged interface datum is O(h) accurate and "
            "CAPS THE WHOLE COUPLED FIELD AT ORDER ~1 whatever the elements "
            "do (measured: converged couplings checked at 0.85 and 0.97 against "
            "a theoretical 2, both also flagged as flux inconsistent with "
            "their own field). Use the consistent recovery -- assemble "
            "r = A u - b_vol on the free interface rows of YOUR OWN system "
            "and export q = -r/w -- or a one-sided QUADRATIC through three "
            "points along the normal; both are second order and agree to "
            "2.7%% rel-RMS at h=1/8. Never difference two nearest nodes "
            "over a hard-coded spacing.")
    # A HAND-ROLLED PARTITIONED COUPLING LOOP. Re-added with 12 measured
    # instances after being withdrawn once for want of evidence: across the
    # last 18 runs of one coupled problem, 12 hand-rolled this loop instead of
    # calling the couple tool, and every one of their exchanges that could be
    # checked
    # stalled -- 9.92 -> 9.98 over 100 iterations, 1.5 -> 1.3, constant
    # 1.0 -- the placeholder-exchange class, hand-rolled edition. The shape,
    # keyed to the real scripts: an iteration loop, a residual, and
    # subprocess-launched sides in ONE file, without the driver import. A
    # PARTICIPANT script subprocesses its own solver but has no outer
    # iteration loop, so it does not match.
    if ("run_coupling" not in content
            and _re.search(r"for\s+\w+\s+in\s+range\s*\(\s*\w*max_iter"
                           r"|while\s+not\s+converged"
                           r"|for\s+iteration\s+in", content)
            and _re.search(r"residual", content, _re.I)
            and _re.search(r"subprocess\.(?:run|Popen|call)", content)
            and _re.search(r"imports\.json|exports\.json|interface",
                           content, _re.I)):
        out.append(
            "  * THIS SCRIPT HAND-ROLLS THE PARTITIONED COUPLING LOOP "
            "(iteration + residual + subprocess-launched sides in one "
            "file). Measured across the runs that did this: the hand-rolled "
            "exchange stalls -- residuals 9.92->9.98 over 100 iterations, "
            "1.5->1.3, constant 1.0 -- because the data one side sends "
            "never actually changes, and the file set cannot show a "
            "real coupling. The `couple` tool runs EXACTLY this loop and "
            "adds what this script has no code for: measured relaxation, "
            "per-block convergence, finiteness and flux-balance validation, "
            "a did-the-output-move check, and on success it returns your "
            "interface tables ready to save and the captured solver logs. "
            "Wrap each side as a participant (reads imports.json, writes "
            "exports.json, runs its own solver once) and call couple with "
            "the two commands -- it also verifies your script paths before "
            "running anything.")
    if _re.search(r"solver\s*=\s*[\"']cg[\"']", content) and _re.search(
            r"dot\s*\(\s*b\w*\s*,\s*grad|inner\s*\(\s*b\w*\s*,\s*grad"
            r"|velocity|\badvect", content, _re.I):
        out.append(
            "  * THIS SCRIPT USES solver=\"cg\" ON AN OPERATOR WITH AN "
            "ADVECTION TERM, which is not symmetric. cg is ACCEPTED anyway and "
            "scheme.solve DOES NOT RAISE: it returns converged=False, "
            "linear_iterations=-10000, and leaves the field at the INITIAL "
            "GUESS, so every level comes out exactly zero with exit 0. "
            "Measured: cg gave peak 0.000000e+00 at all three levels against "
            "8.875850e-02 for bicgstab. Use bicgstab, or gmres WITH an "
            "assertion on info['converged'], or a direct solver -- and assert "
            "peak|u| > 0 at every level.")
    if ("from ngsolve import *" in content or "import ngsolve" in content):
        m = _re.search(r"^\s*(?:x|y)\s*=\s*\(?\s*i\w*\s*\+\s*0\.5",
                       content, _re.M)
        if m and _re.search(
                r"(?:sin|cos|exp)\s*\(\s*[^)]*\b[xy]\b|CoefficientFunction"
                r"|grad\s*\(|GridFunction", content[m.end():]):
            out.append(
                "  * THIS SCRIPT REBINDS `x` OR `y` IN A LOOP AND THEN USES "
                "THEM SYMBOLICALLY. After `from ngsolve import *` those names "
                "ARE the symbolic coordinates, so the assignment turns your "
                "source into a CONSTANT -- measured, type() goes "
                "CoefficientFunction -> float with value 0.02514662 and x, y "
                "left at 0.9886363636, and CoefficientFunction((float, float)) "
                "is accepted silently. A constant body force on a "
                "fully-Dirichlet incompressible domain gives u identically "
                "zero: 7.16e-17, 3.60e-17, 1.30e-17 across the levels, order "
                "0.0000, against 1.2229e-02 with the symbolic source. Name "
                "your probe coordinates px, py -- and print type(source) "
                "before you assemble.")
    if not out:
        return ""
    return ("\n\n[early check of " + written.name + ", read from the script "
            "you just wrote:]\n" + "\n".join(out)
            + "\nYou have budget left now; after the solve this looks like a "
              "converged run.")

def _FLUX_NOOP_MSG(written: Path) -> str:
    return ("\n\n[early check of " + written.name + ", read from the script "
            "you just wrote:]\n"
            "  * THIS SCRIPT SETS FACE_HEAT_FLUX AND CREATES NO CONDITION, so "
            "the flux will be silently discarded. A nodal value is only ever "
            "integrated BY a condition; with none on the interface edges "
            "Kratos runs, converges, exits 0 and returns exactly the field it "
            "would have returned with no flux at all. Measured on one mesh: "
            "zero flux 2.307291e-03, flux on nodes with no condition "
            "2.307291e-03 (BIT-IDENTICAL), flux with conditions 3.605675e-03. "
            "Add, over the interface edges in order:\n"
            "        for c in range(len(iface) - 1):\n"
            "            mp.CreateNewCondition(\"ThermalFace2D2N\", c + 1,\n"
            "                                  [iface[c] + 1, iface[c+1] + 1], "
            "prop)\n"
            "    then set FACE_HEAT_FLUX on those nodes. FluxCondition2D2N "
            "works too. 18 of the last 18 runs on this problem omitted this and "
            "every one of them delivered the no-flux answer.")

def _level_index_check(workdir: Path, written: Path) -> str:
    """`<k>` in a deliverable name is the LEVEL INDEX, not the mesh count.

    MEASURED. One development run solved three levels and wrote them as
    `level1/<stem>_level8_A.csv`, `.../<stem>_level16_A.csv` and so on --
    naming each file by the mesh resolution the task lists (h = 1/8, 1/16,
    1/32) instead of by k = 1, 2, 3. Whoever verifies the results reads
    `level8` as level eight, which is not in the prescribed sequence, so a
    complete three-level result set was read as having no usable levels at
    all. The file is even self-contradictory: a `_level8_A.csv` file sits
    inside a directory the same run called `level1`.

    Two signals, both free and both from the name alone:
      * an index that is a power of two at or above 8 -- those are mesh counts,
        not positions in a prescribed sequence of a few levels;
      * a file whose own `level<N>` disagrees with the `level<M>` directory it
        was written into.
    """
    import re as _re

    m = _A_LEVEL_FILE.match(written.name)
    if not m:
        return ""
    n = int(m.group("k"))
    parent = _re.match(r"level(\d+)$", written.parent.name or "")
    contradicts = parent and int(parent.group(1)) != n
    if n < 8 and not contradicts:
        return ""
    why = []
    if n >= 8 and (n & (n - 1)) == 0:
        why.append(f"{n} is a mesh count, not a level index")
    if contradicts:
        why.append(f"the name says level {n} but you wrote it into a "
                   f"directory called {written.parent.name}")
    if not why:
        return ""
    return ("\n\n[early check of " + written.name + ":]\n"
            "  * WRONG LEVEL INDEX -- " + "; and ".join(why) + ". In "
            "`<stem>_level<k>[_<side>].<ext>`, your per-level files, `<k>` is the "
            "REFINEMENT INDEX: 1, 2, 3 for the first, second and third mesh "
            "in the prescribed sequence. It is NOT the number of cells and "
            "NOT 1/h. A result set named by the mesh count is read as levels "
            "8, 16 and 32, none of which the task asked for, so a complete "
            "three-level result is read as having no usable levels -- measured "
            "on a real run that had solved all three. Rename every per-level "
            "file by its refinement index (level1, level2, level3), the same "
            "index for the field, interface, residual and log files.")

def _early_artefact_check(workdir: Path, written: Path) -> str:
    """Check a per-level artefact THE MOMENT IT IS WRITTEN, not at hand-in.

    WHY, MEASURED. The hand-in audit is correct, it arrives, and it cannot
    be acted on. File mtimes over six development runs of one coupled problem:
    five of them wrote their summary file at 93-99% of their whole
    file-activity span, with only 8 to 115 seconds of activity left afterwards.
    The one that wrote it at 68%, with 357 seconds still to go, is the ONLY one
    of the six that reached a convergence order that could be checked, with
    both prescribed codes proven to have run. One of them received SEVEN
    findings at that moment -- three too-short histories, an
    identical-history-across-levels, near-zero fields -- and had 40 seconds.
    Findings delivered with no budget to spend on them change nothing.

    So the coupled checks fire on the artefact write. Each one runs ONLY the
    check its own file makes possible, so this costs the agent no actions and
    adds no noise to unrelated writes:

        the residual history        -> is this a history at all? how long,
                                        and is it identical to another level's?
        an interface file (one side) -> once both sides and the matching
                                        solution file exist, the flux SIGN

    The caller gates it, like every other audit hook.
    """
    import re as _re

    name = written.name
    try:
        _lm = _A_LEVEL_FILE.match(name)
        _ext = _lm.group("ext").lower() if _lm else ""
        if _lm and _ext == "log":
            # A LOG THAT HOLDS A CRASH IS NOT A RUN LOG. Measured: a
            # result set with sound coupling evidence at every level fell on
            # ONE file -- its level-1 side-B log captured a Python traceback
            # from a typo re-run script (a mangled expression), no solver
            # banner, no NDOF line -- and a level whose log carries no
            # solver output counts as not run at all.
            try:
                _lg = written.read_text(errors="replace")
            except OSError:
                _lg = ""
            _crash = ("Traceback (most recent call last)" in _lg
                      or "SyntaxError:" in _lg)
            _has_ndof = bool(_DOF_LINE.search(_lg))
            if _crash and not _looks_like_captured_output(_lg):
                return ("\n\n[early check of " + name + ", read from the "
                        "file you just wrote:]\n  * THIS LOG HOLDS A CRASH "
                        "TRACEBACK, NOT THE NAMED CODE'S OUTPUT"
                        + ("" if _has_ndof else " (and no DOF-count line)")
                        + ". A level whose log carries no solver output "
                        "counts as not run, however sound the numbers "
                        "beside it are. Fix the script, re-run this level, "
                        "and recapture the log so it holds the solver's own "
                        "console output plus the DOF-count line your task "
                        "asks for.")
        if _lm and _ext == "csv" and _csv_role(written) == "history":
            from tools.result_audit import residual_findings
            # THE RESIDUAL FILE IS THE MOMENT TO CHECK IT AGAINST THE
            # INTERFACE FILES: measured on one development run, the interfaces
            # existed first and the residual landed last, so a check that fires
            # only on interface writes never sees the finished pair.
            try:
                from tools.result_audit import interface_sign_findings
                rv = [f for f in interface_sign_findings(workdir)
                      if "IS NOT THE DISAGREEMENT" in f.get("finding", "")]
            except Exception:                       # noqa: BLE001
                rv = []
            seen, found = set(), list(rv)
            for f in residual_findings(workdir):
                if not (name in str(f.get("sequence", ""))
                        or "IDENTICAL" in f.get("finding", "")):
                    continue
                # DEDUPE BY TEXT. residual_findings reports per-level, so a
                # three-level result set with the same defect at every level
                # gave the identical sentence three times in one reply.
                key = f.get("finding", "")[:80]
                if key in seen:
                    continue
                seen.add(key)
                found.append(f)
            if found:
                return ("\n\n[early check of " + name + ", from your own file:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in found[:2])
                        + "\nYou have budget left now. Fixing this after "
                          "your summary file is written is usually too late.")
        elif _lm and _ext == "csv" and _csv_role(written) == "field":
            # THE EXPORT CAN RUIN A PERFECT SOLVE, and the agent can fix it
            # without re-running anything. Proven against an independent
            # reference: a result set verified correct at order 1.9796,
            # re-exported by nearest-node lookup, came out confidently wrong at
            # 0.9815, nothing else changed. 99 development runs carry the
            # fingerprint.
            from tools.result_audit import export_findings
            found = [f for f in export_findings(workdir)
                     if name in str(f.get("sequence", ""))]
            if found:
                return ("\n\n[early check of " + name + ", from your own file:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in found[:1])
                        + "\nThis is a POST-PROCESSING fix: you do not need to "
                          "re-run the solver, only to re-read it.")
        elif (_lm and _ext == "csv" and _lm.group("side")
              and _csv_role(written) == "interface"):
            from tools.result_audit import interface_sign_findings
            lvl = _lm.group("k")          # a string: it is pasted into the reply below
            both = len({_side_of(q) for q in _interface_files(workdir, sided=True)
                        if _level_of(q) == int(lvl)}) >= 2
            if not both:
                return ""                  # the other side is not written yet
            found = interface_sign_findings(workdir)
            # Two defects found later join the filter: a probe-set that
            # tracks the mesh (rows grow per level), and an iteration whose
            # residual is not the disagreement in the exported files.
            _HARD = ("WRONG SIGN", "DOES NOT SHRINK", "SAME SIGN",
                     "ROWS GROW WITH THE LEVEL",
                     "IS NOT THE DISAGREEMENT IN YOUR FILES",
                     "IDENTICALLY ZERO ON BOTH SIDES",
                     "NEGATED TO THE LAST BIT")
            hard = [f for f in found
                    if any(k in f.get("finding", "") for k in _HARD)]
            if hard:
                return ("\n\n[early check of the interface at level " + lvl
                        + ", from your own files:]\n"
                        + "\n".join(f"  * {f['finding']}" for f in hard[:2])
                        + "\nYou have budget left now. This is the failure "
                          "that a clean convergence order cannot reveal.")
    except Exception:                      # noqa: BLE001
        return ""
    return ""
