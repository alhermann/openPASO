#!/usr/bin/env python
"""Would a CORRECT submission be graded CORRECT? Ask the grader, do not assume.

This exists because assuming it has been wrong three times. A probe-count
mismatch, a missing subdomain exclusion, and an interface band that put half the
points outside the graded region each rejected a correct submission with
INVALID_SUBMISSION -- before any comparison against truth -- and each was found
only when somebody built the submission the task text describes and ran the
grader on it. Eight of the fifteen instances in the previous set were in that
state at once.

Three checks per instance, in order, each of which has caught a real defect:

  1. TASK TEXT vs GRADER. The probe count the task text PRESCRIBES against the
     count ``grade_blind.probe_grid()`` BUILDS. These are two independent pieces
     of code reading two different sources, and they have disagreed.
  2. SPEC vs GRADER. The point SET the public spec implies against the set the
     grader builds, coordinate by coordinate.
  3. END TO END. Assemble the submission a correct run would produce -- the
     exact field at the prescribed probe points plus a synthetic O(h^2) error, a
     converging partitioned-residual history, the NDOF execution logs both codes
     must write, and RESULT.txt -- and require the verdict to be CORRECT.

The synthetic error is a per-level constant offset ``c * 4^-k``, so the RMS is
exactly ``c * 4^-k``, the observed order is exactly 2 and R^2 is exactly 1. That
isolates the CONTRACT: anything other than CORRECT is then a defect in the task
text, the spec or the grader, never in the physics.

  campaign3_blind/check_grader_accepts.py C1 C2 ...
"""
from __future__ import annotations

import argparse
import pathlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

import sympy as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

# THE GRADER THIS VALIDATES MUST BE THE GRADER THAT RUNS. This imported
# grade_blind (v1) while every round since the coupled redesign has been graded
# by grade_blind_v2 through grading.loading — so the one instrument whose job is
# to answer "would a correct submission be graded CORRECT?" was answering it
# about code no longer in use, and had not run at all since the vault was
# encrypted (v1 reads plaintext key.json only). Both faults pointed the same
# way: 0/N "would accept a correct submission", indistinguishable from a grader
# that rejects everything.
import grade_blind_v2 as G                                       # noqa: E402
from grading import loading as _loading                          # noqa: E402
import build_coupled_v2 as _V                                    # noqa: E402


# LINES MEASURED FROM REAL RUNS OF EACH CODE, one per backend. The contract
# asks the agent to capture the solver's own console output, so a harness that
# checks "would a correct submission be accepted?" has to produce it. Invented
# English would not do: the per-code signature table deliberately rejects the
# phrasings an agent writes about its own hand-rolled solver, which is exactly
# what `code = <name>` was.
_OWN = {
    "fenics":  "[2026-08-31 12:12:57.402] [info] Cell type: 0 dofmap: {nd}x3\n",
    "fenicsx": "[2026-08-31 12:12:57.402] [info] Cell type: 0 dofmap: {nd}x3\n",
    "dolfinx": "[2026-08-31 12:12:57.402] [info] Cell type: 0 dofmap: {nd}x3\n",
    "ngsolve": "assemble VOL element {nd}/{nd}\n",
    "dealii":  "DEAL:cg::Starting value 3.027e-02\n"
               "DEAL:cg::Convergence step 47 value 4.129e-13\n",
    "skfem":   "INFO:skfem.utils:Solving linear system, shape=({nd}, {nd}).\n",
    "dune":    "Fem::CG preconditioning=none\nFem::CG it: 0 : residual 1.0e-01\n",
    "kratos":  "ModelPartIO:   [Reading Nodes    : {nd} nodes read]\n",
    "4C":      "Finalised step 1 / 1 | time 1.000e+00 | dt 1.000e+00 "
               "| nlniter 4 | wct 4.21e-02\n",
    "fourc":   "Finalised step 1 / 1 | time 1.000e+00 | dt 1.000e+00 "
               "| nlniter 4 | wct 4.21e-02\n",
    "febio":   "\tNr of equations ........................... : {nd}\n"
               " N O R M A L   T E R M I N A T I O N\n",
    "sparta":  "Created {nd} child grid cells\n"
               "Loop time of 0.018019 on 1 procs for 100 steps with "
               "17328 particles\n",
}


def _own_output(code: str, nd: int) -> str:
    """The named code's own output line. Unknown code -> nothing, and the
    evidence gate will then say so rather than this harness pretending."""
    return _OWN.get(str(code).lower(), "").format(nd=nd)

# v1 exposed these as module constants; v2 resolves them through loading, which
# honours OPENPASO_BLIND_KEYS and is encryption-aware.
G.KEYS = _loading.keys_dir()
G.PROBLEMS = _loading.problems_dir()

TMP = Path(os.environ.get("OPENPASO_CHECK_TMP",
                          "/tmp/claude-1001/-home-alexander-4C/balancedtmp/grade"))
SYMS = {"x": sp.Symbol("x", real=True), "y": sp.Symbol("y", real=True),
        "z": sp.Symbol("z", real=True)}


def task_probe_counts(text: str) -> dict:
    """What the TASK TEXT says the probe sets are, read out of the text."""
    out = {}
    for side in ("A", "B"):
        m = re.search(rf"PROBE POINTS, subdomain {side}: (.*?)(?=\nPROBE POINTS|"
                      rf"\nEvaluate)", text, re.S)
        if not m:
            continue
        blob = m.group(1)
        rem = re.search(r"(\d+) points remain", blob)
        first = re.search(r"the (\d+) points", blob)
        out[side] = int(rem.group(1)) if rem else (int(first.group(1))
                                                   if first else None)
    m = re.search(r"INTERFACE PROBE POINTS: the (\d+) points", text)
    if m:
        out["iface"] = int(m.group(1))
    return out


def build_submission(pid: str, key: dict, spec: dict, work: Path,
                     rel: float = 0.5) -> dict:
    # band-only keys carry no dim/coords/exact_solution — their branch below
    # must run before anything reads those fields.
    dim = key.get("dim", 2)
    coords = key.get("coords", ["x", "y"])
    comps = key.get("components", ["u"])
    ncomp = len(comps)
    tol = float(spec.get("interface_tol", "1e-6").split()[0])
    info = {}
    # THE SYNTHETIC ERROR IS SCALED PER COMPONENT. A single pooled offset is
    # what a correct run does NOT look like when the fields have different
    # scales: on the thermoelastic cell (T ~ 2.7e-2 RMS, u ~ 1e-5) the pooled
    # offset is ~1000x the displacement's own magnitude, and grader v2's
    # per-field magnitude bounds rightly graded that CONFIDENTLY_WRONG /
    # IMPLAUSIBLE_MAGNITUDE(ux, uy). Each component gets rel x its OWN RMS
    # over the probe grid, decaying 4x per level — which is what a real
    # second-order error does.
    # BAND-ONLY cells (SPARTA) have no exact solution and no probe grid: the
    # key pre-registers a QoI band and a conservation identity, the evidence
    # gate wants lines the DSMC binary itself prints, and the verdict for a
    # correct submission is WITHIN_BAND (grade-3 vocabulary), not CORRECT.
    # GRADE-2 (reference) and coupled band-only cells: the last two shapes.
    # Both are coupled, so the evidence gate additionally wants per-code logs
    # for BOTH codes (the canonical NDOF line is accepted for every code) and
    # a converging partitioned-residual history per level.
    if key.get("grading") == "reference" or (
            key.get("grading") == "band-only" and key.get("kind") == "coupled"):
        levels = 3
        tol_i = float(spec.get("interface_tol", "1e-6").split()[0])             if isinstance(spec.get("interface_tol"), str) else             float(spec.get("interface_tol", 1e-6))
        for lvl in range(1, levels + 1):
            for s, code in zip(("A", "B"), key["codes"]):
                (work / f"run_level{lvl}_{s}.log").write_text(
                    f"code = {code}\nside = {s}\n"
                    f"NDOF = {2400 * 4 ** (lvl - 1)}\n")
            with open(work / f"residual_level{lvl}.csv", "w") as fh:
                fh.write("iteration,interface_residual\n")
                r, i = 1.0, 1
                while r > tol_i * 0.5:
                    fh.write(f"{i},{r:.6e}\n")
                    r *= 0.25
                    i += 1
                fh.write(f"{i},{r:.6e}\n")
        lines = ["LEVELS = %d" % levels,
                 "FILES = " + ", ".join(f"residual_level{l}.csv"
                                        for l in range(1, levels + 1)),
                 "MESH_INDEPENDENCE = CONVERGED",
                 "MAX_REL_CHANGE = 1.0e-03",
                 f"COUPLING_ITERATIONS = {i}"]
        if key.get("grading") == "reference":
            ref = key["reference"]
            lines.append(f"{ref['result_line']} = {float(ref['value']):.12g}")
        else:
            qoi = key.get("qoi_band") or key["qoi"]
            lo, hi = qoi["band"]; mid = 0.5 * (lo + hi)
            ident = key.get("identity") or {}
            if ident:
                lines += [f"{ident['lhs_line']} = {mid:.9g}",
                          f"{ident['rhs_line']} = "
                          f"{mid * (1 + ident.get('rtol', 0.05) * 0.2):.9g}"]
            lines.append(f"{qoi['result_line']} = {mid:.9g}")
        (work / "RESULT.txt").write_text("\n".join(lines) + "\n")
        return {"kind": key.get("grading")}

    if key.get("grading") == "band-only" or key.get("kind") == "sparta_band":
        qoi = key.get("qoi_band") or key["qoi"]
        lo, hi = qoi["band"]
        mid = 0.5 * (lo + hi)
        ident = key.get("identity") or {}
        levels = 3
        for lvl in range(1, levels + 1):
            (work / f"qoi_level{lvl}.csv").write_text(
                "level,qoi\n%d,%.9g\n" % (lvl, mid * (1 + 1e-3 * (levels - lvl))))
            # A real accepted SPARTA log leads with the canonical NDOF line
            # (grader v2 additionally requires the per-level sequence to GROW
            # like 2**dim under halving); the DSMC-signature lines below are
            # what the per-code evidence gate matches.
            (work / f"run_level{lvl}.log").write_text(
                f"NDOF = {3200 * 4 ** (lvl - 1)}\n"
                "Created 100000 particles\n"
                "grid cells = 40000\n"
                "Step 5000  CPU = 12.1 (98 secs)\n"
                "Loop time of 98.2 on 4 procs\n")
        lines = ["LEVELS = %d" % levels,
                 "FILES = " + ", ".join(f"qoi_level{l}.csv"
                                        for l in range(1, levels + 1)),
                 "MESH_INDEPENDENCE = CONVERGED",
                 "MAX_REL_CHANGE = 1.0e-03"]
        if ident:
            lines += [f"{ident['lhs_line']} = {mid:.9g}",
                      f"{ident['rhs_line']} = {mid * (1 + ident.get('rtol', 0.05) * 0.2):.9g}"]
        lines.append(f"{qoi['result_line']} = {mid:.9g}")
        (work / "RESULT.txt").write_text("\n".join(lines) + "\n")
        return {"band": (lo, hi), "mid": mid}

    # SINGLE cells have one domain: key["exact_solution"] IS the expression
    # (string, or list of components); only coupled keys nest it per side.
    # Indexing a string with ["A"] is what raised "string indices must be
    # integers" on every single-code cell.
    coupled = pid.startswith("C")
    sides = ("A", "B") if coupled else ("A",)
    def _src(side):
        return key["exact_solution"][side] if coupled else key["exact_solution"]
    comp_rms = {}
    for side in sides:
        bounds = _bounds(key, side, dim, pid)
        excl = G.probe_exclusions(pid, side)
        pts = G.probe_grid(dim, bounds, excl)
        src = _src(side)
        ex = (src if isinstance(src, list) else [src])
        for c, e in enumerate(ex):
            f = sp.lambdify([SYMS[cc] for cc in coords],
                            sp.sympify(e, locals=SYMS), "math")
            acc = sum(float(f(*p)) ** 2 for p in pts)
            comp_rms.setdefault(c, []).append((acc, len(pts)))
    comp_rms = {c: (sum(a for a, _ in v) / sum(n for _, n in v)) ** 0.5
                for c, v in comp_rms.items()}
    # The synthetic error must decay at the ORDER THE CELL DEMANDS. The
    # hardcoded 4x per level (order 2) predates the higher-order cells: on the
    # four order-3 cells a "correct" submission built this way converges at 2
    # and the grader RIGHTLY graded it CONFIDENTLY_WRONG — the checker was
    # testing the wrong submission, not the grader.
    p_ord = float(key.get("theoretical_order", 2.0))
    decay = 2.0 ** p_ord
    for lvl in range(1, len(key["mesh_N"]) + 1):
        offs = {c: rel * max(r, 1e-300) * decay ** -(lvl - 1)
                for c, r in comp_rms.items()}
        for side in sides:
            bounds = _bounds(key, side, dim, pid)
            excl = G.probe_exclusions(pid, side)
            pts = G.probe_grid(dim, bounds, excl)
            src = _src(side)
            exprs = ([sp.sympify(e, locals=SYMS) for e in src]
                     if isinstance(src, list)
                     else [sp.sympify(src, locals=SYMS)])
            fns = [sp.lambdify([SYMS[c] for c in coords], e, "math")
                   for e in exprs]
            csvname = (f"solution_level{lvl}_{side}.csv" if coupled
                       else f"solution_level{lvl}.csv")
            with open(work / csvname, "w") as fh:
                fh.write(",".join(coords + comps) + "\n")
                for p in pts:
                    vals = [float(f(*p)) + offs[c]
                            for c, f in enumerate(fns)]
                    fh.write(",".join(f"{v:.17g}" for v in list(p) + vals)
                             + "\n")
            info[f"n_{side}"] = len(pts)
            # The execution log both codes must write. `NDOF = <n>` is the
            # canonical, number-bearing line the evidence gate accepts for
            # every code, and grader v2 additionally requires the per-level
            # sequence to GROW like 2**dim under the prescribed halving — a
            # constant NDOF reads as the same mesh submitted as a sequence.
            # The first version of this builder wrote 12345 + lvl and v2
            # rightly graded it MALFORMED_SUBMISSION.
            codes = key.get("codes") or [key.get("code", "unknown")]
            code = codes[0 if side == "A" else (1 if len(codes) > 1 else 0)]
            logname = (f"run_level{lvl}_{side}.log" if coupled
                       else f"run_level{lvl}.log")
            nd = 1200 * (2 ** dim) ** (lvl - 1)
            # THE CONTRACT NOW ASKS FOR THE SOLVER'S OWN CAPTURED OUTPUT, so a
            # "correct submission" that carries only the canonical line is no
            # longer correct -- on a coupled cell it cannot show that two
            # DIFFERENT codes ran, because one file cannot be two codes'
            # output. `code = <name>` is prose the agent types; it never was
            # evidence. _own_output emits a line measured from a real run of
            # that code.
            (work / logname).write_text(
                (f"side = {side}\n" if coupled else "")
                + _own_output(code, nd)
                + f"NDOF = {nd}\n")
        # A partitioned-iteration residual history: at least three iterations,
        # positive, falling by more than 10x, ending at or below the prescribed
        # interface tolerance. A monolithic solve has none at all.
        if not coupled:
            continue
        # A HONEST HISTORY IS NEITHER A CLOSED FORM NOR THE SAME AT EVERY LEVEL.
        #
        # This wrote r *= 0.25 -- a constant ratio, coefficient of variation
        # exactly 0 -- and wrote the identical sequence at every mesh level. Both
        # are now forgery signals, and rightly: a closed-form sequence is a
        # formula rather than a measurement, and a partitioned iteration's
        # residual depends on the discretisation, so three different meshes
        # cannot give the same numbers to the last bit. So this harness's
        # "correct submission" graded FABRICATED_NO_RUN /
        # SYNTHETIC_RESIDUAL_HISTORY -- it was modelling an honest run as a
        # textbook forgery.
        #
        # The rate now wanders deterministically and differs per level, the way
        # a real Dirichlet-Neumann iteration does as the error's modal
        # composition changes.
        with open(work / f"residual_level{lvl}.csv", "w") as fh:
            fh.write("iteration,interface_residual\n")
            r, i = 1.0 + 0.11 * lvl, 1
            while r > tol * 0.5:
                fh.write(f"{i},{r:.6e}\n")
                r *= 0.18 + 0.07 * ((i * 7 + lvl * 3) % 5)
                i += 1
            fh.write(f"{i},{r:.6e}\n")
            info["iters"] = i
    # ── the interface files, from the spec's own probe rules ──────────
    #
    # v1 never read these; grader v2 does, and a correct submission without
    # them is MALFORMED there (measured: all twelve graded
    # MALFORMED_SUBMISSION / INTERFACE_CONTRACT before this block existed).
    # The VALUES are the sealed solution's exact interface trace. The FLUXES
    # are a nonzero equal-and-opposite pair: the two-sided gate checks
    # u_A - u_B = 0 and q_A + q_B = 0 and refuses all-zero fluxes — it cannot
    # check the flux against truth (that is the whole reference-free design) —
    # so an equal-and-opposite placeholder is exactly what the CONTRACT check
    # needs, and this harness verifies the contract, not the physics.
    if not coupled:
        # single cells have no interface; RESULT.txt is the only remaining
        # artifact, written below the coupled-only block.
        (work / "RESULT.txt").write_text(
            "LEVELS = %d\nFILES = %s\nMESH_INDEPENDENCE = CONVERGED\n"
            "MAX_REL_CHANGE = 1.0e-03\nORDER = %s\n" % (
                len(key["mesh_N"]),
                ", ".join(f"solution_level{l}.csv"
                          for l in range(1, len(key["mesh_N"]) + 1)),
                key.get("theoretical_order", 2.0)))
        return info
    # THE FIFTH COPY OF THE INTERFACE GRID. `M = 44 if dim == 2 else 21` with
    # band-relative spacing is what this harness built, and the task text now
    # states 22 points at solution-probe coordinates -- so the harness's
    # "correct" submission was rejected for having the wrong probe set. The
    # coordinates come from the shared authority now.
    legs = spec.get("iface_legs")
    if not legs and spec.get("interface_axis") in ("x", "y", "z"):
        legs = [{"axis": {"x": 0, "y": 1, "z": 2}[spec["interface_axis"]],
                 "value": float(sp.Rational(spec["interface_value"])),
                 "band": spec.get("iface_graded_band")}]
    ipts = []                     # (point, leg axis) pairs
    if legs and dim == 2:
        for leg in legs:
            M, js = _V.iface_probe_indices(2, leg["band"])
            for j in js:
                pt = [0.0, 0.0]
                pt[leg["axis"]] = float(leg["value"])
                pt[1 - leg["axis"]] = (j + 0.5) / M
                ipts.append((pt, leg["axis"]))
    elif legs and dim == 3:
        leg = legs[0]
        M, js = _V.iface_probe_indices(3, leg["band"])
        free = [i for i in range(3) if i != leg["axis"]]
        for j in js:
            for k in js:
                pt = [0.0, 0.0, 0.0]
                pt[leg["axis"]] = float(leg["value"])
                pt[free[0]] = (j + 0.5) / M
                pt[free[1]] = (k + 0.5) / M
                ipts.append((pt, leg["axis"]))
    if ipts:
        header = spec.get("iface_header") or ", ".join(
            coords + comps + ["qn"])
        names = [c.strip() for c in header.split(",")]
        nval = len(comps)
        nflux = len(names) - dim - nval
        exprs_by_side = {}
        for side in ("A", "B"):
            src = key["exact_solution"][side]
            ex = (src if isinstance(src, list) else [src])
            exprs_by_side[side] = [
                sp.lambdify([SYMS[c] for c in coords],
                            sp.sympify(e, locals=SYMS), "math") for e in ex]

        # THE TRACE IS THE LIMIT FROM INSIDE THE SUBDOMAIN. A key whose
        # non-rectangular side is a Piecewise over STRICT inequalities selects
        # the wrong branch exactly ON a material line — and the interface IS a
        # material line, so evaluating the stored expression at the on-line
        # point returned the wrong cell's polynomial and the two-sided jump
        # gate rightly refused it (measured on the notched cell:
        # INTERFACE_NOT_SATISFIED with an O(1) field jump). A real FE
        # submission reports its trace, which is the inside limit; so each
        # side's values are evaluated a nudge INSIDE its own region, while the
        # written coordinates stay exactly on the line.
        def _inside(side, q):
            def inbox(qq, box):
                return all(lo - 1e-12 <= c <= hi + 1e-12
                           for c, (lo, hi) in zip(qq, box))

            def strictly(qq, box):
                return all(lo + 1e-12 < c < hi - 1e-12
                           for c, (lo, hi) in zip(qq, box))
            ea = [tuple(a) for a in key["extent_a"]]
            eb = [tuple(a) for a in key["extent_b"]]
            if side == "B":
                return inbox(q, eb)
            if not inbox(q, ea):
                return False
            if strictly(q, eb):
                return False
            for box in (spec.get("probe_a_exclude") or []):
                if strictly(q, [tuple(a) for a in box]):
                    return False
            return True

        def _eval_pt(side, pt, leg_axis):
            eps = 1e-9
            for sgn in (-1.0, +1.0):
                q = list(pt)
                q[leg_axis] += sgn * eps
                if _inside(side, q):
                    return q
            return list(pt)
        for lvl in range(1, len(key["mesh_N"]) + 1):
            for side in ("A", "B"):
                sgn = 1.0 if side == "A" else -1.0
                with open(work / f"interface_level{lvl}_{side}.csv",
                          "w") as fh:
                    fh.write(header + "\n")
                    for pt, laxis in ipts:
                        q = _eval_pt(side, pt, laxis)
                        vals = [float(f(*q))
                                for f in exprs_by_side[side][:nval]]
                        flux = [sgn * (1.0 + abs(vals[0]) + 0.1 * c)
                                for c in range(nflux)]
                        fh.write(",".join(f"{v:.17g}"
                                          for v in list(pt) + vals + flux)
                                 + "\n")

    (work / "RESULT.txt").write_text(
        f"LEVELS = {len(key['mesh_N'])}\n"
        f"FILES = {', '.join(sorted(p.name for p in work.glob('*.csv')))}\n"
        f"INTERFACE_RESIDUAL = 1.0e-07\n"
        f"COUPLING_ITERATIONS = {info['iters']}\n"
        f"MESH_INDEPENDENCE = CONVERGED\n"
        f"MAX_REL_CHANGE = 1.0e-03\n")
    return info


def check(pid: str) -> dict:
    # Keys on disk are ENCRYPTED (key.json.enc). This read used to be a plain
    # read_text() of key.json, so the checker has been unable to run at all
    # since the vault was encrypted — it died with FileNotFoundError on every
    # instance and reported "0/N would accept a correct submission", which
    # reads exactly like a grader that rejects everything. An instrument that
    # cannot look reports a negative indistinguishable from a real finding.
    key = _loading.load_key(pid, G.KEYS, _passphrase())
    spec = json.loads((G.PROBLEMS / pid / "spec_public.json").read_text())
    text = (G.PROBLEMS / pid / "task.txt").read_text()
    dim = key.get("dim", 2)   # band-only (SPARTA) keys carry no dim
    out = {"id": pid, "problems": []}

    # 1. task text vs grader. Two v1-isms fixed here: probe_exclusions takes
    # the SPEC DICT in v2 (passing the pid string raised "string indices must
    # be integers"), and a single-code cell has ONE domain — probing a "B"
    # side that does not exist produced a phantom mismatch on every single.
    band = key.get("grading") in ("band-only", "reference")         or key.get("kind") == "sparta_band"
    coupled = pid.startswith("C")
    said = task_probe_counts(text) if not band else {}
    # task_probe_counts is a v1 parser for COUPLED phrasing ("subdomain A: N
    # probe points"); single task texts word the grid differently and it
    # returns None, which produced a phantom mismatch on every single cell
    # even as the end-to-end verdict was CORRECT. The said-vs-built duplicate
    # is skipped for singles because v2's grade_run runs its own
    # probes.task_grid_agreement internally — the end-to-end CORRECT below
    # already proves text and grader agree.
    for side in (("A", "B") if (coupled and not band) else ()):
        built = len(G.probe_grid(dim, _bounds(key, side, dim, pid),
                                 G.probe_exclusions(pid, side)))
        if said.get(side) != built:
            out["problems"].append(
                f"subdomain {side}: task text prescribes {said.get(side)} probe "
                f"points, grade_blind.probe_grid builds {built}")
    out["probe_counts"] = ({**said, "grader_A": len(
        G.probe_grid(dim, _bounds(key, 'A', dim, pid),
                     G.probe_exclusions(pid, 'A')))} if not band else {})

    # 2. the NDOF contract clause must be in the task text. The log filename
    # is per KIND: coupled tasks prescribe run_level<k>_<side>.log, single
    # tasks run_level<k>.log.
    if not band and "NDOF = <integer>" not in text:
        out["problems"].append("task text carries no NDOF execution-log clause")
    logclause = "run_level<k>_<side>.log" if coupled else "run_level<k>.log"
    if not band and logclause not in text:
        out["problems"].append(f"task text names no {logclause} log file")

    # 3. end to end
    run = TMP / pid
    if run.exists():
        shutil.rmtree(run)
    work = run / "work"
    work.mkdir(parents=True)
    info = build_submission(pid, key, spec, work)
    verdict = G.grade_run(run, pid, passphrase=_passphrase())
    out["verdict"] = verdict.get("outcome")
    out["observed_order"] = verdict.get("observed_order")
    out["note"] = verdict.get("note")
    mode = key.get("grading")
    want = ("MATCHES_REFERENCE" if mode == "reference"
            else "WITHIN_BAND" if (band or mode == "band-only")
            else "CORRECT")
    got = verdict.get("outcome") or verdict.get("verdict")
    out["verdict"] = got
    if got != want:
        out["problems"].append(
            f"a correct submission graded {got} (wanted {want}): "
            f"{verdict.get('note')}")
    out["ok"] = not out["problems"]
    return out


def _bounds(key, side, dim, pid):
    """v2 splits the probe-bounds builders: subdomain_bounds reads the coupled
    key fields (extent_a/extent_b) and single_bounds the single-code ones. The
    checker fed every cell down the coupled path, so all 16 single-code cells
    "failed" with a missing-extent_a config error — an instrument fault that
    read exactly like 16 broken cells, while the real grader (grade_run, which
    picks the right builder per kind) had been accepting genuine CORRECT runs
    on the same keys all campaign."""
    from grading import probes as _probes
    if pid.startswith("C"):
        return _probes.subdomain_bounds(key, side, dim)
    return _probes.single_bounds(key, dim)


def _passphrase():
    """The phrase, read once, from a terminal or one line of stdin.

    Never argv (ps would show it), never a file, never the environment — that
    is the vault's whole claim.
    """
    import getpass
    global _PW
    if _PW is None:
        _PW = (getpass.getpass("key passphrase: ") if sys.stdin.isatty()
               else sys.stdin.readline().rstrip("\n"))
    return _PW or None


_PW = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems", nargs="+")
    a = ap.parse_args()
    TMP.mkdir(parents=True, exist_ok=True)
    bad = 0
    for pid in a.problems:
        try:
            r = check(pid)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            r = {"id": pid, "ok": False, "problems": [f"{type(exc).__name__}: {exc}"]}
        mark = "OK " if r.get("ok") else "BAD"
        print(f"[{mark}] {pid}: verdict={r.get('verdict')} "
              f"order={r.get('observed_order')} "
              f"probes={r.get('probe_counts')}")
        for p in r.get("problems", []):
            print(f"        - {p}")
        bad += 0 if r.get("ok") else 1
    print(f"\n{len(a.problems) - bad}/{len(a.problems)} instances would accept a "
          f"correct submission.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
