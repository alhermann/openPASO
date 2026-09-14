"""The interface phase — the quantities that move when the coupling breaks.

Measured (DESIGN.md Amendment 2 §2): every interface-mechanism mutation of a
partitioned coupling still self-converges at order ~1.85, indistinguishable
from the correct run, while the two-sided flux jump goes from roundoff (~1e-14)
to O(1). So for a coupled cell the jump is not a diagnostic to report — it is a
GATE, and this module is where it closes.

Repairs carried by this rebuild:

* **Per-leg normal axis.** The previous phase hardcoded axis 0 and applied the
  single `iface_graded_band` to every tangential coordinate. On D5 — whose
  interface is a bent polyline with one leg normal to x and one normal to y —
  that rejects every leg-2 point of a CORRECT submission as "outside the
  graded band". The interface is now a list of LEGS, each with its own normal
  axis, position and graded band, from structured spec data
  (`iface_legs` | `interface_axis`+`interface_value` | extents that touch);
  a coupled spec from which no leg can be derived is a hard error, never a
  guess.

* **The graded band is honoured per leg, in both directions.** Points at the
  interface ENDS sit on Dirichlet-Neumann corners where the recovered flux
  gets WORSE under refinement (measured 2.11x -> 2.51x over a 4x refinement):
  grading them fails a correct submission. Points OUTSIDE the band are refused
  just as firmly: excluding the ends must not become a way for the agent to
  choose where it is measured.

* **NOT CHECKED is not PASSED.** An interface submission that is missing,
  empty, without flux columns, or carrying identically-zero fluxes balances
  trivially and proves nothing; each of those cases is surfaced as an explicit
  finding on the cell (the same discipline as
  `core.quality_checks.check_interface_balance`, whose silent-pass siblings
  were closed on 2026-08-11), and none of them lets the cell grade CORRECT.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from . import constants as C
from .loading import GraderConfigError, interface_mod

IFACE_FILE = re.compile(r"interface_level(\d+)_([ABab])\.csv$")
SOL_FILE = re.compile(r"solution_level(\d+)_([ABab])\.csv$")
_AXIS = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class Leg:
    axis: int          # coordinate the leg is a level set of
    value: float       # the level-set value
    band: tuple | None  # graded band on the tangential coordinate(s)


def _num(v) -> float:
    """Spec numbers arrive as floats or as fraction strings ('5/8')."""
    if isinstance(v, (int, float)):
        return float(v)
    return float(Fraction(str(v).strip()))


def interface_legs(spec: dict, key: dict, dim: int) -> list[Leg]:
    """The structured description of where the interface is graded.

    Priority: explicit `iface_legs` in the public spec; then the C-series
    `interface_axis`/`interface_value` pair; then a single leg derived from
    key extents that touch on exactly one axis. Anything else refuses — the
    grader must never guess geometry (that is how probe grids end up built
    over the wrong region).
    """
    legs = spec.get("iface_legs")
    if legs:
        out = []
        for i, leg in enumerate(legs):
            try:
                band = leg.get("band")
                out.append(Leg(int(leg["axis"]), _num(leg["value"]),
                               tuple(float(b) for b in band) if band else None))
            except (KeyError, TypeError, ValueError) as ex:
                raise GraderConfigError(
                    f"iface_legs[{i}] is malformed ({ex}); each leg needs "
                    f"'axis', 'value' and optionally 'band'")
        return out

    band = spec.get("iface_graded_band")
    band = tuple(float(b) for b in band) if band else None

    axis_name = spec.get("interface_axis")
    if axis_name is not None and spec.get("interface_value") is not None:
        axis = _AXIS.get(str(axis_name).strip().lower())
        if axis is None or axis >= dim:
            raise GraderConfigError(
                f"interface_axis {axis_name!r} is not a coordinate of this "
                f"{dim}D problem. A bent interface (e.g. a polyline) cannot "
                f"be described by one axis: record `iface_legs` in the public "
                f"spec — a list of {{axis, value, band}} per leg, matching "
                f"the INTERFACE PROBE POINTS the task text states.")
        return [Leg(axis, _num(spec["interface_value"]), band)]

    ea, eb = key.get("extent_a"), key.get("extent_b")
    if ea and eb:
        touching = [i for i in range(dim)
                    if abs(ea[i][1] - eb[i][0]) < 1e-9
                    or abs(eb[i][1] - ea[i][0]) < 1e-9]
        if len(touching) == 1:
            i = touching[0]
            value = ea[i][1] if abs(ea[i][1] - eb[i][0]) < 1e-9 else ea[i][0]
            return [Leg(i, float(value), band)]

    raise GraderConfigError(
        "no structured interface description: the spec carries neither "
        "`iface_legs` nor `interface_axis`/`interface_value`, and the key "
        "extents do not touch on exactly one axis. A bent or non-adjacent "
        "interface must be described in spec_public.json (public: the task "
        "text already prints it); the grader will not guess where the "
        "coupling is graded.")


def iface_column_counts(spec: dict, dim: int, ncomp: int) -> tuple[int, int]:
    """(n values, n fluxes) per interface row, from the public `iface_header`
    (e.g. 'x, y, T, ux, uy, qn, tx, ty'). Without a header the legacy layout
    (ncomp values then ncomp fluxes) is assumed."""
    header = spec.get("iface_header")
    if not header:
        return ncomp, ncomp
    names = [c.strip() for c in str(header).split(",") if c.strip()]
    nflux = len(names) - dim - ncomp
    if nflux < 1:
        raise GraderConfigError(
            f"iface_header {header!r} has {len(names)} columns, which cannot "
            f"hold {dim} coordinates + {ncomp} components + a flux")
    return ncomp, nflux


def _assign_to_leg(pts, legs):
    """point index -> leg index by |p[axis] - value| <= PROBE_TOL, or None."""
    out = []
    for p in pts:
        hit = None
        for li, leg in enumerate(legs):
            if abs(p[leg.axis] - leg.value) <= C.PROBE_TOL:
                hit = li
                break
        out.append(hit)
    return out


def outward_signs(spec: dict, leg: Leg) -> dict | None:
    """+1 if a side's outward normal points along +e_axis at this leg, else -1.

    Derived from the structured extents, never from prose. Returns ``None``
    when the extents do not place the two sides on opposite sides of the leg —
    a bent interface (C5/D5, where A is not a rectangle) or a spec without
    extents — because a guessed sign turns a correct submission into a
    "reversed convention" finding.
    """
    ea, eb = spec.get("extent_a"), spec.get("extent_b")
    if not ea or not eb or leg.axis >= len(ea) or leg.axis >= len(eb):
        return None
    a_hi, b_lo = float(ea[leg.axis][1]), float(eb[leg.axis][0])
    a_lo, b_hi = float(ea[leg.axis][0]), float(eb[leg.axis][1])
    tol = 1e-9 * max(1.0, abs(leg.value))
    if abs(a_hi - leg.value) <= tol and abs(b_lo - leg.value) <= tol:
        return {"A": 1.0, "B": -1.0}          # A below the leg, B above
    if abs(b_hi - leg.value) <= tol and abs(a_lo - leg.value) <= tol:
        return {"A": -1.0, "B": 1.0}
    return None


def flux_from_field_phase(work: Path, spec: dict, dim: int, ncomp: int,
                          nval: int, nflux: int, legs: list) -> dict:
    """Is each side's reported interface flux its OWN field's normal derivative?

    THE HOLE THIS FILLS. Every other coupled check compares the two sides to
    each other, so a submission in which both sides come from one computation
    passes them all. The bit-exact-zero rule catches the crudest form of that.
    This catches the general form: the reported flux must be a constant times
    the normal derivative of the field the SAME side submitted, and the
    constant it implies must be the same at every interface point.

    Measured, with the answers sealed, on C8_27b_BARE_seed4 — the check
    recovers the task's own conductivities from the agent's field alone:

        level 1  side A  implied k = 0.9999999 (+/-0.17%)   side B  999.99992
        level 2  side A  implied k = 0.9999999 (+/-0.23%)   side B  999.99992
        level 3  side A  implied k = 0.9620255 (+/-0.18%)   side B  1014.1155

    Calibration across the 98 applicable coupled runs in the tree: the implied
    coefficient's spread along the interface is 0.9% (median) for consistent
    submissions with a p90 of 17.7%, against p10 = 39.3% and a median of 179%
    for inconsistent ones. The 30% threshold sits inside that gap; it was
    chosen from this measurement, not picked.

    WHAT IT ACTUALLY CAUGHT, MEASURED, AND IT IS NOT A HEADLINE. Swept through
    this phase over all 427 coupled run directories on disk: **it flags zero
    runs that the two-sided jump gate passes.** Every INCONSISTENT it finds
    (17 INTERFACE_NOT_SATISFIED, 7 NOT_CHECKED) was already failing. So the
    hole it closes is one that no submission in the tree exploits, and no
    number here may be presented as a detection it made.

    Its measured value is elsewhere, and is real: 19 runs are CONSISTENT here
    while the two-sided gate says INTERFACE_NOT_SATISFIED. For those, each side
    computed its flux honestly from its own field and the TRANSMISSION
    CONDITION is what is wrong — a different repair instruction from "your flux
    was invented", and one no other check can give. Another 8 agree with the
    two-sided gate in the passing direction, which is corroboration rather
    than new information.

    IT RECORDS AND NAMES; IT DOES NOT YET DECIDE. An INCONSISTENT verdict is
    strong evidence that the flux was not computed from the field, but the rule
    that turns evidence into FABRICATED demands positive proof of invention,
    and this check's false-positive rate has not been measured against a
    regrade under the current gate. So it writes findings and a reason and
    leaves the outcome to the phase that owns it. Its CONSISTENT verdict is
    already useful in the other direction: a flux that reproduces a single
    coefficient to seven digits was not invented.
    """
    IF = interface_mod()
    out = {"verdict": "NOT_ASSESSED", "per_side": [], "findings": [],
           "reasons": []}
    comps, why = IF.scalar_flux_components(spec)
    if not comps:
        out["verdict"] = "NOT_APPLICABLE"
        out["detail"] = why
        return out
    if len(legs) != 1:
        out["detail"] = (f"{len(legs)} interface legs: the outward normal "
                         f"cannot be assigned per side from the extents alone")
        return out
    leg = legs[0]
    signs = outward_signs(spec, leg)
    if signs is None:
        out["detail"] = ("the extents do not place the two sides on opposite "
                         "sides of the interface, so the outward normal "
                         "direction is not derivable and nothing is asserted")
        return out

    ifs, sols = {}, {}
    for f in sorted(work.rglob("*level*.csv")):
        m = IFACE_FILE.match(f.name)
        if m:
            ifs[(int(m.group(1)), m.group(2).upper())] = f
        m = SOL_FILE.match(f.name)
        if m:
            sols[(int(m.group(1)), m.group(2).upper())] = f

    verdicts = []
    for (lvl, side), ifp in sorted(ifs.items()):
        sp = sols.get((lvl, side))
        if sp is None:
            continue
        gi, _ = IF.read_interface_csv(ifp, dim, nval, nflux)
        gs, _ = IF.read_interface_csv(sp, dim, nval, 0)
        if gi is None or gs is None:
            continue
        ipts, _iv, iq = gi
        fpts, fv, _fq = gs
        dudn = IF.recover_normal_derivative(fpts, fv, ipts, leg.axis,
                                            leg.value, signs[side])
        res = IF.flux_ratio_consistency(iq, dudn, components=comps)
        res.update({"level": lvl, "side": side})
        out["per_side"].append(res)
        verdicts.append(res["verdict"])

    if not verdicts:
        out["detail"] = ("no level has both an interface file and a solution "
                         "file for the same side")
        return out

    # JUDGE THE FINEST LEVEL, NOT THE WORST ONE.
    #
    # The recovery is a one-sided O(h^2) extrapolation, so at a coarse mesh its
    # own error inflates the spread and a fixed tolerance measures the MESH
    # rather than the agent. Measured on C8_27b_MCP_seed4, side B:
    #
    #     level 1  k = 975.8  spread 34.8%   <- would fail a fixed threshold
    #     level 2  k = 993.1  spread 15.1%
    #     level 3  k = 996.5  spread  3.0%   <- converging to k = 1000
    #
    # That is a flux which does follow from its field, seen through a coarse
    # estimator. This is the same lesson the two-sided jump gate learned the
    # hard way (see the long note in interface_phase): a mesh-dependent
    # estimate must be judged where it is accurate, or on its trend.
    finest = max(r["level"] for r in out["per_side"])
    at_finest = [r["verdict"] for r in out["per_side"]
                 if r["level"] == finest]
    for v in ("INCONSISTENT", "SIGN_CONVENTION", "NOT_ASSESSED", "CONSISTENT"):
        if v in at_finest:
            out["verdict"] = v
            break
    out["judged_at_level"] = finest
    coarse = sorted({r["verdict"] for r in out["per_side"]
                     if r["level"] != finest} - set(at_finest))
    if coarse:
        out["coarse_level_verdicts"] = coarse
    if out["verdict"] == "INCONSISTENT":
        worst = [r for r in out["per_side"]
                 if r["verdict"] == "INCONSISTENT" and r["level"] == finest]
        out["reasons"].append("FLUX_NOT_FROM_SUBMITTED_FIELD")
        out["findings"].append(
            "the reported interface flux is not a constant multiple of the "
            "normal derivative of this side's own submitted field, at "
            + ", ".join(f"level {r['level']} side {r['side']}" for r in worst[:6])
            + ". For a scalar conduction flux q_n = -k du/dn the ratio must be "
              "the conductivity at EVERY interface point; a varying ratio "
              "means the flux and the field did not come from one another. "
              "Export the flux from the same assembled system that produced "
              "the field.")
    elif out["verdict"] == "SIGN_CONVENTION":
        out["reasons"].append("FLUX_SIGN_CONVENTION")
        out["findings"].append(
            "the reported flux is proportional to this side's own field "
            "normal derivative but with the OPPOSITE sign, so the flux was "
            "computed with the INWARD normal. The task defines "
            "q_n = -(K grad u) . n_out with n_out pointing OUT of the "
            "subdomain; with the sign reversed the two sides' fluxes appear "
            "to sum to zero when they do not, and vice versa.")
    elif out["verdict"] == "NOT_ASSESSED":
        flat = [r for r in out["per_side"]
                if any("flat at the interface" in (d.get("detail") or "")
                       for d in r.get("per_component", []))]
        if flat:
            out["findings"].append(
                "the submitted FIELD is flat at the interface — its normal "
                "derivative is identically zero — at "
                + ", ".join(f"level {r['level']} side {r['side']}"
                            for r in flat[:6])
                + ". A field with no gradient at the interface transports "
                  "nothing across it, so whatever flux was reported cannot "
                  "have come from it.")
    return out


def interface_phase(work: Path, spec: dict, key: dict, dim: int,
                    ncomp: int, mesh_N, legs: list | None = None) -> dict:
    """Grade the interface submission. Returns
        verdict   INTERFACE_SATISFIED | INTERFACE_NOT_SATISFIED | NOT_CHECKED
        malformed list of contract violations (missing/empty/short files,
                  stray points) — these make the SUBMISSION malformed
        reasons   machine-readable gate reasons when not satisfied
        findings  explicit NOT-CHECKED / refusal findings (never silent)
        per_level worst jumps per graded level

    `legs` is normally validated up front by the orchestrator (a spec from
    which no leg can be derived is a CELL defect and must halt grading before
    any submission is read); it is derived here only when called standalone.
    """
    IF = interface_mod()
    legs = legs if legs is not None else interface_legs(spec, key, dim)
    nval, nflux = iface_column_counts(spec, dim, ncomp)

    out = {"verdict": "NOT_CHECKED", "malformed": [], "reasons": [],
           "findings": [], "per_level": [], "legs": [
               {"axis": leg.axis, "value": leg.value, "band": leg.band}
               for leg in legs]}

    # ONE-SIDED evidence: does each side's flux follow from its OWN field?
    # Every other check here is two-sided, so a submission whose two sides come
    # from a single computation satisfies all of them. Computed for every
    # coupled run, recorded whatever it says, and never allowed to raise: a
    # diagnostic that can abort grading is worse than no diagnostic.
    # Its FINDINGS surface on the cell; its REASONS deliberately do NOT join
    # `out["reasons"]`, because the caller feeds those straight into the
    # outcome (`all_reasons = list(iface_reasons)` in grade_blind_v2). Until
    # this check's false-positive rate is measured against a regrade under the
    # current gate, it must not be able to change a grade — the promise in
    # flux_from_field_phase's docstring is kept here or nowhere.
    try:
        out["flux_from_field"] = flux_from_field_phase(
            work, spec, dim, ncomp, nval, nflux, legs)
        out["findings"].extend(out["flux_from_field"]["findings"])
    except Exception as exc:                                   # noqa: BLE001
        out["flux_from_field"] = {"verdict": "ERROR", "detail": repr(exc)}

    lvls: dict[int, dict[str, Path]] = {}
    # RECURSIVE, like submission._submission_candidates. This was a
    # non-recursive glob while the solution-file discovery next door used
    # rglob — the exact asymmetry submission.py's docstring records as fixed
    # for solutions and never fixed here. Measured consequence: of the 5
    # coupled cells booked "no interface_level files", 4 HAD submitted them
    # (into results/, output/, level1/), and one of those four satisfies the
    # interface gate. Agents organise their output into subdirectories; the
    # grader must find it where the rest of the grader already looks.
    for f in sorted(work.rglob("interface_level*.csv")):
        m = IFACE_FILE.match(f.name)
        if m:
            lvls.setdefault(int(m.group(1)), {})[m.group(2).upper()] = f

    nlevels = len(mesh_N or [])
    if not lvls:
        out["malformed"].append(
            "no interface_level<k>_<side>.csv files: the task requires them, "
            "and the interface flux is the only reference-free quantity that "
            "can detect convergence to the wrong transmission condition")
        out["findings"].append(
            "Interface NOT CHECKED: nothing was submitted for it. This is "
            "not a passing check — nothing was compared.")
        return out
    missing_levels = [k for k in range(1, nlevels + 1)
                      if set(lvls.get(k, {})) != {"A", "B"}]
    if missing_levels:
        out["malformed"].append(
            f"interface files missing or one-sided for level(s) "
            f"{missing_levels}: every prescribed level needs both "
            f"interface_level<k>_A.csv and _B.csv")

    zero_flux_levels, graded = [], []
    for lvl in sorted(lvls):
        sides = lvls[lvl]
        if set(sides) != {"A", "B"}:
            continue
        parsed = {}
        for s, p in sorted(sides.items()):
            got, why = IF.read_interface_csv(p, dim, nval, nflux)
            if got is None:
                out["malformed"].append(f"{p.name}: {why}")
                break
            parsed[s] = got
        if len(parsed) != 2:
            continue

        # the grader owns the interface evaluation set: every point must lie
        # ON a declared leg and INSIDE that leg's graded band
        stray = []
        assignment = {}
        for s, (pts, _v, _q) in parsed.items():
            hits = _assign_to_leg(pts, legs)
            assignment[s] = hits
            for p, hit in zip(pts, hits):
                if hit is None:
                    stray.append((s, p, "on no declared interface leg"))
                    continue
                leg = legs[hit]
                if leg.band is None:
                    continue
                lo, hi = leg.band[0] - 1e-9, leg.band[1] + 1e-9
                tang = [c for i, c in enumerate(p) if i != leg.axis]
                if any(not (lo <= c <= hi) for c in tang):
                    stray.append((s, p, f"outside the graded band {leg.band}"))
        if any(leg.band is None for leg in legs):
            out["findings"].append(
                "IFACE BAND NOT RECORDED for at least one leg: the interface "
                "ends (Dirichlet-Neumann corners) are not being excluded for "
                "this cell; record iface_graded_band / iface_legs bands in "
                "the public spec")
        if stray:
            out["malformed"].append(
                f"level {lvl}: {len(stray)} interface point(s) refused, e.g. "
                f"{stray[0][1]} ({stray[0][2]}). The interface ends are "
                f"corners of the split problem and are deliberately not "
                f"graded; points elsewhere than the prescribed probes would "
                f"let the agent choose where it is measured.")
            continue

        # all-zero flux balances trivially and proves nothing — the signature
        # of a transfer that never ran, a wrong field name, or an unfilled
        # buffer. NOT CHECKED, and said so.
        qa = [x for _p, _v, q in [parsed["A"]] for row in q for x in row]
        qb = [x for _p, _v, q in [parsed["B"]] for row in q for x in row]
        if qa and qb and not any(abs(x) > 0 for x in qa + qb):
            zero_flux_levels.append(lvl)
            out["findings"].append(
                f"Interface flux balance NOT CHECKED at level {lvl}: both "
                f"sides exported fluxes that are identically zero. That "
                f"balances trivially and says nothing about the coupling.")
            continue

        # per-leg two-sided jumps at matched points with opposite outward
        # normals: continuity means u_A - u_B = 0 and q_A + q_B = 0
        worst_u = worst_q = 0.0
        leg_ok = True
        for li in range(len(legs)):
            sub = {}
            for s, (pts, vals, flux) in parsed.items():
                keep = [i for i, hit in enumerate(assignment[s]) if hit == li]
                sub[s] = ([pts[i] for i in keep], [vals[i] for i in keep],
                          [flux[i] for i in keep])
            if not sub["A"][0] and not sub["B"][0]:
                continue
            d, why = IF.two_sided_jumps(sub["A"], sub["B"])
            if d is None:
                out["malformed"].append(f"level {lvl}, leg {li + 1}: {why}")
                leg_ok = False
                break
            worst_u = max(worst_u, d["jump_u_rel"])
            worst_q = max(worst_q, d["jump_q_rel"])
        if not leg_ok:
            continue
        graded.append({"level": lvl, "jump_u_rel": worst_u,
                       "jump_q_rel": worst_q})

    out["per_level"] = graded
    if not graded:
        if zero_flux_levels:
            out["reasons"].append("INTERFACE_FLUX_ALL_ZERO")
            out["verdict"] = "NOT_CHECKED"
        return out

    if zero_flux_levels:
        out["reasons"].append("INTERFACE_FLUX_ALL_ZERO")
    # THE GATE MUST TEST WHAT ITS OWN MESSAGE CLAIMS: a jump that STAYS O(1)
    # under refinement. It took the MAX over levels against a fixed tolerance,
    # which is a different test and a much worse one.
    #
    # Why it is worse. When both sides recover the interface flux from their
    # assembled system, the two exported arrays differ by the consistent-to-
    # nodal conversion of the P1 boundary mass matrix, so the graded jump is
    # h^2 |q''| / (6|q|) — a MESH RULER with no physics in it. Two independent
    # reviews demonstrated the consequence: a coupling with one side's
    # conductivity 4x WRONG fails at n=8 and n=16 and PASSES at n=32, purely
    # on resolution. And a correct coupling whose tractions are evaluated by
    # direct stress sampling — which is what the task text describes, "the
    # traction evaluated from that subdomain's own solution" — is O(h) at the
    # interface and fails a fixed 5e-3 while converging perfectly well.
    #
    # Measured in this campaign's own grades: 3 coupled cells failed SOLELY on
    # this gate, and 2 of the 3 had a flux jump decreasing at every level
    # (C8 seed 4, BARE: 3.3% -> 2.0% -> 1.1%; MCP: 8.9% -> 3.6% -> 1.5%).
    # Those are converging couplings graded as unphysical.
    #
    # It also mattered asymmetrically: the consistent recovery that passes a
    # fixed tolerance is described in the openPASO payload and nowhere in the task
    # text, so a grading-critical rule was published to one arm. Testing for
    # non-convergence instead removes that, because both recoveries converge.
    #
    # THE REPLACEMENT. Fail when the jump does not shrink under refinement,
    # which is precisely the mutation signature the gate was calibrated on
    # (mutations sit at 0.75-3.0 and flat; a correct run is at roundoff or
    # decreasing). A single level cannot show a trend, so there the fixed
    # tolerance still applies — it is all the evidence there is.
    worst_u = max(g["jump_u_rel"] for g in graded)
    worst_q = max(g["jump_q_rel"] for g in graded)

    def _shrinks(key):
        """Does the jump fall by a clear factor across the sequence?"""
        v = [g[key] for g in graded]
        if len(v) < 2:
            return None                      # no trend available
        if max(v) <= C.IFACE_JUMP_TOL:
            return True                      # at roundoff/tolerance throughout
        first, last = v[0], v[-1]
        if first <= 0:
            return last <= C.IFACE_JUMP_TOL
        # h halves per level, so a genuine O(h) recovery falls ~2x per level.
        # Require a clear overall fall, not a per-step monotone chain, so that
        # one noisy level does not condemn a converging sequence — AND a floor
        # on where it lands, because decay alone admitted a 49% mismatch.
        return (last < first * (C.IFACE_JUMP_DECAY ** (len(v) - 1))
                and last <= C.IFACE_JUMP_CEILING)

    su, sq = _shrinks("jump_u_rel"), _shrinks("jump_q_rel")
    if su is None or sq is None:
        bad = worst_u > C.IFACE_JUMP_TOL or worst_q > C.IFACE_JUMP_TOL
        why = (f"only {len(graded)} graded level(s), so no refinement trend "
               f"exists and the fixed tolerance {C.IFACE_JUMP_TOL:g} is all "
               f"the evidence there is")
    else:
        bad = not (su and sq)
        _u = ", ".join("%.3e" % g["jump_u_rel"] for g in graded)
        _q = ", ".join("%.3e" % g["jump_q_rel"] for g in graded)
        why = (f"the jump does not shrink under refinement "
               f"(field [{_u}], flux [{_q}])")
    # A BIT-EXACT ZERO FLUX JUMP IS NOT A PERFECT COUPLING. IT IS ONE FIELD.
    #
    # Two independently solved subdomains cannot agree on the interface flux to
    # the last bit. Exactly 0.0 means both sides were evaluated from the same
    # solution — the signature of a monolithic solve reported as a partitioned
    # one. C7_27b_BARE_seed2 scored 0.0 for field AND flux at all three levels
    # and graded CORRECT on that basis; its own notes say the coupling was
    # "simulated based on the monolithic solution".
    #
    # Only the FLUX carries this meaning. A zero FIELD jump is ordinary and must
    # not be flagged: Dirichlet-Neumann sets one side's interface displacement
    # to the other's, so field agreement can be exact by construction. Measured
    # across the 60 graded interface levels in the tree: jump_u_rel is exactly
    # zero 12 times, jump_q_rel exactly 3 times — and those 3 are this one
    # fabricated run.
    exact_zero_flux = [g["level"] for g in graded if g["jump_q_rel"] == 0.0]
    if exact_zero_flux and not bad:
        # NOT "the subdomains disagree" — the opposite. There is nothing here
        # to agree or disagree, so the gate has no evidence either way and
        # says so, rather than accusing the physics of being wrong.
        #
        # Two solves converged to a 1e-6 relative interface tolerance leave a
        # jump of about that size, not of zero. A bit-exact cancellation means
        # one array was written twice with a sign flip: either both sides were
        # read off a single field (C7_27b_BARE_seed2, which then graded
        # CORRECT), or the agent took "the fluxes are equal and opposite" as an
        # instruction to construct side B from side A. The second is an honest
        # misreading rather than a forgery, and it still leaves the coupled
        # claim — that two codes agreed — with no support at all.
        out["verdict"] = "NOT_CHECKED"
        out["reasons"].append("INTERFACE_NO_TWO_SIDED_EVIDENCE")
        out["findings"].append(
            f"the relative FLUX jump is exactly 0.0 at level(s) "
            f"{exact_zero_flux}. Two independently solved subdomains do not "
            f"agree to the last bit; a bit-exact cancellation means side B's "
            f"flux was constructed from side A's rather than computed, so "
            f"this submission carries no evidence that two codes met at the "
            f"interface. Export each side's flux from its OWN assembled "
            f"system and let the small residual mismatch show.")
        return out

    if bad:
        out["verdict"] = "INTERFACE_NOT_SATISFIED"
        out["reasons"].append("INTERFACE_NOT_SATISFIED")
        out["findings"].append(
            f"the two subdomains disagree at the interface: max relative "
            f"field jump {worst_u:.3e}, max relative FLUX jump {worst_q:.3e}. "
            f"{why}. A jump that stays O(1) under refinement means the scheme "
            f"converged to a fixed point of the wrong transmission condition "
            f"— which the observed order cannot see, because convergence to a "
            f"wrong answer is still convergence.")
    elif zero_flux_levels:
        out["verdict"] = "NOT_CHECKED"
    else:
        out["verdict"] = "INTERFACE_SATISFIED"
    return out
