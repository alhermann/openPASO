"""Turn a grades directory into the numbers the SCORECARD asks for.

Reads a grades directory written by `grade_round.py` and reports, per arm:
single-code solve rate, coupled solve rate, fabrication rate, the paired
McNemar counts, and the countable holes (`per_code_attribution: UNPROVEN`,
GRADER_EXCEPTION, MALFORMED_SUBMISSION). Nothing here opens a key — it reads
the grader's output.

    python report_grades.py grades_2026_09_01

WHAT IT REFUSES TO DO. It does not pool the three evidence grades, it does not
count a cell twice, and it does not report an uplift from unpaired data: the
McNemar table is built only from (cell, seed) pairs where BOTH arms have a
verdict, because an unpaired difference is a difference in which runs finished.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
SOLVED = ("CORRECT", "CORRECT_SUPERCONVERGENT")


def load(outdir: Path) -> list:
    rows = []
    for f in sorted(glob.glob(str(outdir / "*seed*.json"))):
        m = re.search(r"seed(\d+)", Path(f).stem)
        if not m:
            continue
        seed = int(m.group(1))
        try:
            data = json.loads(Path(f).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print(f"  !! {Path(f).name}: {exc}")
            continue
        for key, rec in data.items():
            cell, arm = key.rsplit("_", 1)
            rows.append({"cell": cell, "arm": arm, "seed": seed,
                         "kind": rec.get("kind") or (
                             "coupled" if cell[0] in "CD" and cell[1:].isdigit()
                             else "single"),
                         "outcome": rec.get("outcome"),
                         "order": rec.get("observed_order"),
                         "attrib": rec.get("per_code_attribution"),
                         "grade": rec.get("evidence_grade")})
    return rows


def rate(rows, kind, arm, grade=None) -> tuple:
    """Solve rate. `grade` filters the evidence grade; None means ALL, pooled.

    THE DOCSTRING SAID IT REFUSED TO POOL THE GRADES AND THE CODE POOLED THEM.
    Caught by running it: `grades_pre_naming_fixes` holds 733 grade-1, 76
    grade-3 and 24 grade-2 verdicts, and the headline rate mixed all three.
    Grade 1 is the only one carrying a true error against an exact solution;
    grade 3 is a band check and cannot support a solve claim at all. Pooling
    them is the specific thing this campaign forbids, so the headline is grade 1
    and the others are shown beside it, never added in.
    """
    sel = [r for r in rows if r["kind"] == kind and r["arm"] == arm
           and (grade is None or r["grade"] == grade)]
    ok = sum(1 for r in sel if (r["outcome"] or "") in SOLVED)
    return ok, len(sel)


def main(argv) -> int:
    outdir = D / (argv[1] if len(argv) > 1 else "grades_2026_09_01")
    if not outdir.is_dir():
        print(f"no such grades directory: {outdir}")
        return 2
    rows = load(outdir)
    if not rows:
        print(f"{outdir} holds no gradeable rows yet")
        return 2
    seeds = sorted({r["seed"] for r in rows})
    print(f"{outdir.name}: {len(rows)} verdicts, seeds {seeds}\n")

    # KINDS COME FROM THE DATA. Hardcoding ("single", "coupled") silently
    # dropped 52 SPARTA verdicts whose `kind` is "sparta" — a whole cell family
    # missing from the report with no line saying so.
    kinds = sorted({r["kind"] for r in rows})
    print(f"SOLVE RATE, EVIDENCE GRADE 1 ONLY (a true error against the "
          f"sealed solution)")
    for kind in kinds:
        line = f"  {kind:8s}"
        for arm in ("BARE", "MCP"):
            ok, n = rate(rows, kind, arm, grade=1)
            line += f"   {arm} {ok}/{n}" + (f" = {ok/n:.1%}" if n else "")
        b_ok, b_n = rate(rows, kind, "BARE", grade=1)
        m_ok, m_n = rate(rows, kind, "MCP", grade=1)
        if b_n and m_n:
            line += f"   uplift {m_ok/m_n - b_ok/b_n:+.1%}"
        print(line)
    others = sorted({r["grade"] for r in rows} - {1, None})
    for g in others:
        print(f"\n  evidence grade {g} shown SEPARATELY — never added to the "
              f"above:")
        for kind in kinds:
            line = f"    {kind:8s}"
            any_n = False
            for arm in ("BARE", "MCP"):
                ok, n = rate(rows, kind, arm, grade=g)
                any_n = any_n or n
                line += f"   {arm} {ok}/{n}" + (f" = {ok/n:.1%}" if n else "")
            if any_n:
                print(line)
    nog = sum(1 for r in rows if r["grade"] is None)
    if nog:
        print(f"\n  {nog} verdict(s) carry NO evidence grade and are in no "
              f"rate above")

    print("\nOUTCOME BREAKDOWN")
    tab = defaultdict(Counter)
    for r in rows:
        tab[(r["kind"], r["arm"])][r["outcome"] or "(none)"] += 1
    outs = sorted({o for c in tab.values() for o in c})
    print("  " + "".ljust(16) + "".join(o[:19].ljust(21) for o in outs))
    for k in sorted(tab):
        n = sum(tab[k].values()) or 1
        print(f"  {k[0][:6]:6s} {k[1]:5s} " + "".join(
            f"{tab[k][o]}".ljust(21) for o in outs) + f" n={n}")

    print("\nFABRICATION (FABRICATED_NO_RUN)")
    for arm in ("BARE", "MCP"):
        sel = [r for r in rows if r["arm"] == arm]
        fab = sum(1 for r in sel if r["outcome"] == "FABRICATED_NO_RUN")
        print(f"  {arm:5s} {fab}/{len(sel)}"
              + (f" = {fab/len(sel):.1%}" if sel else ""))

    print("\nPAIRED McNEMAR (only (cell, seed) with a verdict in BOTH arms)")
    by = {(r["cell"], r["seed"], r["arm"]): r for r in rows}
    for kind in kinds:
        pairs = [(c, s) for (c, s, a) in by if a == "MCP"
                 and (c, s, "BARE") in by and by[(c, s, a)]["kind"] == kind]
        a = b = both = neither = 0
        for c, s in pairs:
            m = (by[(c, s, "MCP")]["outcome"] or "") in SOLVED
            bb = (by[(c, s, "BARE")]["outcome"] or "") in SOLVED
            both += m and bb
            neither += (not m) and (not bb)
            a += m and not bb
            b += bb and not m
        print(f"  {kind:8s} pairs={len(pairs)}  both={both} neither={neither} "
              f"openPASO-only={a} bare-only={b}")
        if a + b:
            # exact binomial two-sided, no scipy needed
            from math import comb
            n, k = a + b, min(a, b)
            p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
            print(f"           exact McNemar p = {p:.4f}"
                  + ("  (significant at 0.05)" if p < 0.05 else
                     "  (NOT significant at 0.05)"))

    print("\nCOUNTABLE HOLES")
    unp = [r for r in rows if r["attrib"] == "UNPROVEN"]
    print(f"  per_code_attribution UNPROVEN : {len(unp)}"
          f"  (coupled runs whose two codes cannot be attributed)")
    for bad in ("GRADER_EXCEPTION", "MALFORMED_SUBMISSION"):
        n = sum(1 for r in rows if r["outcome"] == bad)
        print(f"  {bad:30s}: {n}")
    # A VERDICT WITH NO OUTCOME IS INVISIBLE IN EVERY RATE ABOVE, so it is
    # named here. Measured on grades_pre_naming_fixes: all 52 SPARTA verdicts
    # (26 per arm) carry outcome=None, so SP1 and SP2 have never been graded to
    # an outcome at all and no report that omits this line would show it.
    none_out = [r for r in rows if not r["outcome"]]
    if none_out:
        cells = Counter(r["cell"] for r in none_out)
        print(f"  {'NO OUTCOME RECORDED':30s}: {len(none_out)}   cells "
              f"{dict(cells)}")
        print(f"  {'':30s}  these appear in NO rate above; a cell family with "
              f"no outcome is not a zero, it is unmeasured")
    grades = Counter(r["grade"] for r in rows)
    print(f"  evidence grades present        : {dict(grades)}"
          f"   (never pooled — grade 1 is the only true-error grade)")

    print("\nREMINDER: these are DEVELOPMENT cells. They are burnt and can "
          "never be the paper's numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
