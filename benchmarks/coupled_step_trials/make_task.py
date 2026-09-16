#!/usr/bin/env python3
"""Build a ONE-SIDE task for a coupled family, from public data only.

WHY ONE SIDE. Measured over the campaign's own record, the coupled families die
before they couple: of 217 graded cells with files only 99 got both sides to
three levels, and the silent side is B in 16 of 17 stuck cells. So the question
that gates everything else is not "does the coupling converge" but "can each
code be made to write a participant that RUNS and EXPORTS at all".

This asks exactly that, for one code, at one mesh level, with no partner: the
imports file is handed over as zeros, which is physically a free/insulated
interface and is fine, because the trial is about the contract and the API, not
about the coupled answer.

Everything here comes from `spec_public.json` -- the same public facts the task
text gives an agent. No sealed key, no reference solution, no seed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

CAMPAIGN = Path.home() / "Schreibtisch" / "ofa-v2" / "campaign3_blind"
ALIAS = {"4C": "fourc", "FEniCSx": "fenics", "dolfinx": "fenics"}


def spec(problem: str) -> dict:
    return json.loads((CAMPAIGN / "problems" / problem / "spec_public.json").read_text())


def code_of(s: dict, side: str) -> str:
    codes = s.get("codes") or []
    raw = codes[0 if side == "A" else 1] if len(codes) > 1 else ""
    return ALIAS.get(raw, str(raw)).lower()


def _source(s: dict, side: str) -> str:
    src = s.get("source_public")
    if isinstance(src, dict):
        v = src.get(side)
        if isinstance(v, (list, tuple)):
            return "\n".join(f"      f_{c} = {e}"
                             for c, e in zip(s.get("components") or ["x", "y"], v))
        return f"      f = {v}"
    return f"      f = {src}"


def build(problem: str, side: str) -> str:
    s = spec(problem)
    code = code_of(s, side)
    role = (s.get("roles") or {}).get(side, "")
    extent = s.get(f"extent_{side.lower()}")
    dim = s.get("dim", 2)
    coords = s.get("coords") or ["x", "y", "z"][:dim]
    n0 = (s.get("mesh_N") or [8])[0]
    comps = s.get("components") or ["u"]
    other = "B" if side == "A" else "A"

    lines = [
        f"Write ONE participant script for the {s.get('code_a_label' if side == 'A' else 'code_b_label', code)} "
        f"side of a partitioned coupling, run it once, and prove it produced its exports.",
        "Do not write the partner side. Do not run a coupling.",
        "",
        f"Read the participant contract FIRST with knowledge(topic='coupling', solver='{code}')"
        + (f", physics='{s.get('physics_family')}'" if s.get("physics_family") else "")
        + ". Follow it exactly: the script reads ./imports.json, solves once, writes ./exports.json.",
        "",
        f"THE SUBDOMAIN, which is the {role.upper()} side:",
        f"  code: {code}",
        f"  domain: subdomain {side} = " + " x ".join(
            f"({lo}, {hi})" for lo, hi in (extent or [])),
        f"  equation: {s.get('equation')}",
        f"  coefficients: {s.get('coefficients')}",
        f"  elements: {s.get('element')}",
        f"  mesh: uniform, {n0} cells per side (the coarsest level only)",
        f"  outer boundary: {s.get('bc_text')}",
        f"  interface: {s.get('interface')} -- this is the {role} boundary of this subdomain",
        f"  the exchanged quantity is the {s.get('flux_word', 'flux')}",
        "  source term:",
        _source(s, side),
        "",
        "FOR THIS TRIAL THE PARTNER DOES NOT EXIST. Create ./imports.json yourself with a",
        f"ZERO imported field of the shape the contract requires for partner '{other}', and run",
        "against it. A zero exchange is physically a free interface; that is fine. This trial",
        "is about whether your script RUNS and EXPORTS, not about the coupled answer.",
        "",
        "YOU ARE DONE when all three are true in your working directory:",
        "  1. the participant script exists,",
        "  2. running it exits 0,",
        f"  3. exports.json exists, carrying {', '.join(comps)} at the interface probe points",
        "     in the shape the contract prescribes.",
        "Print the first three lines of exports.json as your final answer.",
    ]
    if s.get("probe_iface"):
        lines.insert(-3, f"  interface probe points: {s['probe_iface']}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(build(sys.argv[1], sys.argv[2]))
