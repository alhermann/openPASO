"""OASiS-side 4C deck lint: names the defects of a deck the AGENT wrote, from the deck text and 4C's
own console log. A verification gate, not a generator: it never writes or completes a deck.

The same defect classes ride inside the served coupling contracts as text the participant runs at
exit (why_4c_did_not_finish); this copy lets the ladder read a side directory and put 4C's own
error line and the named defects into the next step's brief, so a worker that must repair a deck
starts from the defect and not from the whole job. Every class here was measured on a worker deck.
"""
from __future__ import annotations

import re
from pathlib import Path

_TOPO_WORDS = ("DNODE", "DLINE", "DSURFACE", "DVOL")


def fourc_error_lines(log_text: str, n: int = 8) -> str:
    """4C's own error block after 'PROC 0 ERROR' (the first n non-empty lines), or ''."""
    lines = log_text.splitlines()
    for i, ln in enumerate(lines):
        if "PROC 0 ERROR" in ln:
            said = [l.strip() for l in lines[i + 1:i + 14]
                    if l.strip() and not l.startswith("---") and "MPI_ABORT" not in l]
            return " | ".join(said[:n])
    return ""


def lint_deck(text: str) -> list[str]:
    """Deck defects measured on worker decks (each one made 4C stop or solve the wrong problem)."""
    why: list[str] = []
    secs = re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", text, re.M)
    dup = sorted({x for x in secs if secs.count(x) > 1})
    if dup:
        why.append("section(s) written twice: " + ", ".join(dup))
    if "Thermo_Structure_Interaction" in text:
        if "CLONING MATERIAL MAP" not in text:
            why.append("TSI needs a CLONING MATERIAL MAP pairing the structure material with the MAT_Fourier thermal material")
        if "COUPVARIABLE" not in text or "Temperature" not in text.split("COUPVARIABLE", 1)[-1][:40]:
            why.append('TSI DYNAMIC/PARTITIONED needs COUPVARIABLE "Temperature" (the default gives zero thermal strain)')
        if re.search(r"\b(WALL|SOLID) QUAD4\b|\bTRI3\b", text):
            why.append("4C has no 2-D thermo-elastic element; the route is a one-element-thick SOLIDSCATRA HEX8 slab")
        if "monitor_reaction" in text and "IO/MONITOR STRUCTURE DBC" not in text:
            why.append("TAG: monitor_reaction writes nothing without an IO/MONITOR STRUCTURE DBC section")
        if "DESIGN VOL THERMO DIRICH" in text:
            why.append("DESIGN VOL THERMO DIRICH imposes the temperature volume-wide (no heat equation is solved); "
                       "use SURF (outer) and POINT (interface) THERMO DIRICH")
    if "Scalar_Transport" in text:
        if "THERMAL DYNAMIC:" in text and "SCALAR TRANSPORT DYNAMIC:" not in text:
            why.append("Scalar_Transport needs `SCALAR TRANSPORT DYNAMIC`, not `THERMAL DYNAMIC`")
        if "CALCFLUX_BOUNDARY" not in text or "FLUX CALC" not in text:
            why.append('a consistent boundary flux needs CALCFLUX_BOUNDARY "diffusive" AND a `SCATRA FLUX CALC LINE CONDITIONS` entry on the interface line')
        if re.search(r"^IO:\s*$", text, re.M):
            why.append("an `IO:` section in a Scalar_Transport deck is rejected; the VTU appears without it")
    if re.search(r'PROBLEMTYPE:\s*"?Thermo"?\s*$', text, re.M):
        why.append("PROBLEMTYPE Thermo writes no scatra flux output and knows no CALCFLUX_BOUNDARY; a consistent heat flux comes from Scalar_Transport")
    badkw = sorted({w for w in re.findall(r'"NODE\s+\d+\s+(D[A-Z]+)\s+\d+"', text) if w not in _TOPO_WORDS})
    if badkw:
        why.append(f"topology entries use {', '.join(badkw)} -- the entity words are DNODE, DLINE, DSURFACE, DVOL "
                   "(anything else defines nothing and 4C silently drops the conditions on it)")
    topo = set(re.findall(r"D(?:NODE|LINE|SURF|VOL)\s+(\d+)", text))
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if head.startswith("DESIGN") and head.endswith("CONDITIONS"):
            entries = [e for e in re.split(r"^\s*-\s", b, flags=re.M)[1:] if e.strip()]
            noid = [e for e in entries if not re.search(r"\bE:\s*\d+|NODE_SET_NAME", e)]
            if noid:
                why.append(f"{len(noid)} entr{'y' if len(noid) == 1 else 'ies'} in {head} without `E: <id>`")
            missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", b) if x not in topo}, key=int)
            if missing:
                why.append(f"{head} names E id(s) {', '.join(missing[:6])} that no *-NODE TOPOLOGY section defines")
    if re.search(r"FUNCT\d+:", text) and re.search(r"\bFUNCT:\s*\[\s*0(\s*,\s*0)*\s*\]", text) \
            and not re.search(r"\bFUNCT:\s*\[[^\]]*[1-9]", text):
        why.append("FUNCT blocks are defined but no condition references one (FUNCT: [0,...] everywhere): the sources never reach the load")
    return why


def side_dir_report(side: Path) -> dict:
    """What 4C left behind in a participant's directory: which runs finished (their VTU folders),
    4C's own error lines per log, and the named defects per deck. Reads only the agent's own files."""
    side = Path(side)
    vtk_dirs = sorted(p for p in side.glob("*-vtk-files") if p.is_dir())
    finished = {}
    for d in vtk_dirs:
        kinds = sorted({re.sub(r"-\d+-\d+\.vtu$", "", q.name) for q in d.glob("*.vtu")})
        if kinds:
            finished[d.name[:-len("-vtk-files")]] = kinds
    monitors = sorted(p.name for p in side.glob("*_monitor_dbc.yaml"))
    errors = {}
    for lg in sorted(side.glob("*.log")) + sorted(side.glob("*.txt")):
        try:
            said = fourc_error_lines(lg.read_text(errors="ignore"))
        except OSError:
            continue
        if said:
            errors[lg.name] = said
    defects = {}
    decks = sorted(side.glob("*.4C.yaml")) or [p for p in sorted(side.glob("*.yaml")) if "monitor_dbc" not in p.name]
    for dk in decks:
        try:
            why = lint_deck(dk.read_text(errors="ignore"))
        except OSError:
            continue
        if why:
            defects[dk.name] = why
    return {"finished": finished, "monitors": monitors, "errors": errors, "defects": defects,
            "decks": [d.name for d in decks]}
