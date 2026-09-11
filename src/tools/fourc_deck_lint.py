"""OASiS-side 4C deck lint: names the defects of a deck the AGENT wrote, from the deck text and 4C's
own console log. A verification gate, not a generator: it never writes or completes a deck.

The same defect classes ride inside the served coupling contracts as text the participant runs at
exit (why_4c_did_not_finish); this copy lets the ladder read a side directory and put 4C's own
error line and the named defects into the next step's brief, so a worker that must repair a deck
starts from the defect and not from the whole job. Every class here was measured on a worker deck.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
from pathlib import Path

_TOPO_WORDS = ("DNODE", "DLINE", "DSURFACE", "DVOL")
_VALID_CACHE: dict[str, set] = {}


def valid_sections(bin_path, ld: str | None = None) -> set:
    """Section names the INSTALLED binary accepts, from its own grammar dump (`4C -p`): the
    `sections:` names and the legacy string sections (elements, coordinates, topology). Cached
    per binary; empty when the binary is not there (then no section is judged)."""
    key = str(bin_path or "")
    if key in _VALID_CACHE:
        return _VALID_CACHE[key]
    valid: set = set()
    try:
        if key and Path(key).is_file():
            env = dict(os.environ)
            if ld:
                env["LD_LIBRARY_PATH"] = f"{ld}:{env.get('LD_LIBRARY_PATH', '')}"
            dump = subprocess.run([key, "-p"], capture_output=True, text=True, timeout=180, env=env).stdout
            valid = set(re.findall(r"^    - name: (.+?)\s*$", dump, re.M)) | set(
                re.findall(r"^  - ([A-Z][A-Z0-9 _/.:-]*?)\s*$", dump.split("legacy_string_sections:", 1)[-1], re.M))
            valid |= {"TITLE"}
    except Exception:                                   # noqa: BLE001
        valid = set()
    _VALID_CACHE[key] = valid
    return valid


def unknown_sections(text: str, valid: set) -> list[str]:
    """Deck sections the binary's grammar does not know, each with the closest names it does know.
    Measured: the dominant worker failure is a section name invented by analogy
    ('IO/RUNTIME VTK OUTPUT/THERMO', 'SOLIDSCATRA ELEMENTS'), and 4C stops at the first one."""
    if len(valid) < 100:
        return []
    out = []
    names = sorted(valid)
    for s in dict.fromkeys(re.findall(r"^([A-Z][A-Z0-9 _/.:-]*?):\s*$", text, re.M)):
        if s in valid or re.fullmatch(r"FUNCT\d+", s):
            continue
        close = closest_sections(s, names)
        out.append(f"section '{s}' is not in the binary's grammar (`4C -p`)"
                   + (f"; closest known: {', '.join(repr(c) for c in close)}" if close else ""))
    return out


def _tokens(name: str) -> list[str]:
    return [w for w in re.split(r"[ /_-]+", name.upper()) if w]


def closest_sections(unknown: str, names: list, n: int = 5) -> list[str]:
    """Known names ranked by symmetric word similarity: every word of the unknown name against its best
    match in the candidate and back, so 'IO/RUNTIME VTK OUTPUT/THERMO' lists THERMAL DYNAMIC/RUNTIME VTK
    OUTPUT and 'SOLIDSCATRA ELEMENTS' lists STRUCTURE ELEMENTS first (character ratios alone ranked them
    sixth and fourth, measured on the installed grammar)."""
    u = _tokens(unknown)
    if not u:
        return []
    def best(a, bs):
        return max((difflib.SequenceMatcher(None, a, b).ratio() for b in bs), default=0.0)
    scored = []
    for c in names:
        ct = _tokens(c)
        if not ct:
            continue
        sc = (sum(best(a, ct) for a in u) + sum(best(b, u) for b in ct)) / (len(u) + len(ct))
        scored.append((sc, c))
    scored.sort(key=lambda p: (-p[0], p[1]))
    return [c for sc, c in scored[:n] if sc >= 0.45]


def fourc_error_lines(log_text: str, n: int = 8) -> str:
    """4C's own error block after 'PROC 0 ERROR' (the first n non-empty lines), or ''."""
    lines = log_text.splitlines()
    for i, ln in enumerate(lines):
        if "PROC 0 ERROR" in ln:
            said = [l.strip() for l in lines[i + 1:i + 14]
                    if l.strip() and not l.startswith("---") and "MPI_ABORT" not in l]
            return " | ".join(said[:n])
    return ""


def python_stop_lines(log_text: str) -> str:
    """The last lines of a Python traceback in a captured console (the participant's own stop), or ''."""
    i = log_text.rfind("Traceback (most recent call last)")
    if i < 0:
        return ""
    tail = [l.rstrip() for l in log_text[i:].splitlines()[1:] if l.strip()]
    return " | ".join(tail[-3:])


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
    topo = set(re.findall(r"\b(DNODE|DLINE|DSURFACE|DVOL)\s+(\d+)", text))
    for b in re.split(r"^(?=[A-Z][A-Z0-9 _/.:-]*?:\s*$)", text, flags=re.M):
        head = b.split(":", 1)[0].strip()
        if head.startswith("DESIGN") and head.endswith("CONDITIONS"):
            kw = re.search(r"\b(POINT|LINE|SURF|VOL)\b", head)
            kind = {"POINT": "DNODE", "LINE": "DLINE", "SURF": "DSURFACE", "VOL": "DVOL"}[kw.group(1)] if kw else ""
            entries = [e for e in re.split(r"^\s*-\s", b, flags=re.M)[1:] if e.strip()]
            noid = [e for e in entries if not re.search(r"\bE:\s*\d+|NODE_SET_NAME", e)]
            if noid:
                why.append(f"{len(noid)} entr{'y' if len(noid) == 1 else 'ies'} in {head} without `E: <id>`")
            missing = sorted({x for x in re.findall(r"\bE:\s*(\d+)", b) if (kind, x) not in topo}, key=int)
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
    errors, tracebacks = {}, {}
    for lg in sorted(side.glob("*.log")) + sorted(side.glob("*.txt")):
        try:
            txt = lg.read_text(errors="ignore")
        except OSError:
            continue
        said = fourc_error_lines(txt)
        if said:
            errors[lg.name] = said
        tb = python_stop_lines(txt)
        if tb:
            tracebacks[lg.name] = tb
    defects = {}
    decks = sorted(side.glob("*.4C.yaml")) or [p for p in sorted(side.glob("*.yaml")) if "monitor_dbc" not in p.name]
    # the binary the agent's own config names (or the environment's): its grammar judges the section names
    cfg = {}
    try:
        cfg = json.loads((side / "config.json").read_text() or "{}") if (side / "config.json").is_file() else {}
    except Exception:                                   # noqa: BLE001
        cfg = {}
    valid = valid_sections(cfg.get("fourc_bin") or os.environ.get("FOURC_BIN"), cfg.get("fourc_ld") or os.environ.get("FOURC_LD"))
    for dk in decks:
        try:
            txt = dk.read_text(errors="ignore")
        except OSError:
            continue
        why = lint_deck(txt) + unknown_sections(txt, valid)
        if why:
            defects[dk.name] = why
    return {"finished": finished, "monitors": monitors, "errors": errors, "defects": defects,
            "tracebacks": tracebacks, "decks": [d.name for d in decks]}
