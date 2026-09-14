"""The stopword list may not contain a word that is a real input key.

`audit_named_input_keys` skips ALL-CAPS tokens on a stopword list, because the
corpus shouts: "it costs you the ENTIRE field", "TWO WIDESPREAD MYTHS", "WRONG:
... RIGHT: ...". Sixty of the ninety-one candidates in the first full run were
English words of that kind.

A stopword list is also the easiest way to switch this gate off. Put SOUNDSPEED
on it and the fabrication it was built for walks through, silently and for good,
and nothing else in the suite would notice. So the list is not a matter of
judgement here: a word may be on it only if it is not an input key of any
backend, and that is checked against the backends' OWN key grammars, read out of
the software rather than remembered.

    4C      every single-token `name:` value in `4C -p`, the binary's dump of
            everything it accepts (2026 names)
    Kratos  every name defined by a KRATOS_DEFINE_VARIABLE / KRATOS_CREATE_
            VARIABLE macro in the C++ source (2675)
    FEBio   every string registered through ADD_PARAMETER / ADD_PROPERTY /
            REGISTER_FECORE_CLASS (2437), compared case-insensitively, since
            FEBio's tags are lower-case and the prose shouts them

The screen is not vacuous — it rejects SUPG and BDF2, which read exactly like
English-in-caps in a FEniCSx and a scikit-fem entry and are both real 4C
identifiers (`- name: SUPG`, `- name: "BDF2"`). Those two entries were reworded
instead of being silenced, and the words are not on the list.

Skips, never passes, when a corpus is missing: an empty key grammar would clear
every stopword, which is the "instrument that cannot look" failure this project
keeps paying for. Each source is verified with a positive control first.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
import backend_probe  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

FOURC = Path(os.environ.get("FOURC_BINARY", str(backend_probe.fourc_binary())))
# The SOURCE directories, not the checkout root: `.git` is compressed objects
# that a text grep would read as noise, and a junk token scraped out of a
# packfile would land in the "real key" set and reject a legitimate stopword.
KRATOS_SRC = backend_probe.kratos_root() / "kratos"
KRATOS_APPS = backend_probe.kratos_root() / "applications"
FEBIO_SRC = backend_probe.febio_src()

# A name whose presence proves the extractor worked. If the control is missing,
# the grammar was not really read and the result is not evidence of anything.
_CONTROLS = {"fourc": "SUPG", "kratos": "DISPLACEMENT", "febio": "EROSION"}


def _fourc_keys() -> set[str]:
    if not FOURC.is_file():
        return set()
    env = dict(os.environ)
    env.setdefault("LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")
    try:
        p = subprocess.run([str(FOURC), "-p"], capture_output=True, text=True,
                           timeout=600, env=env, stdin=subprocess.DEVNULL)
    except (subprocess.TimeoutExpired, OSError):
        return set()
    return {m.group(1) for m in
            re.finditer(r'^\s*-?\s*name:\s*"?([A-Z][A-Z0-9_]*)"?\s*$',
                        p.stdout, re.M)}


def _grep(pattern: str, *roots: Path) -> str:
    live = [str(r) for r in roots if r.is_dir()]
    if not live:
        return ""
    try:
        p = subprocess.run(["grep", "-rhoE", pattern, *live],
                           capture_output=True, text=True, timeout=900)
    except (subprocess.TimeoutExpired, OSError):
        return ""
    return p.stdout


def _kratos_vars() -> set[str]:
    out = set()
    text = _grep(r"KRATOS_(DEFINE|CREATE)[A-Z_0-9]*VARIABLE[A-Z_0-9]*\([^)]*\)",
                 KRATOS_SRC, KRATOS_APPS)
    for line in text.splitlines():
        for arg in line[line.find("(") + 1:].rstrip(")").split(","):
            arg = arg.strip()
            if re.fullmatch(r"[A-Z][A-Z0-9_]{2,}", arg):
                out.add(arg)
    return out


def _febio_keys() -> set[str]:
    text = _grep(r'(ADD_PARAMETER|ADD_PROPERTY|REGISTER_FECORE_CLASS)'
                 r'\([^;]*"[^"]+"', FEBIO_SRC)
    return {s.upper() for s in re.findall(r'"([^"]+)"', text)}


# The screen run over the ORIGINAL stopword block, before this pass added the
# prose words. Every one is a short abbreviation, an enum VALUE or a category
# tag rather than a key anyone would invent — YES / NO / NONE / MIN / MAX are
# 4C and FEBio enum values, ALE / CG / SPH / FSI / DEM / HDG / GMRES / MUMPS /
# UMFPACK / ILU / CSV / CROSS / INTEGRATION are 4C names, BC / DOF / DOFS /
# ERROR are FEBio ones, RHS is a Kratos variable and a FEBio string. They were
# admitted before anybody measured, and they are frozen here rather than
# removed: dropping a stopword changes what the gate reports and is its own
# pass. The point of recording them is that the number may not grow.
_LEGACY_OVERLAP = {
    "ALE", "BC", "CG", "CROSS", "CSV", "DEM", "DOF", "DOFS", "ERROR", "FSI",
    "GMRES", "HDG", "ID", "ILU", "INTEGRATION", "MAX", "MIN", "MUMPS", "NO",
    "NONE", "RHS", "SPH", "TOL", "UMFPACK", "YES",
}


def _usable_grammars():
    grammars = {"fourc": _fourc_keys(), "kratos": _kratos_vars(),
                "febio": _febio_keys()}
    usable, unusable = {}, []
    for name, keys in grammars.items():
        if _CONTROLS[name] in keys:
            usable[name] = keys
        else:
            unusable.append(f"{name} ({len(keys)} names read, positive control "
                            f"{_CONTROLS[name]} absent)")
    if not usable:
        pytest.skip("no backend key grammar could be read here, so a clean "
                    "result would mean nothing: " + "; ".join(unusable))
    return usable, unusable


def test_no_prose_stopword_is_a_real_input_key() -> None:
    """The words added by the prose triage: hard rule, no exceptions."""
    from audit_named_input_keys import _PROSE_STOPWORDS

    usable, unusable = _usable_grammars()
    offenders = sorted((w, be) for w in _PROSE_STOPWORDS
                       for be, keys in usable.items() if w in keys)
    assert not offenders, (
        "these stopwords are real input keys and must come off the list — a "
        "key on it can never be reported, however it is fabricated:\n"
        + "\n".join(f"    {w}  is a real {be} key" for w, be in offenders)
        + "\n\nIf the word genuinely appears as English in some entry, reword "
          "the ENTRY. SUPG and BDF2 were handled that way."
        + (f"\n\n(checked against {', '.join(usable)}; "
           f"not checked: {'; '.join(unusable)})" if unusable else ""))


def test_legacy_stopword_overlap_does_not_grow() -> None:
    """The pre-existing list holds 25 real names. It may shrink, never grow."""
    from audit_named_input_keys import _PROSE_STOPWORDS, _STOPWORDS

    usable, _ = _usable_grammars()
    legacy = _STOPWORDS - _PROSE_STOPWORDS
    found = {w for w in legacy for keys in usable.values() if w in keys}
    new = sorted(found - _LEGACY_OVERLAP)
    assert not new, (
        "these words were added to the stopword list and are real input "
        "keys:\n    " + "\n    ".join(new)
        + "\n\nPut new prose words in _PROSE_STOPWORDS, which is screened "
          "against the backends' key grammars, rather than in the block above "
          "it. A word that is a real key belongs in neither.")
    gone = sorted(_LEGACY_OVERLAP - found)
    if gone:
        print(f"\n{len(gone)} legacy overlaps have been resolved and can be "
              f"dropped from _LEGACY_OVERLAP: {' '.join(gone)}")
