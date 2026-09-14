#!/usr/bin/env python3
"""Screen every input key openPASO names against the backend that would consume it.

WHY THIS EXISTS
---------------
The 4C particle pass found this being served as knowledge:

    "SOUNDSPEED too low -> fluid compresses unrealistically;
     rule of thumb c >= 10 * v_max"

`SOUNDSPEED` appears in **zero** files of 4C's source and zero of its 2171
input decks. So does `SMOOTHING_LENGTH`. Both were listed beside real material
parameters (`DYNAMIC_VISCOSITY`, `BULK_MODULUS`) as things an SPH deck must set,
and one of them carried a numeric tuning rule. An agent following that advice
writes a deck 4C refuses to parse, and the rule of thumb is advice about nothing.

A footnote that is really the same lesson twice. THIS FILE used to write that
pair as `DYN_VISCOSITY`, and that spelling does not exist either: `4C -p` gives
MAT_ParticleSPHFluid the parameters INITRADIUS, INITDENSITY, REFDENSFAC,
EXPONENT, BACKGROUNDPRESSURE, BULK_MODULUS, DYNAMIC_VISCOSITY, BULK_VISCOSITY,
ARTIFICIAL_VISCOSITY, INITTEMPERATURE, THERMALCAPACITY — and `DYN_VISCOSITY`
appears in zero files of src/, apps/ and tests/. The gate's own documentation
was citing an invented key as the example of a real one, for the same reason the
knowledge did: it is a plausible abbreviation sitting next to a true name.

Nothing caught it. The quoted-diagnostics auditor screens error strings the
knowledge puts in quotes; an invented *input key* is not a quoted diagnostic, so
it walked straight through. This closes that hole: every ALL-CAPS identifier the
knowledge names is looked up in the backend's own corpus, and the ones that
resolve nowhere are reported.

WHAT IT REPORTS, AND WHAT IT REFUSES TO REPORT
----------------------------------------------
Candidates, not verdicts. Four exclusions keep it from crying wolf, each one
learned from a false accusation this project already made and had to withdraw:

  * **Retractions are not fabrications.** "There is no SOUNDSPEED key" is the
    CORRECTED entry. It contains the token precisely because the absence is the
    knowledge. Reusing `_is_retracted` from the quoted-diagnostics auditor rather
    than writing a second copy — two implementations of one rule is how the
    matcher defect happened.
  * **Prose is not a key.** `CFL`, `MPI`, `VTK`, `YAML` are English-in-caps.
    Screened by a stopword list plus a shape rule.
  * **A backend with no resolvable corpus yields UNKNOWN**, never "fabricated".
    Kratos was once judged against scipy because its own source was not on disk,
    and every claim looked invented.
  * **Compiled backends assemble key names at runtime**, so a miss in the source
    text is weaker evidence there than in a Python backend. Reported with that
    caveat attached rather than silently equated.

SELF-CONTROL
------------
`--selftest` runs the audit against a branch that predates the 4C fix and one
that follows it. The pre-fix tree must flag SOUNDSPEED; the post-fix tree must
not. A gate that cannot demonstrate it detects the thing it was built for is
just another unverified claim.
"""
from __future__ import annotations

import argparse
import ast
import pathlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from audit_quoted_diagnostics import (  # noqa: E402
    _is_retracted, collect_entries, search_roots, signal_of,
)

# ── what counts as a candidate key ──────────────────────────────────────────
# Shape: ALL CAPS, may carry digits and underscores. Four characters minimum —
# below that the false-positive rate from prose swamps the signal.
_KEY = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)*\b")

# English-in-caps, format names, physics abbreviations, units and the project's
# own category tags. None of these are input keys, and every one of them occurs
# in ordinary warning prose.
_STOPWORDS = {
    # category tags used by the warning format itself
    "NUMERICAL", "API", "INPUT", "SYNTAX", "PHYSICS", "INTEGRATION", "SETUP",
    "PERFORMANCE", "OUTPUT", "UNITS", "MESH", "VALIDATION", "BC", "CROSS",
    # formats, tools, protocols
    "XML", "YAML", "JSON", "HDF5", "VTK", "VTU", "VTP", "CSV", "TXT", "PVD",
    "MPI", "OPENMP", "CUDA", "GPU", "CPU", "RAM", "OS", "CLI", "API", "URL",
    # Symbols from MPI, libc and reference BLAS. They appear in warning text
    # because they are what the process dies inside, and they are identifier-
    # shaped, so shape alone cannot separate them from an input key. None of
    # them is a key of any backend here.
    "MPI_ABORT", "MPI_COMM_WORLD", "SIGABRT", "SIGFPE", "SIGSEGV", "SIGKILL",
    "DGEMM", "DGESV", "XERBLA",
    "UFL", "PETSC", "MUMPS", "UMFPACK", "SUPERLU", "BLAS", "LAPACK", "GMSH",
    # physics and numerics in caps
    "CFL", "PDE", "ODE", "FEM", "DEM", "SPH", "MPM", "PFEM", "DSMC", "DOF",
    "DOFS", "RHS", "LHS", "FSI", "TSI", "ALE", "DG", "HDG", "CG", "GMRES",
    "BDF", "RK", "MMS", "SI", "EOS", "PD", "VEM", "AMG", "ILU", "LU", "QR",
    "SVD", "PCG", "BICGSTAB", "NAN", "INF", "TOL", "ABS", "REL", "MIN", "MAX",
    # English words that appear capitalised for emphasis
    "NOT", "NO", "AND", "OR", "BUT", "ALL", "ANY", "ONLY", "MUST", "NEVER",
    "ALWAYS", "TODO", "NOTE", "WARNING", "ERROR", "FATAL", "OK", "YES",
    "TRUE", "FALSE", "NONE", "NULL", "ID", "IDS", "II", "III", "IV", "3D",
    "2D", "1D", "PASS", "FAIL", "SKIP", "THE", "IS", "IT", "IF", "ON", "OFF",
}

# ── ORDINARY ENGLISH, CAPITALISED FOR EMPHASIS ──────────────────────────────
# Kept as its own set, not merged into the block above, so the screen in
# tests/test_named_key_stopwords.py has something exact to hold to a hard rule.
#
# Running that screen over the ORIGINAL block finds 25 words in it that are real
# identifiers of some backend — ALE, CG, SPH, FSI, MUMPS, UMFPACK, GMRES and
# INTEGRATION are 4C names, BC / DOF / MIN / MAX / ERROR are FEBio ones, RHS is
# both a Kratos variable and a FEBio string. Most are enum VALUES or three-letter
# abbreviations rather than keys anyone would invent, which is presumably why
# they were admitted; the point is that nobody had measured it. That set is
# recorded and frozen by the test rather than quietly cleaned up here, because
# removing a stopword changes what the gate reports and belongs in its own pass.
# What the split buys is that the boundary is now visible: everything below has
# been screened, and the list cannot grow past the screen.
_PROSE_STOPWORDS = {
    # The corpus shouts. "it costs you the ENTIRE field", "TWO WIDESPREAD MYTHS
    # ABOUT THIS ENTRY", "WRONG: ... RIGHT: ...", "case-SENSITIVE" — 60 of the
    # 91 candidates in the first full run were words of this kind, and they
    # reached candidacy because some marker word (`parameter`, `set`, `field`,
    # `written`) happened to sit inside the 60-character window beside them.
    #
    # ADMISSION IS NOT BY EYE. A word may only be listed if it is not an input
    # key of ANY backend, and that was measured, not assumed, against three
    # key grammars read from the software itself:
    #
    #   4C      2026 single-token key and enum names, from `4C -p` — the
    #           binary's own dump of everything it accepts
    #   Kratos  2675 variable names, from the KRATOS_DEFINE_VARIABLE /
    #           KRATOS_CREATE_VARIABLE macro calls in the 48-application source
    #   FEBio   2437 registered strings, from ADD_PARAMETER / ADD_PROPERTY /
    #           REGISTER_FECORE_CLASS in febio-src, compared case-insensitively
    #
    # All 60 below are absent from all three, exactly and case-insensitively.
    # The screen is not vacuous: run on the words it REJECTED it rejects them
    # for cause. SUPG and BDF2 read exactly as English-in-caps in a FEniCSx and
    # a scikit-fem entry, and both are real 4C identifiers (`- name: SUPG`,
    # `- name: "BDF2"`), so neither is here; those two entries were reworded
    # instead. COUPLED and CURVED occur in 4C only inside multi-word section
    # titles ("DESIGN LINE SLIP SUPPLEMENTAL CURVED BOUNDARY CONDITIONS"),
    # never as a key, and are admitted. tests/test_named_key_stopwords.py
    # re-runs the screen so this list cannot quietly acquire a real key.
    "ABORTS", "ABOVE", "ACCURATE", "ALSO", "APPENDED", "BARE", "BEFORE",
    "BOTH", "CANNOT", "CAVEAT", "CHILD", "CHILDREN", "COLLAPSED",
    "CONNECTIVITY", "CORRECTIONS", "COUPLED", "CREATES", "CURVED", "DELETES",
    "DIRECTLY", "DISCARDED", "DRAINAGE", "EARLIER", "ENTIRE", "EVERY",
    "FALSIFIES", "FEEDBACK", "FREQUENCY", "HARD", "IDENTICAL", "INSENSITIVE",
    "ITSELF", "LATER", "LOWEST", "MAGNITUDE", "MEASURED", "MISSING", "MYTHS",
    "NEGATIVELY", "NONSENSE", "NOTHING", "OVERRIDDEN", "PERCENTAGE",
    "REFERENCED", "REFUTED", "REWRITES", "SAMPLE", "SECOND", "SENSITIVE",
    "STILL", "THAT", "THREE", "UNASSIGNED", "UNVERIFIED", "UPPERCASE",
    "VERIFIED", "WHERE", "WHICH", "WIDESPREAD", "WRONG",
    # Added 2026-08-10, same rule, same evidence. SILENT arrived with the 4C
    # signal work ("leaving the null space open is SILENT"): `4C -p` reports 0
    # occurrences and 4C's src/ 0, so it is not a 4C identifier.
    "SILENT",
}

# THIS REPOSITORY'S OWN PYTHON NAMES, which are not input keys of anything.
#
# Separate from the prose list because the reason differs: these are real
# identifiers, just not the backend's. `data/kratos_knowledge.py` exports
# per-application constants, and an entry that documents the import surface has
# to name them. Spelling them in lowercase prose to dodge the screen would make
# the entry less useful and less true — the point of that entry is that
# `from kratos_knowledge import KRATOS_KNOWLEDGE` FAILS, which cannot be said
# without writing the name.
#
# Admitted on the same evidence as above: neither is a Kratos variable
# (KratosGlobals.GetVariable raises for both), and neither is in 4C's grammar.
_OPENPASO_OWN_NAMES = {
    "KRATOS_KNOWLEDGE",   # the name that does NOT exist; the entry says so
    "GEOMECHANICS",       # a constant in data/kratos_knowledge.py, not Kratos
}
_STOPWORDS |= _OPENPASO_OWN_NAMES

_STOPWORDS |= _PROSE_STOPWORDS


# A key can be named in order to say it DOES NOT EXIST — which is the corrected
# form of exactly the entry this gate was built to catch:
#
#     "There is no SOUNDSPEED key. The material is MAT_ParticleSPHFluid ..."
#
# The shared `_is_retracted` does not recognise that phrasing, and widening the
# shared regex would silently change the quoted-diagnostics auditor's results on
# a corpus that has already been audited against it. So the absence test lives
# here and is anchored to the TOKEN rather than to the clause: the phrase has to
# be talking about this identifier, not merely sitting near it.
_ABSENCE_BEFORE = re.compile(
    r"(?:there\s+(?:is|are|was|were)\s+no"
    r"|no\s+such"
    r"|not\s+a\s+(?:valid|real|recognised|recognized)"
    r"|does\s+not\s+(?:exist|accept|have)"
    r"|never\s+(?:existed|accepted))"
    r"[^.;]{0,40}$", re.I)
_ABSENCE_AFTER = re.compile(
    r"^[^.;]{0,40}?(?:does\s+not\s+exist"
    r"|is\s+not\s+a\s+(?:valid|real|4c|febio|kratos)?\s*(?:key|section|option)"
    r"|is\s+no\s+such"
    r"|was\s+invented"
    r"|appears\s+in\s+(?:zero|no)\b)", re.I)

# A rejection IS an absence claim. "adding AREA0 to the material fails to match
# section 'MATERIALS'" and "writing OUTPUT_SCATRA under IO/RUNTIME VTK OUTPUT is
# a hard abort" are the CORRECTED form of exactly the fabrication this gate
# exists for, and both were being reported as fresh fabrications.
#
# Kept as its own regex, anchored at ^, rather than as one more alternative
# inside _ABSENCE_AFTER: that pattern opens with an unguarded `[^.;]{0,40}?`,
# and a rejection branch nested under it can step straight over the and/or
# guard below. The guard is what stops "`max_elems` is an EROSION parameter and
# is not accepted by hex_refine" from silencing EROSION, a token that sentence
# says nothing about — measured, not assumed: without it EROSION disappeared
# from febio's list.
_ABSENCE_AFTER_REJECT = re.compile(
    r"^(?:(?!\band\b|\bor\b|[,;.])[^.;]){0,32}?"
    r"(?:fails\s+to\s+match"
    r"|is\s+(?:a\s+)?(?:hard\s+)?(?:parse\s+)?abort"
    r"|is\s+rejected"
    r"|are\s+rejected"
    r"|is\s+not\s+accepted"
    r"|is\s+refused)", re.I)

# THE SOLVER'S OWN VERDICT, quoted verbatim. When a code answers a name with
# "is unknown" or "not defined", that is not an entry making a claim about the
# name — it is the software being asked and saying no, which is the strongest
# possible evidence the identifier does not exist and the best thing an entry
# can put in a Signal clause.
#
# The two that forced this are Kratos's, inside the CORRECTED PARTICLE_FRICTION
# entry. Its opening sentence ("PARTICLE_FRICTION is not a Kratos variable at
# all — it appears in zero files") the matcher already understood, while the two
# quoted proofs further down went on being reported as fresh fabrications:
#
#     'Error: Value type for "PARTICLE_FRICTION" not defined'
#     'Kernel.GetVariable() ERROR: Variable PARTICLE_FRICTION is unknown.'
#
# Anchored HARD — the denial must begin at the token, past nothing but a closing
# quote. No `[^.;]{0,40}?` run-up, so there is no window in which an unrelated
# clause can drift in and silence a neighbouring token, which is the failure the
# and/or guard above exists to prevent.
#
# Two deliberate narrowings, because "not defined" is the one phrase here that
# has an innocent reading:
#
#   * bare `not defined` fires only through a CLOSING QUOTE, so it has to be
#     inside a quoted message. "SOUNDSPEED not defined in the template" means
#     the deck omitted it, not that the key does not exist, and must stay
#     flaggable.
#   * `is not defined` is not accepted at all, for the same reason — "the
#     parameter is not defined by default" is about a value, not a name.
#
# Measured before narrowing: exactly five occurrences in the whole corpus depend
# on this rule, all Kratos, all inside quoted solver output, and all still
# silenced afterwards except INCLUDE_TRIANGLE — whose denial is a CMake message
# with no closing quote, and which resolves in the corpus anyway (6 files), so
# it is checked rather than excused and the verdict is identical.
_ABSENCE_AFTER_SOLVER = re.compile(
    r"^(?:[\"'\x60]\s*not\s+defined"
    r"|[\"'\x60]?\s*is\s+unknown"
    r"|[\"'\x60]?\s*is\s+not\s+(?:known|registered|recognised|recognized))",
    re.I)

# "... is NOT a SLIP_COEFF component of ...", "never takes THICKNESS or
# PLANE_ASSUMPTION", "writing OUTPUT_SCATRA ... is a hard abort" put the denial
# BEFORE the token, in shapes the general _ABSENCE_BEFORE does not cover.
#
# "the spelling" and "the earlier" have to sit IMMEDIATELY before the token.
# Loose, they silenced ngsolve's "the spelling the shipped IMEX template itself
# uses — DELETES ..." and dune's "the earlier 'cg cannot solve it' wording was
# REFUTED ...", neither of which says anything about the token beside it.
_ABSENCE_BEFORE_EXTRA = re.compile(
    r"(?:\bnot\s+a\b"
    r"|\bnever\s+takes\b"
    r"|\bnever\s+accepts\b"
    r"|\bno\s+\w+\s+called\b"
    # The solver's own "Unknown type 'X'" / "Unknown celltype X" is the
    # strongest statement possible that X does not exist, and quoting it is the
    # right way to record an absence.
    r"|\bunknown\s+celltype\b"
    r"|\bunknown\s+type\b"
    # CPython's own ImportError. "cannot import name KRATOS_KNOWLEDGE from
    # kratos_knowledge" is the interpreter reporting that the module does not
    # define that name — the same class of evidence as "Unknown type 'X'", and
    # anchored the same way, immediately before the token.
    r"|\bcannot\s+import\s+name\s+(?=$)"
    r"|\bthe\s+spelling\s+(?=$)"
    r"|\bthe\s+earlier\s+[\x60'\"]?(?=$)"
    r"|\bwriting\b"
    r"|\badding\b)"
    r"[^.;]{0,60}$", re.I)


def _absence_asserted(text: str, start: int, end: int) -> bool:
    """True when the surrounding prose says this identifier does NOT exist."""
    back = text[max(0, start - 120):start]
    if _ABSENCE_BEFORE.search(back) or _ABSENCE_BEFORE_EXTRA.search(back):
        return True
    fwd = text[end:end + 180]
    return bool(_ABSENCE_AFTER.search(fwd)
                or _ABSENCE_AFTER_REJECT.search(fwd)
                or _ABSENCE_AFTER_SOLVER.search(fwd))


# Words that mark the token beside them as an input key rather than emphasis.
_KEYISH = re.compile(
    r"\b(?:key|keys|parameter|parameters|section|sections|option|options|flag|"
    r"flags|field|fields|attribute|attributes|entry|variable|variables|set|"
    r"setting|specify|specifies|required|writes?|written)\b", re.I)


def _identifier_shaped(tok: str) -> bool:
    """An underscore or a digit is positive evidence of an identifier."""
    return "_" in tok or any(c.isdigit() for c in tok)


def _in_list_with_identifier(text: str, start: int, end: int) -> bool:
    """True when the token sits in a comma list containing a real identifier.

    This is the rule that catches the case shape alone cannot. The fabricated
    entry read

        "... for each particle type (density, DYN_VISCOSITY, BULK_MODULUS,
         SOUNDSPEED)"

    `SOUNDSPEED` is a plain all-caps word, exactly like the emphasis-caps this
    gate must ignore (`CHECKERBOARD`, `AUTOMATICALLY`). What distinguishes it is
    the company it keeps: enumerated alongside `DYN_VISCOSITY` and
    `BULK_MODULUS`, both underscore-shaped and one of them real. A word being
    listed as a peer of input keys is the claim that it is one — and note the
    rule works on SHAPE, not on truth. `DYN_VISCOSITY` is itself not a 4C
    parameter (the real spelling is `DYNAMIC_VISCOSITY`); it lent SOUNDSPEED
    credibility anyway, which is the whole mechanism being modelled here.
    """
    lo = text.rfind("(", max(0, start - 200), start)
    seg_start = lo + 1 if lo != -1 else max(0, start - 200)
    hi = text.find(")", end, end + 200)
    seg_end = hi if hi != -1 else min(len(text), end + 200)
    seg = text[seg_start:seg_end]
    if "," not in seg:
        return False
    peers = [p.strip(" '\"`") for p in seg.split(",")]
    peers = [p for p in peers if p != text[start:end]]
    return any(_KEY.fullmatch(p) and _identifier_shaped(p) and
               p not in _STOPWORDS for p in peers)


def candidate_keys(text: str) -> list[tuple[str, int, int]]:
    """ALL-CAPS tokens `text` presents AS input keys, with spans.

    Shape is not enough. `SOUNDSPEED` (invented) and `CHECKERBOARD` (ordinary
    prose in caps) are the same shape, so a shape rule either misses the
    fabrication or reports every emphasised word — the first pass over the
    corpus did the latter, returning 147 "unresolved keys" for Kratos of which
    the overwhelming majority were English. A gate nobody reads catches nothing.

    So a token has to earn candidacy by evidence that the text is naming a key:
    identifier shape, quoting, a key-marker word beside it, or membership in a
    list whose other members are confirmed identifiers.
    """
    out = []
    for m in _KEY.finditer(text):
        tok, i, j = m.group(0), m.start(), m.end()
        if tok in _STOPWORDS or len(tok) < 4:
            continue
        if text[max(0, i - 1):i] == "[":      # the warning's [Category] tag
            continue
        # `-DTRILINOS_APPLICATION=ON` is a CMake flag, not an input key, and the
        # token match starts one character late — at the D. That produced four
        # confident false candidates in one Kratos run
        # (DCONTACT_STRUCTURAL_MECHANICS_APPLICATION and friends), each of them
        # a real build flag reported as an invented key.
        if tok.startswith("D") and text[max(0, i - 1):i] == "-":
            continue
        quoted = (text[max(0, i - 1):i] in "'\"`"
                  or text[j:j + 1] in "'\"`")
        near = bool(_KEYISH.search(text[max(0, i - 60):i])
                    or _KEYISH.search(text[j:j + 60]))
        if not (_identifier_shaped(tok) or quoted or near
                or _in_list_with_identifier(text, i, j)):
            continue
        out.append((tok, i, j))
    return out


# An openPASO environment variable is not a backend input key. `FEBIO_BINARY` is
# underscore-shaped, so it is a candidate on shape alone, and the entry naming
# it says plainly what it is:
#
#     "Symlink the built binary to ~/FEBio/bin/febio4 or set FEBIO_BINARY so
#      check_availability() finds it."
#
# It is read at src/backends/febio/backend.py:96 as os.environ.get(
# "FEBIO_BINARY"). Looking for it in FEBio is looking in the wrong building —
# FEBio has never heard of it and never will, and the entry does not claim
# otherwise.
#
# WHY THIS CANNOT BE USED TO HIDE A FABRICATION. The set is not a hand-written
# list: it is read out of this repo's own AST, and a name qualifies only by
# appearing as the literal argument of os.environ[...] / os.environ.get(...) /
# os.getenv(...). An invented input key lives in a deck template or in prose —
# to reach this set someone would have to make openPASO read it from the
# environment, at which point it is an openPASO variable and this is true. 24
# names qualify today, every one of them plainly ours: FOURC_BINARY,
# KRATOS_ROOT, FEBIO_BINARY, SPARTA_BINARY, OFA_DISABLE_PITFALLS, LD_LIBRARY_PATH.
_ENV_VAR_CACHE: dict[str, set[str]] = {}


def openpaso_env_vars() -> set[str]:
    """ALL-CAPS names this repo reads out of the process environment."""
    if str(REPO) in _ENV_VAR_CACHE:
        return _ENV_VAR_CACHE[str(REPO)]
    out: set[str] = set()
    for py in sorted((REPO / "src").rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            lit = None
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Attribute)
                    and node.value.attr == "environ"
                    and isinstance(node.slice, ast.Constant)):
                lit = node.slice.value
            elif (isinstance(node, ast.Call) and node.args
                  and isinstance(node.args[0], ast.Constant)):
                fn = node.func
                name = getattr(fn, "attr", getattr(fn, "id", ""))
                base = getattr(getattr(fn, "value", None), "attr", "")
                if name == "getenv" or (name == "get" and base == "environ"):
                    lit = node.args[0].value
            if isinstance(lit, str) and _KEY.fullmatch(lit):
                out.add(lit)
    _ENV_VAR_CACHE[str(REPO)] = out
    return out


def key_present(key: str, roots: list[Path]) -> bool:
    """Does this identifier occur anywhere in the backend's corpus?

    `-a` is not optional: without it grep skips files it decides are binary and
    reports nothing found. That single flag was the difference between a
    measured 91% fabrication rate and the true 73% in an earlier pass.

    Returns (present, whole_word). The second value is the open defect this
    pass measured and did not fix, recorded rather than left to be rediscovered.

    A plain substring search answers YES for any key that is a SUBSTRING of a
    longer identifier, so a fabrication that happens to sit inside a real name
    is invisible. Measured, on the case that exposed it: dune-fem's implicit
    Runge-Kutta schemes are ImplicitEuler, CrankNicolson, DIRK23, DIRK34 and
    SDIRK22 (dune/fem/solver/rungekutta/timestepcontrol.hh:154). The knowledge
    offered `DIRK22`, which is not one of them — and substring search resolves
    DIRK22 against the letters of SDIRK22 and calls it real. DIRK22 was found
    by reading the header, not by this gate.

    `-w` fixes that (word-constituent characters are letters, digits and
    underscore, so DIRK22 no longer matches inside SDIRK22, and
    STRUCTURAL_MECHANICS no longer matches inside
    KRATOS_STRUCTURAL_MECHANICS_APPLICATION). It is NOT yet the pass/fail rule,
    because switching it on surfaces 63 further candidates across all nine
    backends including 4C, and shipping 63 untriaged reds — or parking them in
    the baseline — would be worse than shipping a named, measured limitation.
    They are listed by name in `substring_only` on every run, so the next pass
    starts from a list rather than from a rediscovery.
    """
    if not roots:
        return False, False
    paths = [str(r) for r in roots]
    for flags in (["-w"], []):
        cmd = ["grep", "-r", "-a", "-l", *flags, "-F", "--", key] + paths
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        except (subprocess.TimeoutExpired, OSError):
            return False, False
        if p.stdout.strip():
            return True, flags == ["-w"]
    return False, False


# A YAML/XML key sitting at the start of a line inside a template string. This
# is an unambiguous key POSITION — no context heuristic needed, and therefore
# almost no false positives.
_TEMPLATE_KEY = re.compile(r"^[ \t]*([A-Z][A-Z0-9_]{2,})[ \t]*:", re.M)


def template_keys(backend: str) -> list[tuple[Path, str]]:
    """Keys the deck TEMPLATES emit — the most dangerous surface in the corpus.

    `collect_entries` only returns strings containing "Signal:", i.e. warning
    text. Templates carry no Signal clause, so the whole set of keys openPASO
    actually WRITES INTO INPUT FILES was invisible to this audit. `AREA0` was
    caught only because it happened to appear in a warning as well as in the
    arterial-network template; a key that appeared solely in a template would
    have passed silently.

    Covers both shapes: keys emitted from a multi-line Python string, and keys
    written in a deck file that ships beside the backend.

    That is the same mistake `test_assembled_payload_is_clean` was written to
    correct one level up — an audit can be rigorous and still be pointed at the
    wrong surface. A wrong key in prose misleads a reader who can push back. A
    wrong key in a template is written into the deck verbatim, every run.
    """
    be_dir = REPO / "src" / "backends" / backend
    out: list[tuple[Path, str]] = []
    if not be_dir.is_dir():
        return out

    # Decks that ship as FILES, not as Python string literals. 4C's templates
    # moved out of .py into src/backends/fourc/decks/*.4C.yaml so the exact
    # bytes that were executed are the exact bytes that ship — which is right,
    # and which silently moved the single most dangerous surface in the corpus
    # out from under this gate. A key invented in a file is written into the
    # user's input deck exactly as surely as one invented in a string literal.
    for deck in sorted(be_dir.rglob("*.yaml")) + sorted(be_dir.rglob("*.yml")):
        text = deck.read_text(errors="ignore")
        for k in _TEMPLATE_KEY.findall(text):
            if k not in _STOPWORDS:
                out.append((deck, k))

    for py in sorted(be_dir.rglob("*.py")):
        try:
            tree = ast.parse(py.read_text(errors="ignore"))
        except (SyntaxError, OSError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant)
                    and isinstance(node.value, str)):
                continue
            text = node.value
            # A template is multi-line and has several key-looking lines; a
            # sentence that merely starts with a capitalised word does not.
            if text.count("\n") < 2:
                continue
            hits = _TEMPLATE_KEY.findall(text)
            if len(hits) < 2:
                continue
            for k in hits:
                if k not in _STOPWORDS:
                    out.append((py, k))
    return out


def audit(backend: str, verbose: bool = False) -> dict:
    src_roots, corpus_roots, notes = search_roots(backend)
    roots = list(src_roots) + list(corpus_roots)
    res = {"backend": backend, "roots": [str(r) for r in roots],
           "notes": notes, "unresolved": [], "checked": 0, "entries": 0}
    if not roots:
        res["status"] = "UNKNOWN"
        res["reason"] = (f"no corpus on disk for {backend}; a missing corpus "
                         f"cannot distinguish an invented key from a real one")
        return res

    # THE RULE THIS PROJECT KEEPS RELEARNING. `search_roots` falls back to
    # whatever scientific-Python packages are installed when the backend's own
    # module will not import. Judging Kratos knowledge against scipy makes every
    # real Kratos variable look invented: the first run of this gate reported
    # 121 of 139 Kratos keys "unresolved", and the roots it actually searched
    # were scipy, numpy, meshio and mpi4py. Not one Kratos file.
    #
    # An audit that cannot see the software it is auditing has no verdict to
    # give. It says so, and names the interpreter that would fix it.
    unresolved_primary = [n for n in (notes or [])
                          if "not importable" in str(n)]
    if unresolved_primary:
        res["status"] = "UNKNOWN"
        res["reason"] = (
            f"{unresolved_primary[0]} — the roots on hand "
            f"({', '.join(Path(p).name for p in res['roots'][:4])}) are not "
            f"{backend}. Re-run with an interpreter where {backend} imports; "
            f"a fallback corpus can only manufacture false accusations.")
        return res

    seen: dict[str, list] = {}
    for path, entry in collect_entries(backend):
        res["entries"] += 1
        for tok, i, j in candidate_keys(entry):
            if _is_retracted(entry, i, j) or _absence_asserted(entry, i, j):
                continue          # "there is no SOUNDSPEED key" is the fix
            seen.setdefault(tok, []).append((str(path), signal_of(entry)[:90]))

    # Template keys are added with their own marker: they are the surface that
    # gets written into a deck, so an unresolved one is strictly more serious
    # than an unresolved one in prose.
    for path, tok in template_keys(backend):
        seen.setdefault(tok, []).append((str(path), "[DECK TEMPLATE]"))

    ours = openpaso_env_vars()
    res["substring_only"] = []
    for tok in sorted(seen):
        res["checked"] += 1
        if tok in ours:
            continue          # our own environment variable, see openpaso_env_vars
        present, whole_word = key_present(tok, roots)
        if not present:
            res["unresolved"].append({
                "key": tok,
                "occurrences": len(seen[tok]),
                "first_seen": seen[tok][0][0],
                "signal": seen[tok][0][1],
            })
        elif not whole_word:
            # Resolved only inside a longer identifier. Weaker evidence than a
            # whole-word hit and the gate's known blind spot — see key_present.
            res["substring_only"].append(tok)
    if res["entries"] == 0:
        # "OK, 0 keys checked, 0 unresolved" is the shape of a pass, and it is
        # what this returned for SPARTA — whose knowledge in this tree is a
        # 509 KB command reference containing no warnings at all. A green tick
        # for a backend nobody looked at is the most expensive kind of wrong,
        # because a pass ends the investigation.
        res["status"] = "NO_ENTRIES"
        res["reason"] = (
            f"no warning text found for {backend}: nothing under "
            f"src/backends/{backend} carries a 'Signal:' clause, in Python or "
            f"in JSON. Either this backend has no warnings yet, or its "
            f"knowledge is stored in a shape the collector does not read. "
            f"Both are worth knowing; neither is a pass.")
        return res

    part = corpus_completeness(backend)
    if part:
        res["corpus_partial"] = part
        res["status"] = "PARTIAL_CORPUS"
        # "Not found" means "not found in what is installed". Kratos ships ~40
        # applications; this host has three. BIOT_COEFFICIENT lives in
        # Poromechanics and DEM_SURFACE_LOAD in DEM, so neither can be resolved
        # here — and neither is thereby shown to be invented. Calling them
        # fabrications would repeat the scipy mistake one level down.
        res["unverifiable"] = res.pop("unresolved")
        return res
    res["status"] = "OK" if not res["unresolved"] else "CANDIDATES"
    return res


def corpus_completeness(backend: str) -> str:
    """Describe a corpus known to be a SUBSET of the backend, else ''.

    Only Kratos needs this today: it is distributed as a core plus optional
    application packages, each a separate wheel, and a key belonging to an
    application that is not installed is invisible no matter how real it is.
    """
    if backend != "kratos":
        return ""
    import glob
    # The full source build, if present, makes the caveat unnecessary: 28
    # applications is the whole of Kratos as this project uses it, so a key that
    # does not resolve there really does not resolve.
    full = glob.glob("/mnt/kratos-tier2/kv/lib/python*/site-packages/"
                     "KratosMultiphysics")
    if full:
        n = len([d for d in Path(full[0]).iterdir()
                 if d.is_dir() and d.name.lower().endswith("application")])
        if n >= 20:
            return ""
    sp = glob.glob("/home/alexander/Schreibtisch/open-fem-agent/.venv/lib/"
                   "python*/site-packages")
    if not sp:
        return ""
    apps = sorted({Path(p).name.split("-")[0]
                   for p in glob.glob(f"{sp[0]}/kratos*application-*.dist-info")})
    if not apps:
        return ""
    return (f"only {len(apps)} Kratos applications installed "
            f"({', '.join(a.replace('kratos', '').replace('application', '') for a in apps)}"
            f" + core); keys belonging to any other application "
            f"(DEM, Poromechanics, CFD, ...) cannot be resolved on this host")


def selftest() -> int:
    """Prove the gate detects what it was built to detect.

    Pre-fix tree must flag SOUNDSPEED; post-fix tree must stay quiet about it.
    """
    pre = Path("/home/alexander/Schreibtisch/ofa-verify-ngs")
    post = Path("/home/alexander/Schreibtisch/ofa-know-4c")
    print("SELFTEST — the gate must flag the known fabrication and only it\n")
    ok = True
    for label, tree, expect in (("pre-fix ", pre, True), ("post-fix", post, False)):
        f = tree / "src" / "backends" / "fourc" / "backend.py"
        if not f.is_file():
            print(f"  {label}: {f} missing — cannot run control")
            ok = False
            continue
        text = f.read_text(errors="ignore")
        hits = [t for t, i, j in candidate_keys(text)
                if t == "SOUNDSPEED" and not _is_retracted(text, i, j)
                and not _absence_asserted(text, i, j)]
        got = bool(hits)
        good = got == expect
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {label} tree: SOUNDSPEED "
              f"{'flagged' if got else 'not flagged'} "
              f"(expected {'flagged' if expect else 'not flagged'})")
    print(f"\nselftest {'PASSED' if ok else 'FAILED'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("backends", nargs="*")
    ap.add_argument("--repo", default=None, metavar="DIR",
                    help="audit ANOTHER checkout's knowledge. The campaign "
                         "branches hold the current corpus; this worktree is "
                         "stale for several backends (its SPARTA is the old "
                         "13-pitfall version against 211 on the consolidation "
                         "branch), so auditing only what is under this script "
                         "measures the wrong tree.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.repo:
        import audit_quoted_diagnostics as aqd
        global REPO
        REPO = aqd.REPO = pathlib.Path(args.repo).resolve()

    if args.selftest:
        return selftest()

    names = args.backends or ["fourc", "fenics", "dealii", "ngsolve", "skfem",
                              "kratos", "dune", "febio", "sparta"]
    out = []
    for b in names:
        r = audit(b)
        out.append(r)
        head = f"{b:<10} {r['status']:<11}"
        if r["status"] == "UNKNOWN":
            print(f"  {head} {r['reason']}")
            continue
        # "unresolved" and "unverifiable" are deliberately different words and
        # must never be printed under one heading: the first says the corpus was
        # searched and the key is not in it, the second says the corpus cannot
        # answer. Collapsing them is how a coverage gap gets read as a fabrication.
        hits = r.get("unresolved")
        label = "unresolved"
        if hits is None:
            hits, label = r.get("unverifiable", []), "UNVERIFIABLE here"
        print(f"  {head} {r['checked']:>4} distinct keys checked across "
              f"{r['entries']} entries -> {len(hits)} {label}")
        if r.get("corpus_partial"):
            print(f"        corpus is partial: {r['corpus_partial']}")
        for u in hits[:8]:
            print(f"        {u['key']:<28} x{u['occurrences']:<3} {u['signal']}")
        # Never folded into the count above. These RESOLVED, so they are not
        # accusations; they resolved only inside a longer identifier, which is
        # the weaker of the two answers and the gate's known blind spot.
        sub = r.get("substring_only") or []
        if sub:
            print(f"        substring-only ({len(sub)}, resolve inside a "
                  f"longer identifier -- next triage batch, see key_present): "
                  + " ".join(sub))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\n  written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
