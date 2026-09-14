"""Inject fabrications and require the gate to flag them.

`test_named_input_keys_exist` reports zero unresolved keys for all nine
backends. Zero is also what a blind gate reports, and the two are
indistinguishable from the number alone — so this file supplies the other half
of the measurement: known-bad keys that must FAIL, and known-good ones that must
PASS, run through the audit's own decision path rather than a paraphrase of it.

The bad cases are not invented for the test. Every one is a fabrication this
project actually shipped and had to withdraw:

    SOUNDSPEED, SMOOTHING_LENGTH   served as required 4C SPH material
                                   parameters; in zero files of 4C
    AREA0                          written into the arterial-network deck
                                   template; MAT_CNST_ART uses DIAM
    FLUID_DYNAMIC                  4C's section is 'FLUID DYNAMIC', with a
                                   space
    PARTICLE_FRICTION              served as a required Kratos DEM material
                                   key; the real ones are STATIC_FRICTION /
                                   DYNAMIC_FRICTION
    FREESTREAM_VELOCITY            the real name is FREE_STREAM_VELOCITY
    DYN_VISCOSITY                  4C's parameter is DYNAMIC_VISCOSITY; this
                                   spelling was cited as REAL in the auditor's
                                   own docstring until 2026-08-09

Each is presented the way knowledge presents a key — in a list of peers, or
beside a marker word — because a fabrication that is never a candidate is never
checked, and candidacy is where this gate is most easily blunted.

The good cases guard the opposite failure. Widening an absence cue or a stopword
to clear a false positive is cheap and it is how a gate stops working; a
tightening that also silences SDIRK22, KSP_DIVERGED_PC_FAILED or STATIC_FRICTION
shows up here instead of in a user's deck.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

# (backend, token, entry, must_flag, why)
CASES = [
    ("fourc", "SOUNDSPEED",
     "[Input] Set the SPH material parameters for each particle type "
     "(density, DYNAMIC_VISCOSITY, BULK_MODULUS, SOUNDSPEED). Signal: x.",
     True, "the original fabrication, in its original list shape"),
    ("fourc", "SMOOTHING_LENGTH",
     "[Input] The SMOOTHING_LENGTH parameter controls the kernel support. "
     "Signal: x.",
     True, "the second original fabrication"),
    ("fourc", "AREA0",
     "[Input] MAT_CNST_ART takes the AREA0 parameter for the reference "
     "cross-section. Signal: x.",
     True, "the arterial-network deck-template fabrication"),
    ("fourc", "FLUID_DYNAMIC",
     "[Input] Write the FLUID_DYNAMIC section. Signal: x.",
     True, "underscore-for-space section-name fabrication"),
    ("fourc", "DYN_VISCOSITY",
     "[Input] MAT_ParticleSPHFluid takes the DYN_VISCOSITY parameter. "
     "Signal: x.",
     True, "plausible abbreviation of a real key; 4C spells it "
           "DYNAMIC_VISCOSITY"),
    ("kratos", "PARTICLE_FRICTION",
     "[Input] Friction is the PARTICLE_FRICTION property of the DEM "
     "material. Signal: x.",
     True, "PRESCRIBED, not retracted -- the retraction in the live corpus "
           "must not make the name itself permanently unflaggable"),
    ("kratos", "FREESTREAM_VELOCITY",
     "[Physics] Set the FREESTREAM_VELOCITY variable on the model part. "
     "Signal: x.",
     True, "PRESCRIBED, not retracted"),
    ("fenics", "SOUNDSPEED",
     "[Input] The SOUNDSPEED parameter must be set. Signal: x.",
     True, "an invented key in a Python backend"),
    ("febio", "SLIP_COEFFICIENT",
     "[Input] The material takes a SLIP_COEFFICIENT parameter. Signal: x.",
     True, "an invented FEBio key"),
    ("sparta", "WALL_TEMPERATURE_FIXED",
     "[Input] The WALL_TEMPERATURE_FIXED key sets the surface temperature. "
     "Signal: x.",
     True, "an invented SPARTA key"),
    ("ngsolve", "BDM_1",
     "[Numerical] Use the BDM_1 space option. Signal: x.",
     True, "notation removed from the corpus; must fail if reintroduced"),
    # --- must NOT flag: real names the gate has to keep resolving ---
    ("dune", "SDIRK22",
     "[Numerical] Switch to the SDIRK22 scheme via the time-stepper "
     "selection parameter. Signal: x.",
     False, "REAL: dune/fem/solver/rungekutta/timestepcontrol.hh:154"),
    ("dune", "DIRK34",
     "[Numerical] Switch to the DIRK34 scheme via the time-stepper "
     "selection parameter. Signal: x.",
     False, "REAL: timestepcontrol.hh:154"),
    ("fenics", "KSP_DIVERGED_PC_FAILED",
     "[Numerical] Signal: reason -11 (KSP_DIVERGED_PC_FAILED) with the "
     "solution full of inf.",
     False, "REAL: petscksp.h:828 -- reachable only via the stack headers"),
    ("dealii", "EPS_TARGET_REAL",
     "[Numerical] set_which_eigenpairs(EPS_TARGET_REAL) with target=100. "
     "Signal: x.",
     False, "REAL: slepceps.h, EPSWhich enumerator 8"),
    ("kratos", "KRATOS_ERROR_IF",
     "[API] Check() raises through the KRATOS_ERROR_IF macro. Signal: x.",
     False, "REAL: a macro, so only the C++ SOURCE can confirm it"),
    ("kratos", "STATIC_FRICTION",
     "[Input] Friction is the STATIC_FRICTION property. Signal: x.",
     False, "REAL: the correct replacement for PARTICLE_FRICTION"),
    ("fourc", "DYNAMIC_VISCOSITY",
     "[Input] MAT_ParticleSPHFluid takes the DYNAMIC_VISCOSITY parameter. "
     "Signal: x.",
     False, "REAL: 4C -p"),
    ("febio", "FEBIO_BINARY",
     "[Setup] Set FEBIO_BINARY so check_availability() finds the build. "
     "Signal: x.",
     False, "an openPASO environment variable, not a FEBio key"),
]

_ROOTS: dict[str, list] = {}


def _verdict(backend: str, tok: str, entry: str):
    """The audit's own decision path for one token in one entry."""
    import audit_named_input_keys as a

    if backend not in _ROOTS:
        src, corpus, _ = a.search_roots(backend)
        _ROOTS[backend] = list(src) + list(corpus)
    roots = _ROOTS[backend]
    if not roots:
        pytest.skip(f"no corpus on disk for {backend}: an absence here would "
                    f"prove nothing, and a presence could not be checked")

    hits = [(i, j) for t, i, j in a.candidate_keys(entry) if t == tok]
    if not hits:
        return False, "not a candidate (stopword, shape, or no key marker)"
    if all(a._is_retracted(entry, i, j) or a._absence_asserted(entry, i, j)
           for i, j in hits):
        return False, "read as an absence claim"
    if tok in a.openpaso_env_vars():
        return False, "classed as an openPASO environment variable"
    present, whole_word = a.key_present(tok, roots)
    if present:
        return False, ("resolves in the corpus" if whole_word else
                       "resolves only as a substring of a longer identifier")
    return True, "flagged"


@pytest.mark.parametrize(
    "backend,tok,entry,must_flag,why", CASES,
    ids=[f"{c[0]}-{c[1]}-{'bad' if c[3] else 'good'}" for c in CASES])
def test_gate_verdict(backend, tok, entry, must_flag, why) -> None:
    flagged, reason = _verdict(backend, tok, entry)
    if must_flag:
        assert flagged, (
            f"{backend}: the fabricated key {tok} is NOT flagged — {reason}.\n"
            f"({why})\nA user following it writes a deck the solver rejects. "
            f"Whatever change made this pass has switched the gate off for "
            f"this shape of fabrication; widen the corpus or fix the entry, "
            f"never the stopword list or an absence cue.")
    else:
        assert not flagged, (
            f"{backend}: {tok} is a REAL name and is being reported as "
            f"invented — {why}.\nA gate that accuses correct knowledge gets "
            f"switched off. Fix the corpus paths, not the assertion.")
