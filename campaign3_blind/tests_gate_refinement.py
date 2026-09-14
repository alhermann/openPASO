"""The interface gate must test what its own message claims.

It said "a jump that stays O(1) under refinement" and implemented "the max over
levels is below 5e-3", which is a different and much worse test.

Why worse. When both sides recover the interface flux from their assembled
system, the two exported arrays differ by the consistent-to-nodal conversion of
the P1 boundary mass matrix, so the graded jump reduces to h^2|q''|/(6|q|) — a
mesh ruler with no physics in it. Two independent reviews reproduced the
consequence on a coupling with one side's conductivity 4x WRONG: it fails at
n=8 and n=16 and PASSES at n=32, on resolution alone.

And in this campaign's own grades, C8 seed 4 failed SOLELY on this gate in both
arms while its flux jump fell at every level — BARE 3.3% -> 2.0% -> 1.1%, MCP
8.9% -> 3.6% -> 1.5%. Converging couplings graded as unphysical. Both grade
CORRECT under the refinement test.

It mattered asymmetrically too: the recovery that passes a fixed tolerance is
described in the openPASO payload and nowhere in the task text, so a
grading-critical rule was published to one arm only. Testing for
non-convergence removes that, because both recoveries converge.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from grading import constants as C          # noqa: E402

TOL, DEC, CEIL = (C.IFACE_JUMP_TOL, C.IFACE_JUMP_DECAY,
                  C.IFACE_JUMP_CEILING)


def _verdict(seq_u, seq_q):
    """The gate's decision.

    THIS MIRRORS grading/iface.py RATHER THAN IMPORTING IT, and an independent
    reviewer was right to flag that as drift risk: a change to the grader that
    this file does not copy would go unnoticed. It is mirrored because
    interface_phase() needs a work directory, a spec and a sealed key to reach
    the verdict, and the point here is to exercise the DECISION over sequences
    that no run has produced. test_the_mirror_matches_the_grader below pins the
    two together on the constants, so a threshold change cannot silently
    diverge.
    """
    graded = [{"jump_u_rel": u, "jump_q_rel": q} for u, q in zip(seq_u, seq_q)]

    def shrinks(key):
        v = [g[key] for g in graded]
        if len(v) < 2:
            return None
        if max(v) <= TOL:
            return True
        first, last = v[0], v[-1]
        if first <= 0:
            return last <= TOL
        return (last < first * (DEC ** (len(v) - 1))
                and last <= CEIL)

    su, sq = shrinks("jump_u_rel"), shrinks("jump_q_rel")
    if su is None or sq is None:
        bad = max(seq_u) > TOL or max(seq_q) > TOL
    else:
        bad = not (su and sq)
    # A bit-exact zero FLUX jump is one field reported as two — see iface.py.
    # A zero FIELD jump is ordinary (Dirichlet-Neumann makes it exact by
    # construction) and every [0,0,0] field case below is expected to PASS.
    if not bad and any(q == 0.0 for q in seq_q):
        return "FAIL"
    return "FAIL" if bad else "PASS"


# (name, field jumps, flux jumps, expected)
CASES = [
    ("consistent recovery, roundoff", [1e-15] * 3, [1.6e-14] * 3, "PASS"),
    ("direct stress recovery, O(h)", [0, 0, 0], [3.3e-2, 2.0e-2, 1.1e-2], "PASS"),
    ("C8 seed 4 MCP, as measured", [0, 0, 0], [8.9e-2, 3.6e-2, 1.5e-2], "PASS"),
    ("C8 seed 4 BARE, as measured", [4.8e-3, 4.0e-3, 3.1e-3],
     [3.27e-2, 2.03e-2, 1.11e-2], "PASS"),
    # the mutation signatures the gate was calibrated on: flat, O(1)
    ("mutation: flat at 0.75", [0, 0, 0], [0.75] * 3, "FAIL"),
    ("mutation: flat and large", [0, 0, 0], [3.0, 2.9, 3.1], "FAIL"),
    ("mutation: growing", [0, 0, 0], [1.9, 2.0, 2.1], "FAIL"),
    ("mutation: barely shrinking", [0, 0, 0], [1.0, 0.9, 0.81], "FAIL"),
    ("field jump flat, flux fine", [0.2] * 3, [1e-14] * 3, "FAIL"),
    ("one level above tolerance", [0.0], [2e-2], "FAIL"),
    ("one level at roundoff", [0.0], [1e-14], "PASS"),
    # C7_27b_BARE_seed2 as measured: both quantities bit-exactly zero at every
    # level, because both sides were read off one monolithic field. It graded
    # CORRECT — the campaign's only coupled success — until the flux rule.
    ("C7 seed 2 BARE, the fabrication", [0.0] * 3, [0.0] * 3, "FAIL"),
    # and the discrimination that makes the rule safe: a zero FIELD jump with a
    # real converging flux is honest Dirichlet-Neumann and must still pass.
    ("zero field, converging flux", [0.0] * 3, [3.3e-2, 2.0e-2, 1.1e-2], "PASS"),
]


def test_the_gate_separates_converging_from_stuck():
    bad = []
    for name, u, q, want in CASES:
        got = _verdict(u, q)
        if got != want:
            bad.append(f"{name}: wanted {want}, got {got}")
    assert not bad, "\n  ".join(bad)


def test_a_mutation_cannot_pass_by_refining():
    """The old gate let a wrong coupling through at a fine enough mesh."""
    # conductivity 4x wrong, jump = h^2|q''|/(6|q|): shrinks like h^2 but the
    # FIELD jump stays put because the transmission condition is wrong
    assert _verdict([0.31, 0.31, 0.31], [2.5e-2, 6.4e-3, 1.6e-3]) == "FAIL"


def test_the_decay_threshold_is_between_mutation_and_o_of_h():
    """0.5 is O(h), 0.25 is O(h^2), ~1.0 is a mutation. 0.75 sits between."""
    assert 0.5 < DEC < 1.0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))


# (name, field jumps, flux jumps, expected) — the magnitude floor
CEILING_CASES = [
    ("decays past the rate but lands at 49%", [0, 0, 0], [1.00, 0.70, 0.49], "FAIL"),
    ("decays but lands at 40%", [0, 0, 0], [0.90, 0.60, 0.40], "FAIL"),
    ("decays to 6%", [0, 0, 0], [0.5, 0.2, 0.06], "PASS"),
    ("O(h), lands at 1.1%", [0, 0, 0], [3.3e-2, 2.0e-2, 1.1e-2], "PASS"),
]


def test_a_decaying_but_enormous_jump_is_refused():
    """Decay alone was not enough.

    Requiring only that the jump SHRINKS admitted a 49% interface mismatch,
    which is not a coupling — the gate's own message says a mismatch means the
    scheme converged to the wrong transmission condition. The ceiling is 20x
    the strict tolerance and an order of magnitude above the largest
    finest-level jump any legitimate run in this campaign produces (1.5%).
    """
    bad = []
    for name, u, q, want in CEILING_CASES:
        got = _verdict(u, q)
        if got != want:
            bad.append(f"{name}: wanted {want}, got {got}")
    assert not bad, "\n  ".join(bad)


def test_the_mirror_matches_the_grader():
    """The mirror must not drift from the code it mirrors.

    _verdict reimplements grading/iface.py's decision. Pin every constant it
    reads, so a threshold changed in one place and not the other fails here
    rather than in a graded round.
    """
    import re
    src = (HERE / "grading" / "iface.py").read_text()
    for const in ("IFACE_JUMP_TOL", "IFACE_JUMP_DECAY", "IFACE_JUMP_CEILING"):
        assert f"C.{const}" in src, (
            f"grading/iface.py no longer reads {const}; the mirror in this "
            f"file is stale")
    # and the shape of the test: decay AND ceiling, both required
    assert re.search(r"IFACE_JUMP_DECAY.*\n.*IFACE_JUMP_CEILING", src), (
        "iface.py no longer applies decay and ceiling together")
