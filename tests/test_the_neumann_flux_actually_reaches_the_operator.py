"""The defect that sank the best coupled runs: an imported flux that vanishes.

MEASURED BY EXECUTION, three Kratos solves on one mesh differing only in
whether ThermalFace2D2N conditions exist on the interface edges:

    zero flux, conditions present     max|T| = 2.307291e-03
    flux on nodes, NO conditions      max|T| = 2.307291e-03   BIT-IDENTICAL
    flux on nodes AND conditions      max|T| = 3.605675e-03

numpy.allclose on the first two is True. A nodal FACE_HEAT_FLUX is only ever
integrated BY a condition; with no condition there is nothing to integrate it,
and Kratos runs, converges, exits 0 and returns the no-flux field.

AND IT IS NOT HYPOTHETICAL. C2_27b_MCP_seed303 is a coupled submission whose
two prescribed codes BOTH genuinely ran, whose coupling genuinely iterated over
three mesh levels, whose side A is correct to three digits, and whose interface
FIELD matches across the seam to 0.000e+00. Its submitted side-B peak is
2.367156e-03. An independent zero-flux solve on the same mesh gives
2.367156e-03 -- identical to seven significant figures -- against a correct
3.670103e-03. Its imported flux never reached the operator. The only signal in
the whole submission was the flux jump GROWING under refinement: 8.139e-01,
9.066e-01, 9.530e-01.

The shape is general, not Kratos-specific: a boundary value attached to nodes
but never integrated over a facet contributes nothing, silently. 4C carries the
same trap twice -- the `DESIGN ... THERMO ...` sections are never evaluated in a
standalone Thermo problem, and a 2D body source must sit on the condition whose
geometry type matches the ELEMENT DIMENSION.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MEASURED_IGNORED = "2.307291e-03"
MEASURED_APPLIED = "3.605675e-03"
SUBMITTED = "2.367156e-03"


def _served_kratos_coupling() -> str:
    from tools.coupling_knowledge import coupling_knowledge
    return coupling_knowledge("kratos")


def test_the_served_kratos_neumann_route_creates_the_conditions():
    """The Neumann-side contract openPASO serves for Kratos must create the
    interface flux conditions in a loop over the interface edges, with the
    solve elided; without the conditions the imported flux is silently
    discarded."""
    from tools.coupling_knowledge import _PARTICIPANT_DIR, _serve_participant
    p = _PARTICIPANT_DIR / "participant_kratos_neumann.py"
    assert p.is_file(), "no served Neumann-side contract for Kratos"
    t = _serve_participant(p)
    assert "import KratosMultiphysics" in t
    assert re.search(r'CreateNewCondition\("(?:FluxCondition2D2N|ThermalFace2D2N)"', t), (
        "the served Neumann contract must CREATE the interface conditions; "
        "without them the imported flux is silently discarded")
    assert re.search(r"for j in range\(NY\):\s*\n\s*mp\.CreateNewCondition", t), (
        "the conditions must be created in a loop over the interface edges")
    assert "strategy.Solve()" not in t and "CreateModelPart" not in t, (
        "the served contract must not carry the solve")
    payload = _served_kratos_coupling()
    assert "THE NEUMANN-SIDE PARTICIPANT" in payload and t in payload, (
        "the Neumann-side contract is not reached from the Kratos coupling payload")


def test_the_served_kratos_traps_carry_the_measurement_that_proves_it():
    payload = _served_kratos_coupling()
    for number in (MEASURED_IGNORED, MEASURED_APPLIED):
        assert number in payload, (
            f"{number} is the measurement that makes this actionable; without "
            f"the numbers a reader cannot tell an assertion from a guess")
    assert "solve once with the imported flux" in payload, (
        "the one-step check (zeroed import vs real import) must be served")
    assert "CreateNewCondition" in payload and "by NAME" in payload


def test_the_two_signs_are_stated_and_are_opposite():
    """The other half of the trap: the flux you APPLY is the partner's number
    unchanged, the flux you REPORT is your OWN outward flux."""
    from tools.coupling_knowledge import _PARTICIPANT_DIR, _serve_participant
    t = _serve_participant(_PARTICIPANT_DIR / "participant_kratos_neumann.py")
    assert "UNCHANGED" in t, "the contract must say the imported flux is applied unchanged"
    assert "outward" in t.lower(), "the contract must state the exported flux is this side's own OUTWARD flux"


def test_the_coupling_must_read_leads_with_it():
    """It is the most expensive single mistake in the coupled cells, so it
    cannot be buried below the material an agent runs out of budget reading."""
    src = (ROOT / "src" / "tools" / "consolidated.py").read_text()
    i = src.index("_COUPLING_MUST_READ = ")
    block = src[i:i + 12000]
    assert "SILENTLY IGNORED WITHOUT A CONDITION" in block
    assert MEASURED_IGNORED in block and MEASURED_APPLIED in block
    # the one-step check an agent can actually afford
    assert "with the imported flux set to zero" in block, (
        "the must-read must give the cheap check: solve the Neumann side once "
        "with the flux zeroed and once with it real, and compare")


def test_the_audit_names_a_flux_that_never_arrived():
    """A side reporting a flux orders below its partner's has not received it.

    Fires on 27 of the 210 coupled runs in the tree, so it is not a guard
    written for a case that never occurs.
    """
    from tools.result_audit import interface_sign_findings

    runs = ROOT / "campaign3_blind" / "runs"
    if not runs.is_dir():
        pytest.skip("run tree absent")
    fired = 0
    for run in runs.glob("C*_27b_*_seed*"):
        w = run / "work"
        if not w.is_dir() or not list(w.rglob("interface_level*_[AB].csv")):
            continue
        try:
            fs = interface_sign_findings(w)
        except Exception:
            continue
        if any("transmission" in f["sequence"] for f in fs):
            fired += 1
    assert fired >= 10, (
        f"the transmission check fired on only {fired} runs; it was measured "
        f"to fire on 27, so either the check or the tree has changed")
