"""A finding delivered at 98% of the run changes nothing. Fire it earlier.

MEASURED, from file mtimes over the six C2 openPASO runs of rounds 4 and 5:

    seed301  RESULT.txt at 98% of the file-activity span,  40s of activity after
    seed302  93%,  115s after
    seed303  68%,  357s after
    seed401  98%,   26s after
    seed402  99%,    8s after
    seed403  99%,   16s after

seed303 is the only one that wrote its submission with real time left, and it
is the only one of the six that reached a gradeable convergence order with
BOTH prescribed codes proven. seed301 received SEVEN findings at the moment of
submission -- three too-short residual histories, an identical-history-across-
levels, two near-zero fields -- and had forty seconds.

The submission audit is not wrong and is not removed. It is simply the last
possible moment to be told. So the two coupled checks now also fire on the
ARTEFACT write, each running only the check its own file makes possible:

    residual_level<k>.csv         length, and identity across levels
    interface_level<k>_<side>.csv the flux SIGN, once both sides and the
                                  matching solution file exist

This is the eleventh time in this repo that a mechanism reached the agent and
still did not reach the case it was built for -- here by arriving too late
rather than by not arriving. The test therefore asserts on the REPLY THE AGENT
GETS from a write, with two real submissions as fixtures.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

RUNS = ROOT / "campaign3_blind" / "runs"
WRONG_SIGN = RUNS / "C2_27b_MCP_seed303" / "work"
BAD_HISTORY = RUNS / "C2_27b_MCP_seed301" / "work"
GOOD = Path("/tmp/claude-1001/-home-alexander-4C/"
            "b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/c2_real/submission")

_PATTERNS = ("interface_level*_[AB].csv", "solution_level*_[AB].csv",
             "residual_level*.csv")


def _reply_when_writing(src: Path, last: str) -> str:
    """Stage every artefact EXCEPT `last`, then write `last` and return the
    reply — which is what the agent actually sees."""
    from langgraph_eval.agent import _read_write_tools_for

    tmp = Path(tempfile.mkdtemp())
    try:
        for pat in _PATTERNS:
            for f in src.rglob(pat):
                if f.name != last:
                    shutil.copy(f, tmp / f.name)
        target = next((f for f in src.rglob(last)), None)
        if target is None:
            pytest.skip(f"fixture {last} absent under {src}")
        wf = [t for t in _read_write_tools_for(tmp, audit_on_submit=True)
              if t.name == "write_file"][0]
        return wf.invoke({"path": last, "content": target.read_text()})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.skipif(not WRONG_SIGN.is_dir(), reason="seed303 tree absent")
def test_the_sign_finding_arrives_when_the_interface_file_is_written():
    out = _reply_when_writing(WRONG_SIGN, "interface_level3_B.csv")
    assert "WRONG SIGN" in out.upper(), (
        "writing the last interface file of a submission whose level-3 side B "
        "flux carries the INWARD normal produced no warning; the agent would "
        "next learn this at submission, with seconds left:\n" + out[:600])
    assert "-250.8" in out or "250.8" in out, (
        "the finding must carry the measured implied coefficient, so the "
        f"agent can see WHICH side and by how much: {out[:400]}")


@pytest.mark.skipif(not BAD_HISTORY.is_dir(), reason="seed301 tree absent")
def test_the_history_finding_arrives_when_the_residual_file_is_written():
    out = _reply_when_writing(BAD_HISTORY, "residual_level3.csv")
    assert "TOO SHORT" in out.upper(), (
        f"a 2-row residual history drew no warning on write: {out[:600]}")
    assert "IDENTICAL" in out.upper(), (
        "three bit-identical histories across three different meshes drew no "
        f"warning on write: {out[:600]}")


@pytest.mark.skipif(not GOOD.is_dir(), reason="reference submission absent")
@pytest.mark.parametrize("last", ["interface_level3_B.csv",
                                  "residual_level3.csv"])
def test_a_correct_submission_is_not_nagged(last):
    """The cost side. A check that fires on the correct case teaches the agent
    to ignore it, which is worse than not having it."""
    out = _reply_when_writing(GOOD, last)
    body = out.split("\n", 1)[1] if "\n" in out else ""
    assert not body.strip(), (
        f"writing {last} of a submission graded CORRECT at order 1.9796 "
        f"produced a warning:\n{body[:600]}")


def test_the_bare_arm_gets_none_of_this():
    """The hook is openPASO's capability; the control arm must be untouched, or
    the measured uplift includes a mechanism the bare arm was also given."""
    from langgraph_eval.agent import _read_write_tools_for

    if not BAD_HISTORY.is_dir():
        pytest.skip("seed301 tree absent")
    tmp = Path(tempfile.mkdtemp())
    try:
        for pat in _PATTERNS:
            for f in BAD_HISTORY.rglob(pat):
                shutil.copy(f, tmp / f.name)
        wf = [t for t in _read_write_tools_for(tmp, audit_on_submit=False)
              if t.name == "write_file"][0]
        out = wf.invoke({"path": "residual_level3.csv", "content": "1,1.0\n"})
        body = out.split("\n", 1)[1] if "\n" in out else ""
        assert not body.strip(), f"the bare arm was audited: {body[:300]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- shell route
def _reply_from_shell(src: Path, audit: bool) -> str:
    """The agent's OWN SOLVER SCRIPT writes the artefacts, so the only channel
    that can see them is the shell command that ran it.

    MEASURED, and this is the twelfth instance of the recurring theme: the
    write_file hook reached NONE of round 7's three openPASO runs. Their work dirs
    hold 6, 3 and 11 Python scripts producing 12, 5 and 15 per-level CSVs -- the
    agent writes a program with write_file and the PROGRAM writes the
    deliverables. This file already recorded the same lesson for RESULT.txt,
    where 57% of submitters wrote it by shell only, and the fix there was never
    extended to the artefacts.
    """
    import shutil
    import tempfile

    from langgraph_eval.agent import _bash_tool_for

    tmp = Path(tempfile.mkdtemp())
    stage = tmp / "stage"
    stage.mkdir()
    try:
        for pat in _PATTERNS:
            for f in src.rglob(pat):
                shutil.copy(f, stage / f.name)
        bt = _bash_tool_for(tmp, audit_on_submit=audit)
        return bt.invoke({"command": "cp stage/*.csv . && echo wrote"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.skipif(not WRONG_SIGN.is_dir(), reason="seed303 tree absent")
def test_a_shell_written_artefact_is_checked_too():
    out = _reply_from_shell(WRONG_SIGN, audit=True)
    assert "WRONG SIGN" in out.upper(), (
        "the artefacts arrived by shell and drew no warning, which is how the "
        f"write_file hook reached zero of round 7's runs:\n{out[:600]}")


@pytest.mark.skipif(not BAD_HISTORY.is_dir(), reason="seed301 tree absent")
def test_one_finding_per_kind_not_one_per_file():
    """A batch write must not repeat the same sentence once per level, and must
    not check only ONE file either.

    The first version of the shell hook took the single newest touched file. A
    solver writes every artefact in one go, so "newest" is arbitrary inside the
    batch: measured on a nine-file write it picked residual_level1.csv, whose
    history was fine, and the interface sign -- the finding that decided the
    round -- was never looked at.
    """
    out = _reply_from_shell(BAD_HISTORY, audit=True)
    assert "TOO SHORT" in out.upper(), out[:400]
    assert out.upper().count("COUPLING HISTORY TOO SHORT") == 1, (
        "the same finding is repeated once per level; one reply, one sentence:"
        f"\n{out[:600]}")
    assert "WRONG SIGN" in out.upper(), (
        "only one artefact kind was checked, so the batch's other check was "
        f"skipped:\n{out[:600]}")


@pytest.mark.skipif(not GOOD.is_dir(), reason="reference absent")
def test_the_shell_route_is_silent_on_a_correct_submission():
    out = _reply_from_shell(GOOD, audit=True)
    for bad in ("WRONG SIGN", "TOO SHORT", "IDENTICAL", "DOES NOT SHRINK"):
        assert bad not in out.upper(), f"{bad} on a CORRECT submission: {out[:400]}"


@pytest.mark.skipif(not WRONG_SIGN.is_dir(), reason="seed303 tree absent")
def test_the_shell_route_is_silent_for_the_bare_arm():
    out = _reply_from_shell(WRONG_SIGN, audit=False)
    assert "WRONG SIGN" not in out.upper(), (
        f"the control arm received an openPASO capability: {out[:400]}")


# ------------------------------------------- defects visible in the SCRIPT
def _reply_writing_script(src: str, audit: bool = True) -> str:
    """What the agent gets back when it writes a participant script.

    THIS IS THE MECHANISM THAT WORKED. Measured over the 18 openPASO runs of the
    coupled cell served the interface-condition fact: 18 of 18 called a
    knowledge door, 18 of 18 set FACE_HEAT_FLUX, and 0 of 18 created the
    condition that makes it do anything. They find openPASO, read it, and grasp
    the concept; the one line that turns a nodal value into a boundary
    condition does not survive from prose into code. So the script is read
    instead, and the reply carries the fix rather than the diagnosis.
    """
    import shutil
    import tempfile

    from langgraph_eval.agent import _read_write_tools_for

    tmp = Path(tempfile.mkdtemp())
    try:
        wf = [t for t in _read_write_tools_for(tmp, audit_on_submit=audit)
              if t.name == "write_file"][0]
        return wf.invoke({"path": "participant.py", "content": src})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_nodal_flux_with_no_condition_is_named_and_fixed():
    bad = ("import KratosMultiphysics as KM\n"
           "n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, q)\n")
    out = _reply_writing_script(bad)
    assert "CREATES NO CONDITION" in out, out[:400]
    # the message must carry the FIX, not only the finding
    assert "CreateNewCondition" in out and "ThermalFace2D2N" in out, (
        "18 runs were told the fact in prose and none acted on it; the reply "
        "has to hand over the code")
    assert "2.307291e-03" in out and "3.605675e-03" in out


def test_the_same_script_with_a_condition_is_not_nagged():
    good = ('import KratosMultiphysics as KM\n'
            'mp.CreateNewCondition("ThermalFace2D2N", 1, [1, 2], prop)\n'
            'n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, q)\n')
    assert "CREATES NO CONDITION" not in _reply_writing_script(good)


def test_cg_on_an_advective_operator_is_named():
    """20 openPASO runs do this. cg is accepted on a non-symmetric operator and
    scheme.solve does NOT raise -- it returns the initial guess, so every level
    is exactly zero with exit 0. Measured 0.000000e+00 against 8.875850e-02
    for bicgstab."""
    bad = ('from dune.fem import galerkin\n'
           'a = dot(b_vec, grad(u))*v*dx\n'
           's = galerkin([a == L], solver="cg")\n')
    out = _reply_writing_script(bad)
    assert "ADVECTION TERM" in out, out[:400]
    assert "-10000" in out and "8.875850e-02" in out
    # cg WITHOUT advection is defensible -- 67 runs do it -- and must be spared
    ok = 'from dune.fem import galerkin\na = grad(u)*grad(v)*dx\ns = galerkin([a==L], solver="cg")\n'
    assert "ADVECTION TERM" not in _reply_writing_script(ok)


def test_rebinding_x_or_y_in_ngsolve_is_named():
    """27 openPASO runs do this. After `from ngsolve import *` those names ARE the
    symbolic coordinates, so the assignment turns the source into a constant:
    measured type() CoefficientFunction -> float, value 0.02514662, giving
    u identically zero and order 0.0000 against 1.2229e-02."""
    bad = ("from ngsolve import *\n"
           "for ix in range(44):\n    x = (ix + 0.5) / 44\n"
           "f = sin(pi * x)\n")
    out = _reply_writing_script(bad)
    assert "REBINDS" in out, out[:400]
    assert "0.02514662" in out and "px, py" in out
    safe = ("from ngsolve import *\n"
            "for ix in range(44):\n    px = (ix + 0.5) / 44\n"
            "f = sin(pi * x)\n")
    assert "REBINDS" not in _reply_writing_script(safe)


def test_the_bare_arm_gets_none_of_the_script_checks():
    bad = ("import KratosMultiphysics as KM\n"
           "n.SetSolutionStepValue(KM.FACE_HEAT_FLUX, q)\n")
    assert "CREATES NO CONDITION" not in _reply_writing_script(bad, audit=False)
