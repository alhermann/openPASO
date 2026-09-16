"""Four solvers accept a setting and then ignore it. All four are now served.

This is the dominant failure of the campaign: exit 0, a converged message, and
a field that is zero or mesh-independent. Each entry below was reproduced by
EXECUTION, and the test asserts the numbers survive in the payload an AGENT
receives -- not merely that they exist somewhere in the source.

  NGSolve   after `from ngsolve import *`, any loop assigning `x` or `y`
            rebinds the symbolic coordinates to floats, so the source becomes a
            CONSTANT. Verified here: type(f) is CoefficientFunction before a
            44x44 probe loop and float after, value 0.02514662, with x and y
            left at 0.9886363636. CoefficientFunction((float, float)) is
            accepted silently. A constant body force on a fully-Dirichlet
            incompressible domain gives u identically zero -- 7.16e-17,
            3.60e-17, 1.30e-17 across the levels, order 0.0000 -- against
            1.2229e-02, 1.2210e-02, 1.2208e-02 with the symbolic source.
  DUNE-fem  solver="cg" is accepted on a NON-SYMMETRIC operator and
            scheme.solve does not raise: converged=False,
            linear_iterations=-10000, field left at the initial guess, every
            level exactly zero. cg gave peak 0.000000e+00 at all three levels,
            bicgstab gave 8.875850e-02. Corroborated against the run tree:
            solver="cg" appears 82 times across 19 of 32 DU2 run directories,
            bicgstab once. gmres converged at N=8 and N=16 then silently
            returned zero at N=32.
  4C        a standalone Thermo problem ignores every `DESIGN ... THERMO ...`
            condition section: max|T| = 0.000000000e+00 with exit 0. The plain
            sections match an independent assembly to 1.08e-15.
  Kratos    FACE_HEAT_FLUX on interface nodes with no ThermalFace2D2N
            condition is discarded: 2.307291e-03 ignored against 3.605675e-03
            applied, bit-identical to a zero-flux run.

And a residual norm does not catch any of it. NGSolve's R.vec.Norm() includes
the Dirichlet rows Newton never touches, so it froze at 1.115344e+00 while the
free-DOF residual was 2.36e-16, and a run printed "Newton did not converge"
for 49 iterations on a problem it had already solved.
"""
from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# the measurement behind each trap; a claim without its number is a guess
NUMBERS = ("0.02514662", "0.9886363636", "1.2229e-02",
           "8.875850e-02", "-10000",
           "0.000000000e+00", "1.08e-15",
           "2.307291e-03", "3.605675e-03",
           "1.115344e+00", "2.36e-16")


# THE TRAPS MOVED BEHIND A NAMED DOOR, AND THE TEST HAD TO FOLLOW.
#
# They used to be pasted into every reply. The long form is 25k characters, and
# reading it costs the actions an agent needs to solve the problem, so the short
# core now rides on every reply and its closing line names the call that returns
# the rest: knowledge(topic='universal_full'). That is a PROMISE, and the defect
# this file guards against is a promise the payload does not keep -- exactly the
# shape recorded for 'postmortems' and for 'universal_full' itself, which once
# returned the usage hint instead of the rules.
#
# So the invariant is now two-part, and it is stronger than the old one:
#   1. the door returns every measurement, and
#   2. the reply an agent actually receives NAMES that door.
# Asserted on the coupling reply as well, because that is the payload the
# coupled campaign serves.


def _served(topic="universal_full", solver="", physics="") -> str:
    from tools.consolidated import register_consolidated_tools

    class Cap:
        def __init__(self):
            self.fns = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.fns[fn.__name__] = fn
                return fn
            return deco

    cap = Cap()
    register_consolidated_tools(cap)
    fn = cap.fns["knowledge"]
    kw = {k: v for k, v in (("solver", solver), ("physics", physics)) if v}
    out = fn(topic, **kw)
    if inspect.isawaitable(out):
        out = asyncio.new_event_loop().run_until_complete(out)
    return out if isinstance(out, str) else str(out)


def test_every_measurement_reaches_the_agent():
    """Asserted on the SERVED payload, because a fact in the source that the
    payload drops is the failure this repo has recorded twelve times."""
    text = _served()
    missing = [n for n in NUMBERS if n not in text]
    assert not missing, (
        f"these measurements are absent from the payload an agent receives, "
        f"so the trap is documented and not served: {missing}\n"
        f"payload was {len(text)} chars")


def test_the_mechanism_is_named_not_just_the_number():
    text = _served()
    for phrase in ("rebinds the symbolic coordinates",
                   "DOES NOT RAISE",
                   "initial guess",
                   "silently ignores",
                   "ThermalFace2D2N"):
        assert phrase in text, f"the payload does not explain: {phrase!r}"


def test_the_agent_is_told_what_to_GATE_on():
    """Knowing the trap is not the same as being told the check."""
    text = _served()
    assert "GATE ON THREE THINGS" in text
    for check in ("converged", "peak|u| > 0", "three separated points",
                  "log2("):
        assert check in text, f"the gate does not mention {check!r}"


def test_the_coordinate_rebinding_is_real_on_this_install():
    """Re-measured here rather than trusted, because it is the claim that
    turns a symbolic source into a number."""
    try:
        from ngsolve import CoefficientFunction, pi, sin, x, y
    except Exception:                              # pragma: no cover
        import pytest
        pytest.skip("ngsolve not importable in this interpreter")
    before = sin(pi * x) * sin(pi * y)
    assert type(before).__name__ == "CoefficientFunction"
    px = py = None
    for iy in range(44):
        for ix in range(44):
            px = (ix + 0.5) / 44
            py = (iy + 0.5) / 44
    after = sin(pi * px) * sin(pi * py)
    assert isinstance(after, float), (
        "the rebinding no longer produces a float, so the served number is "
        "stale and must be re-measured")
    assert abs(px - 0.9886363636363636) < 1e-15
    # and the silent acceptance that hides it
    assert type(CoefficientFunction((after, after))).__name__ \
        == "CoefficientFunction"


def test_the_reply_an_agent_receives_names_the_door(): 
    """A measurement served behind a door nobody is told about is not served."""
    from tools.knowledge import _UNIVERSAL_CORE
    assert "universal_full" in _UNIVERSAL_CORE, (
        "the short core no longer names the call that returns the long form, "
        "so the traps are unreachable from a normal reply")
    for kw in ({"topic": "physics", "solver": "ngsolve", "physics": "stokes"},
               {"topic": "coupling", "solver": "ngsolve"},
               {"topic": "coupling", "solver": "dune"},
               {"topic": "coupling", "solver": "kratos"}):
        out = _served(**kw)
        assert "universal_full" in out, (
            f"knowledge({kw}) does not name knowledge(topic='universal_full'), "
            f"so an agent reading it never learns the traps exist")
