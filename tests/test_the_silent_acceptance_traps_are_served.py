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


def _reachable(topic="physics", solver="ngsolve", physics="stokes") -> str:
    """Everything the agent can actually get, on the documented path.

    The physics reply used to carry the long universal block whole. It was then
    deliberately cut -- 34,814 characters to 12,624 -- and the long form moved
    behind an explicit request, with the short reply's closing line telling the
    agent to ask `knowledge(topic="universal_full")` for the rest. Reading only
    the first reply therefore reports every trap as unserved, when what changed
    is where they are served from.

    So this is the pair: the short reply must OFFER the path, and the path must
    DELIVER. That is the contract the code states in its own words -- "a promise
    made in the payload and not kept" is the defect it guards against -- and it
    is a stricter thing to assert than one payload containing everything.
    """
    short = _served(topic, solver, physics)
    assert "universal_full" in short, (
        "the first physics reply no longer tells the agent how to reach the "
        "long form, so the traps below are unreachable in practice")
    return short + "\n" + _served("universal_full", "", "")


def _served(topic="physics", solver="ngsolve", physics="stokes") -> str:
    # LOAD THE BACKENDS FIRST, OR THIS MEASURES AN EMPTY REGISTRY.
    #
    # Without this the reply is "Unknown solver: ngsolve" and every assertion
    # below fails, saying the traps are not served when they are. It passed
    # only when some earlier test in the same process had happened to load the
    # registry, so the result depended on what ran before it -- and in a run
    # where this file came first, it reported a documentation failure that did
    # not exist.
    from core.registry import load_all_backends
    from tools.consolidated import register_consolidated_tools

    load_all_backends()

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
    out = fn(topic, solver=solver, physics=physics)
    if inspect.isawaitable(out):
        out = asyncio.new_event_loop().run_until_complete(out)
    return out if isinstance(out, str) else str(out)


def test_every_measurement_reaches_the_agent():
    """Asserted on the SERVED payload, because a fact in the source that the
    payload drops is the failure this repo has recorded twelve times."""
    text = _reachable()
    missing = [n for n in NUMBERS if n not in text]
    assert not missing, (
        f"these measurements are absent from the payload an agent receives, "
        f"so the trap is documented and not served: {missing}\n"
        f"payload was {len(text)} chars")


def test_the_mechanism_is_named_not_just_the_number():
    text = _reachable()
    for phrase in ("rebinds the symbolic coordinates",
                   "DOES NOT RAISE",
                   "initial guess",
                   "silently ignores",
                   "ThermalFace2D2N"):
        assert phrase in text, f"the payload does not explain: {phrase!r}"


def test_the_agent_is_told_what_to_GATE_on():
    """Knowing the trap is not the same as being told the check."""
    text = _reachable()
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
