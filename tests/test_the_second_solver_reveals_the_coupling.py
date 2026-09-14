"""Two different codes prepared in one session get the coupled must-read.

MEASURED, round 18 of the coupled development cell. Two of the three openPASO
runs called prepare_simulation twice -- once per prescribed code -- and never
opened any knowledge door. The coupled must-read (the couple() recipe, the
fields-vs-evidence hierarchy, the measured-not-modelled history rule, the
captured-log contract) hung off doors they did not open, so it reached zero
of them; the third run opened a knowledge door once, received it once, and
typed 6.234567890123456e-01 into its residual history anyway. The door every
run DOES open is prepare_simulation -- server.py tells them to call it first,
and 6 of 6 calls in the round were to it.

Preparing two DIFFERENT backends in one session is the agent's own signature
of a partitioned coupling. No task knowledge is used: the trigger is entirely
the agent's behaviour, the header states the inference honestly (a single-code
run exploring alternatives is told to skip), and the must-read is served ONCE.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _prepare():
    from core.registry import load_all_backends
    load_all_backends()
    import tools.consolidated as C

    class M:
        def __init__(self):
            self.t = {}

        def tool(self, *a, **k):
            def d(f):
                self.t[f.__name__] = f
                return f
            return d

    m = M()
    C.register_consolidated_tools(m)
    return m.t["prepare_simulation"]


def test_the_second_different_solver_carries_the_must_read():
    prep = _prepare()
    r1 = prep("fourc", "heat")
    assert "couple(participants=" not in r1, "served before the signature"
    r2 = prep("kratos", "heat")
    for s in ("couple(participants=", "THE FIELDS ARE THE RESULT",
              "MEASURED, NOT MODELLED", "If your task couples them"):
        assert s in r2, s
    # the call's own payload is not displaced
    assert "WHAT DECIDES THIS RUN" in r2


def test_it_is_served_once_not_on_every_call():
    prep = _prepare()
    prep("fourc", "heat")
    prep("kratos", "heat")
    r3 = prep("kratos", "poisson")
    r4 = prep("fourc", "poisson")
    assert "couple(participants=" not in r3
    assert "couple(participants=" not in r4


def test_the_same_solver_twice_reveals_nothing():
    """Re-preparing one code (a retry, a second physics) is not a coupling."""
    prep = _prepare()
    prep("fourc", "heat")
    r2 = prep("fourc", "poisson")
    assert "couple(participants=" not in r2
