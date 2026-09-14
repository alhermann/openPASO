"""OASiS must not tell the model under test that it is being tested.

The knowledge served to the OASiS arm carried, verbatim: "Measured over 256
runs of this campaign, agents stop VOLUNTARILY at a median of 46% of their wall
budget", "Three runs in this campaign wrote a complete manufactured source into
a deck", and the audit tool's own description said "measured over 40 graded
runs of a blind campaign". The bare arm has no knowledge tool, so ONLY the
treatment arm was told, in plain language, that it is one of many measured runs
in a study.

Worse, one line leaked the composition of the evaluation set: "One cell in this
campaign mandates the MORLEY element". That tells the agent such a cell exists.

The lessons those sentences carried are real and are kept. What is removed is
the provenance — an agent should read domain knowledge, not our lab notes.
"""
from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# phrases that tell the reader it is inside a measured study, or describe the
# evaluation set
# A PHRASE LIST IS NOT A FACT ABOUT THE TEXT. The first version of this list
# passed while src/backends/fourc/backend.py still said "Measured failure,
# three separate runs, all graded wrong ... which the agent submitted" — the
# same genus of sentence, minus the words "in this campaign". Widened to the
# shape rather than the wording: any claim that runs were GRADED, that an
# AGENT submitted something, or that a count of runs was measured.
BANNED = re.compile(
    r"this campaign|blind campaign|graded runs of|this benchmark|"
    r"one cell in this|of the campaign|our campaign|the eval(?:uation)? set|"
    r"graded wrong|the agent submitted|separate runs|runs of this|"
    r"\bgraded (?:as )?(?:wrong|correct)\b",
    re.I)


def _tools():
    from core.registry import load_all_backends
    load_all_backends()

    class M:
        def __init__(self):
            self.t = {}

        def tool(self, *a, **k):
            def d(f):
                self.t[f.__name__] = f
                return f
            return d

    import tools.consolidated as C
    m = M()
    for n in dir(C):
        if n.startswith("register") and callable(getattr(C, n)):
            try:
                getattr(C, n)(m)
            except Exception:                     # noqa: BLE001
                pass
    return m.t


def _call(fn, **kw):
    r = fn(**kw)
    return str(asyncio.run(r) if asyncio.iscoroutine(r) else r)


def test_no_served_payload_says_it_is_a_campaign():
    t = _tools()
    checks = [
        ("knowledge/physics", lambda: _call(t["knowledge"], topic="physics",
                                            solver="fenics", physics="poisson")),
        ("knowledge/coupling", lambda: _call(t["knowledge"], topic="coupling",
                                             solver="fenics")),
        ("knowledge/coupling/ngsolve", lambda: _call(t["knowledge"],
                                                     topic="coupling",
                                                     solver="ngsolve")),
        ("prepare_simulation", lambda: _call(t["prepare_simulation"],
                                             solver="fenics",
                                             physics="poisson")),
    ]
    bad = []
    for label, get in checks:
        hits = BANNED.findall(get())
        if hits:
            bad.append(f"{label}: {sorted({h if isinstance(h, str) else h[0] for h in hits})}")
    assert not bad, (
        "served text tells the model under test that it is in a study:\n  "
        + "\n  ".join(bad))


def test_tool_descriptions_do_not_say_it_either():
    """An MCP tool's docstring is surfaced to the model as its description."""
    t = _tools()
    bad = []
    for name, fn in t.items():
        doc = (fn.__doc__ or "")
        hits = BANNED.findall(doc)
        if hits:
            bad.append(f"{name}: {doc[:120]!r}")
    assert not bad, (
        "an MCP tool description tells the model it is in a study:\n  "
        + "\n  ".join(bad))


def test_every_backend_payload_is_clean_not_just_the_default_one():
    """The first pass checked four payloads and missed two live leaks.

    The SPARTA backend carried its own copy of the stopping-early lesson,
    with "Measured over 128 runs of this campaign" intact, served with SPARTA
    physics. And the element rule named Morley — which is exactly the element
    one evaluation cell mandates — inside a block served for ALL nine
    backends. Checking the default payload proved nothing about the other
    eight.
    """
    t = _tools()
    bad = []
    for solver in ("fenics", "ngsolve", "skfem", "dune", "dealii", "kratos",
                   "fourc", "febio", "sparta"):
        for label, out in (
            ("knowledge/physics", _call(t["knowledge"], topic="physics",
                                        solver=solver, physics="poisson")),
            ("prepare_simulation", _call(t["prepare_simulation"],
                                         solver=solver, physics="poisson")),
        ):
            hits = BANNED.findall(out)
            if hits:
                bad.append(f"{solver}/{label}: {sorted({str(h) for h in hits})}")
    assert not bad, "\n  ".join(bad)


def test_no_specific_element_is_named_as_an_evaluation_cell():
    """Naming Morley told every agent that a biharmonic/Morley cell exists.

    The RULE is what matters — degree+1 fails for whole element families and a
    prescribed element carries its own rate — and it states fine without
    naming the one the evaluation set uses.
    """
    t = _tools()
    text = _call(t["knowledge"], topic="physics", solver="skfem",
                 physics="poisson")
    assert "Morley" not in text, (
        "the served rule still names Morley, which is an evaluation cell's "
        "prescribed element")
    # THE RULE MOVED, IT WAS NOT LOST. The universal block was cut from 34,814
    # to 12,624 characters for the instruction budget, and its closing line
    # sends the reader to knowledge(topic='universal_full') for the long form.
    # Asserting the rule in the PHYSICS reply tested the contract as it stood
    # before that cut; asserting it where the reply points is the same
    # protection against losing the knowledge, and it is true.
    assert "universal_full" in text, (
        "the physics reply no longer points at the long form")
    long_form = _call(t["knowledge"], topic="universal_full")
    assert "NONCONFORMING" in long_form or "nonconforming" in long_form, (
        "the element rule itself must survive")
    assert "Morley" not in long_form


def test_the_lesson_survived_the_edit():
    """Removing provenance must not remove the knowledge."""
    t = _tools()
    text = _call(t["knowledge"], topic="physics", solver="fenics",
                 physics="poisson")
    assert "Morley" not in text
    assert "universal_full" in text, "the reply must say where the long form is"
    long_form = _call(t["knowledge"], topic="universal_full")
    # stopping early
    assert "wall budget" in long_form and "VOLUNTARILY" in long_form
    # the wiring rule
    assert "INERT UNTIL IT IS WIRED IN" in long_form
    # the prescribed-element rule, WITHOUT naming the element an evaluation
    # cell prescribes — the rule is the knowledge, the name was the leak
    assert "NONCONFORMING" in long_form or "nonconforming" in long_form
    assert "Morley" not in long_form


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
