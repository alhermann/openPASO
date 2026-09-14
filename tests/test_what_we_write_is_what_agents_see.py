"""Every rule we write must ride the tool the live server exposes.

THE FAILURE THIS PINS. openPASO consolidated 61 tools into 20; the superseded
modules (tools/knowledge.py, tools/coupling.py, ...) stayed in the tree,
importable and editable, with no link to the live tool and no test binding
them. Between rounds 4 and 7 three rules were written into
tools/knowledge.py's `_UNIVERSAL` block — write-the-deliverable-first, the
instruction to run audit_results, and the element-degree rule — and NONE of
them ever reached an agent, because src/server.py registers only
register_consolidated_tools and agents call `knowledge`, never
`get_physics_knowledge`.

The verification I ran each time called register_knowledge_tools directly and
reported "served on 9/9 backends". It was true of a code path the server does
not run. Three rounds of conclusions rest on it: round 5's "prominence
refuted", round 6 and 7's "audit adoption 1 of 51".

So the test is not "does the text exist" — it is "does the text arrive through
the same door an agent walks through", asserted for every backend.
"""
import asyncio
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))


def _live_tools():
    """Exactly what src/server.py exposes — nothing more."""
    from core.registry import load_all_backends
    load_all_backends()

    class FakeMCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **k):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco

    import tools.consolidated as C
    m = FakeMCP()
    for name in dir(C):
        if name.startswith("register") and callable(getattr(C, name)):
            try:
                getattr(C, name)(m)
            except Exception:                       # noqa: BLE001
                pass
    return m.tools


def _call(fn, **kw):
    r = fn(**kw)
    if asyncio.iscoroutine(r):
        r = asyncio.run(r)            # see webui/runner.py: 3.12+ raises otherwise
    return str(r)


# Every rule that must reach an agent, by a phrase unique to it.
REQUIRED = [
    ("WRITE THE ANSWER FILE", "write-the-deliverable-first rule"),
    ("RUN `audit_results`", "instruction to self-check before submitting"),
    ("THE ELEMENT WINS", "prescribed-element / degree-order rule"),
    ("INERT UNTIL IT IS WIRED IN", "defined-but-not-wired rule"),
]
BACKENDS = ["fourc", "fenics", "dealii", "febio", "ngsolve", "skfem",
            "kratos", "dune", "sparta"]
PROBES = ["linear_elasticity", "heat", "thermal", "poisson",
          "rarefied_flow", "scalar_transport", "_general"]


def test_the_server_exposes_a_knowledge_tool():
    assert "knowledge" in _live_tools(), (
        "agents call `knowledge`; if it is not exposed nothing below matters")


def test_superseded_registrars_are_not_the_live_path():
    """Guards the specific illusion: verifying through the dead registrar."""
    live = _live_tools()
    assert "get_physics_knowledge" not in live, (
        "get_physics_knowledge is NOT exposed by the live server — any check "
        "that calls it is verifying a path agents never take")


def test_every_universal_rule_reaches_every_backend_through_the_live_tool():
    """Every rule must be reachable by an agent, on the path the reply names.

    THE LONG FORM MOVED; IT DID NOT DISAPPEAR. These four phrases live in the
    long universal block. The physics reply used to carry that block whole and
    was deliberately cut -- 34,814 characters to 12,624 -- with the short block
    now stating the same ten rules in condensed form and naming where the long
    form is: knowledge(topic='universal_full'). Looking for the long phrases in
    the physics reply alone therefore reports every rule as unreachable, when
    what changed is which call returns them.

    The guard's purpose survives intact. It exists because three rules were
    written into _UNIVERSAL and none ever reached an agent -- server.py
    registers only register_consolidated_tools, so a check that called
    get_physics_knowledge was verifying a path agents never take. So: the
    physics reply must still come back for every backend AND name the way to
    the long form, and the long form must deliver every rule. Both halves go
    through the live tool.
    """
    fn = _live_tools()["knowledge"]
    missing = []
    for be in BACKENDS:
        served = ""
        for ph in PROBES:
            s = _call(fn, topic="physics", solver=be, physics=ph)
            if not s.startswith(("No knowledge", "Unknown")):
                served += s
        assert served, f"{be}: the live tool returned nothing for any probe"
        if "universal_full" not in served:
            missing.append(f"{be}: the reply does not name the long form, so "
                           f"the rules below are unreachable from it")

    long_form = _call(fn, topic="universal_full")
    for phrase, label in REQUIRED:
        if phrase not in long_form:
            missing.append(f"universal_full: {label}")

    assert not missing, (
        "rules that never reach an agent through the live tool:\n  "
        + "\n  ".join(missing))


if __name__ == "__main__":
    asyncio.set_event_loop(asyncio.new_event_loop())
    test_the_server_exposes_a_knowledge_tool()
    print("  pass  server exposes `knowledge`")
    test_superseded_registrars_are_not_the_live_path()
    print("  pass  the superseded registrar is correctly not live")
    test_every_universal_rule_reaches_every_backend_through_the_live_tool()
    print("  pass  all universal rules reach all 9 backends via the live tool")
