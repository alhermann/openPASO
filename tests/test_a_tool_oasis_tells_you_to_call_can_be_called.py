"""If openPASO names a tool in text it serves, the agent must be able to call it.

MEASURED. `couple_levels` runs every prescribed mesh level in one warm-started
call. Its commit message (2026-09-11) gives the reason it was written: "six
proven couplings never reached level 3 when every level cost ten calls". The
campaign allowlist was frozen on 2026-09-01, ten days earlier, and was never
updated. Across 2095 recorded trajectories `couple_levels` is invoked 3 times
against 1076 invocations of `couple`, while openPASO's own text recommending it
appears in 260 of them. Agents were told 260 times to call a tool they did not
have.

`check_input` had the same defect from the other direction: the coupled ladder's
step-1 brief tells the worker to "run check_input(...) until it names no defect".

The failure is silent in both directions -- openPASO cannot see the allowlist, and
the allowlist cannot see what openPASO recommends -- so it needs a test rather than
a memory. A tool named in served text must be either reachable or deliberately
excluded, and the exclusion has to be written down.
"""
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Named in served text, deliberately NOT reachable in a blind campaign run.
# Anything added here is a decision, which is the point of making it explicit.
_DELIBERATELY_UNREACHABLE = {
    "coupled_solve":
        "legacy, fixed toy geometries; `couple` supersedes it and says so in "
        "its own docstring",
    "rediscover_backends":
        "mutates the install; a blind run is immutable by construction",
    "verify_interface_flux":
        "UNRESOLVED — a verification gate of the same class as "
        "verify_pde_consistency, which IS exposed. Named once in served text. "
        "Excluded here to keep this change to what was agreed, not because the "
        "exclusion is known to be right.",
    "transfer_field":
        "UNRESOLVED — a utility named once, in the server instructions. Same "
        "caveat as verify_interface_flux.",
}


def _registered_tool_names():
    tree = ast.parse((ROOT / "src/tools/consolidated.py").read_text())
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in n.decorator_list:
                if isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) \
                        and d.func.attr == "tool":
                    out.add(n.name)
    return out


def _named_in_served_text(names):
    """Tool names that appear, called, inside a string literal in openPASO source.

    A string literal is the conservative stand-in for "served": not every
    literal reaches an agent, but everything that reaches an agent is one.
    """
    files = list((ROOT / "src/tools").glob("*.py")) + \
        [ROOT / "src/core/instructions.py"]
    found = {}
    for f in files:
        try:
            tree = ast.parse(f.read_text())
        except (OSError, SyntaxError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Constant) and isinstance(n.value, str):
                for name in names:
                    if re.search(r"\b" + name + r"\s*\(", n.value):
                        found.setdefault(name, set()).add(f.name)
    return found


def _allowlist():
    tree = ast.parse((ROOT / "langgraph_eval/agent.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CAMPAIGN_MCP_TOOL_ALLOWLIST"
                for t in n.targets):
            return {ast.literal_eval(e) for e in n.value.args[0].elts}
    raise AssertionError("CAMPAIGN_MCP_TOOL_ALLOWLIST is gone — re-point this test")


def test_every_tool_openpaso_recommends_is_reachable_or_explicitly_excluded():
    named = _named_in_served_text(_registered_tool_names())
    unreachable = set(named) - _allowlist()
    undocumented = sorted(unreachable - set(_DELIBERATELY_UNREACHABLE))
    assert not undocumented, (
        "openPASO names these tools in text it serves, and no agent can call them: "
        + ", ".join(f"{t} (in {sorted(named[t])})" for t in undocumented)
        + ". Either add the tool to CAMPAIGN_MCP_TOOL_ALLOWLIST or record why "
          "it is deliberately unreachable in _DELIBERATELY_UNREACHABLE.")


def test_the_exclusion_list_does_not_outlive_its_reason():
    # An exclusion for a tool that is now reachable, or that openPASO no longer
    # mentions, is stale bookkeeping that would hide the next instance.
    named = _named_in_served_text(_registered_tool_names())
    stale = sorted(t for t in _DELIBERATELY_UNREACHABLE
                   if t in _allowlist() or t not in named)
    assert not stale, (
        "these exclusions no longer describe anything — remove them: "
        + ", ".join(stale))


def test_the_two_tools_the_stall_needed_are_reachable():
    # The specific instance this test was written for. couple_levels answers the
    # level-1 stall (197 of 313 coupling cells never reach level 2);
    # check_input is what the ladder's own step-1 brief tells the worker to run.
    allow = _allowlist()
    for tool in ("couple_levels", "check_input"):
        assert tool in allow, f"{tool} is recommended by openPASO and not reachable"
