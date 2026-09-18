"""What openPASO tells the model to call, the product path must carry.

MEASURED, twice. The evaluation surface once recommended three tools it did not expose; the fix
there was an allowlist test. The same defect then sat in the product: `run_agent.py` deliberately
left out `spawn_subagent`, while the served instructions require an independent critic on every
major step and a worker sub-agent per step of a coupled problem. A first real user run on the
browser interface spent ten of its thirteen minutes in critic rounds and never reached a solver;
on the command line the model could not have spawned one at all.

This test reads the tool names out of the served instructions and checks each one against what a
run actually gets: the MCP tools, the host tools, and the ones run_agent adds itself.
"""
import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.instructions import INSTRUCTIONS  # noqa: E402

# Named in the text as something the client provides or as a deliberate "do not use this".
NOT_OURS = {"coupled_solve"}          # named only to say it is superseded by couple


def _registered_mcp_tools() -> set:
    tree = ast.parse((ROOT / "src" / "tools" / "consolidated.py").read_text())
    return {n.name for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any(isinstance(d, ast.Attribute) and d.attr == "tool"
                    or isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute)
                    and d.func.attr == "tool" for d in n.decorator_list)}


def _run_agent_tools() -> set:
    """Tool names run_agent.py adds on top of the MCP set."""
    src = (ROOT / "run_agent.py").read_text()
    return set(re.findall(r'name="([a-z_]+)"', src)) | {
        "run_bash", "read_file", "write_file", "list_files"}


def test_every_tool_named_in_the_served_instructions_is_reachable():
    named = {n for n in re.findall(r"\b([a-z_]{4,})\(", INSTRUCTIONS)}
    available = _registered_mcp_tools() | _run_agent_tools()
    missing = sorted(n for n in named
                     if n in _known_tool_names() and n not in available and n not in NOT_OURS)
    assert not missing, (
        "openPASO's own instructions tell the model to call tools the product path does not "
        f"carry: {missing}")


def _known_tool_names() -> set:
    """Names that are tools somewhere in this repository, so prose verbs are not mistaken for one."""
    return _registered_mcp_tools() | {
        "spawn_subagent", "web_search", "run_bash", "read_file", "write_file", "list_files"}


def test_the_command_line_path_carries_a_sub_agent():
    src = (ROOT / "run_agent.py").read_text()
    assert 'name="spawn_subagent"' in src, (
        "the served instructions require a critic sub-agent on every major step and a worker "
        "per coupled step; the command-line path must provide one")
    assert "not an independent one" in src, (
        "a sub-agent on the parent's own model is a second opinion; say so rather than implying "
        "independence")


def test_a_blocked_web_search_is_not_reported_as_an_empty_web():
    """The wording belongs to the tool, not to a wrapper, so the model is told once."""
    shared = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert "could not search" in shared and "NOT as 'the web has nothing on this'" in shared
    assert "NOT an empty web" not in (ROOT / "run_agent.py").read_text(), (
        "the shared tool says this now; saying it again in the wrapper repeats it to the model")
