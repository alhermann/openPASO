"""No OASiS tool may put a participant script into the agent's workspace.

OASiS answers questions. It does not install runnable solvers. That was
settled once by reverting an automatic delivery (64a922af) — and the manual
door was left standing, so the same thing kept happening through
`materialize_participant`, which shutil-copied the COMPLETE file, solve
included, into the agent's work_dir.

Measured in the C9 iteration of 2026-08-29, after Option B had supposedly
stopped OASiS handing over solvers: agents called that tool 2-4 times per run,
8 to 10 participant files landed in each workspace, and two per run carried the
full solve. Those three runs were void. Fixing the door and not the room is why
this test is about the CLASS rather than that one tool.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
sys.path.insert(0, str(SRC))

PARTICIPANT_DIR_HINTS = ("coupling_participants", "participant_")
COPY_CALLS = {"copy", "copy2", "copyfile", "copytree"}


def _tool_functions(tree: ast.AST):
    """Every function registered as an MCP tool."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for d in node.decorator_list:
            src = ast.dump(d)
            if "mcp" in src and "tool" in src:
                yield node
                break


def _writes_measured_csv(call: ast.Call) -> bool:
    """couple() writes the driver's MEASURED residual history to the path the
    agent asked for (`iteration,interface_residual` rows): numbers the run
    produced, not a script. That is the anti-fabrication route, not a solver
    hand-over. Exempt exactly that shape -- a write whose leftmost literal is
    a CSV header. A participant script never starts with one."""
    if not call.args:
        return False
    node = call.args[0]
    while isinstance(node, ast.BinOp):
        node = node.left
    return (isinstance(node, ast.Constant) and isinstance(node.value, str)
            and node.value.startswith("iteration,"))


def test_no_mcp_tool_copies_a_participant_file():
    """A tool may READ a participant to quote it. It may not copy it out."""
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for fn in _tool_functions(tree):
            body = ast.dump(fn)
            mentions_participants = any(h in body for h in PARTICIPANT_DIR_HINTS)
            if not mentions_participants:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = getattr(f, "attr", None) or getattr(f, "id", None)
                    if name in COPY_CALLS:
                        offenders.append(f"{path.name}::{fn.name} calls {name}()")
                    if name == "write_text" and not _writes_measured_csv(node):
                        offenders.append(
                            f"{path.name}::{fn.name} calls write_text() while "
                            f"handling participants")
    assert not offenders, (
        "an MCP tool installs a participant script into the agent's "
        "workspace:\n  " + "\n  ".join(offenders) +
        "\nOASiS answers questions; it does not hand over runnable solvers. "
        "The knowledge serves the script WITHOUT its solve; there is no "
        "second door.")


def test_the_removed_tool_is_not_advertised_anywhere():
    """A payload naming a tool that does not exist reads as an instruction.

    Same defect class as the ten passages that promised deal.II sources "in
    the directory the payload came from": an agent spends its budget on it.
    """
    from core.registry import load_all_backends
    load_all_backends()
    from tools.coupling_knowledge import coupling_knowledge, precice_knowledge
    from tools.coupling_knowledge import _BACKEND_ORDER

    for solver in [""] + list(_BACKEND_ORDER):
        for fn in (coupling_knowledge, precice_knowledge):
            text = fn(solver)
            assert "materialize_participant" not in text, (
                f"{fn.__name__}({solver!r}) still tells the agent to call "
                f"materialize_participant, which no longer exists")


def test_the_sign_lesson_survived_the_removal():
    """Deleting the tool must not delete what its section taught.

    The advertisement carried the one measured defect that matters: exporting
    the raw traction instead of the negated outward normal flux converges
    smoothly to the wrong answer. That is knowledge and it stays.
    """
    from core.registry import load_all_backends
    load_all_backends()
    from tools.coupling_knowledge import coupling_knowledge
    text = coupling_knowledge("")
    assert "OPPOSITE signs" in text
    assert "UNCHANGED" in text
    low = text.lower()
    assert "converges" in low and "wrong answer" in low


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
