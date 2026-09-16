"""No openPASO tool may put a participant script into the agent's workspace.

openPASO answers questions. It does not install runnable solvers. That was
settled once by reverting an automatic delivery (64a922af) — and the manual
door was left standing, so the same thing kept happening through
`materialize_participant`, which shutil-copied the COMPLETE file, solve
included, into the agent's work_dir.

Measured in the C9 iteration of 2026-08-29, after Option B had supposedly
stopped openPASO handing over solvers: agents called that tool 2-4 times per run,
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


def _copies_a_captured_log(call: ast.Call) -> bool:
    """couple() re-files a participant's OWN captured console output under the
    level that produced it (`participant_output.log` ->
    `participant_output_level<k>.log`), so the coarse level's log survives the
    fine one. That is run EVIDENCE moving inside the agent's own work_dir --
    output the agent's solver printed -- and preserving it is the opposite of
    handing over a solver: it is what lets a grader prove which code ran.

    A participant SCRIPT is never a .log, so exempt exactly this shape: the
    write target mentions a log and the content is a read of another file. The
    gate keeps its teeth on anything that writes a .py, a deck, or a literal
    program."""
    target = getattr(call.func, "value", None)
    if target is None or "log" not in ast.dump(target).lower():
        return False
    if not call.args:
        return False
    arg = ast.dump(call.args[0])
    return "read_text" in arg or "read_bytes" in arg


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
                    if (name == "write_text"
                            and not _writes_measured_csv(node)
                            and not _copies_a_captured_log(node)):
                        offenders.append(
                            f"{path.name}::{fn.name} calls write_text() while "
                            f"handling participants")
    assert not offenders, (
        "an MCP tool installs a participant script into the agent's "
        "workspace:\n  " + "\n  ".join(offenders) +
        "\nopenPASO answers questions; it does not hand over runnable solvers. "
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


def test_the_exemptions_do_not_blunt_the_gate():
    """An exemption is a hole unless it is measured.

    Two writes inside `couple` are innocent -- the measured residual history
    and the re-filing of a participant's own captured log -- and each needed an
    exemption to stop the gate naming a defect that was not there. A gate that
    names an absent defect costs an action for nothing; a gate widened past its
    defect costs the whole constraint. So assert both directions here, on the
    exact shapes, rather than trusting the prose above.
    """
    cases = [
        # (source, exempt?)
        ('(Path(work_dir) / "participant_A.py").write_text(src.read_text())', False),
        ('(Path(work_dir) / "deck.4C.yaml").write_text(deck)', False),
        ('(Path(w) / "solver.py").write_text("import dolfinx\\n")', False),
        ('(Path(w) / "run_level1.log").write_text("NDOF = 693")', False),
        ('(Path(w) / f"participant_output_level{k}.log").write_text('
         '_src.read_text(errors="replace"))', True),
        ('tmp.write_text("iteration,interface_residual\\n" + rows)', True),
    ]
    for code, want in cases:
        call = ast.parse(code).body[0].value
        got = _writes_measured_csv(call) or _copies_a_captured_log(call)
        assert got == want, (
            f"the exemptions read {code!r} as "
            f"{'innocent' if got else 'a violation'}; expected "
            f"{'innocent' if want else 'a violation'}")
