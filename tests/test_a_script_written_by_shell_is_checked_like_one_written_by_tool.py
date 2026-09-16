"""The same script gets the same checks whichever way it reached disk.

MEASURED over the 2095 live trajectories of the honest coupled rounds. The two
checks that name the two commonest blockers ran on the write_file route only:

    4C decks            257 written by write_file, 229 by shell heredoc  (47% unchecked)
    participant scripts 6887 by write_file,        452 by shell heredoc  ( 6% unchecked)

The blockers those checks name are the ones that decide these runs: a
confabulated API symbol appears in 56 of 235 failing cells (ngsolve 32,
dune.ufl 21, ufl 21, skfem 8) and an invented 4C section name in 37 more. The
tetrahedron-orientation check lives behind the same door -- 0 of 18 C10 cells
used the orientation table openPASO serves, every one of them hand-rolled the
construct it warns about, and each then blamed the Kratos element for the
singular system the document predicts.

Both checks gate on the file extension themselves and write nothing, so running
them on the shell route costs a read of a file the route already reads.

This test reads the wiring rather than the behaviour: the shell route must call
every script check the write route calls, and the files it watches must include
the extensions those checks judge. It is deliberately AST-based -- importing
langgraph_eval.agent needs the other venv, and the defect is in which names
appear, not in what they do.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
AGENT = ROOT / "langgraph_eval" / "agent.py"

# The shell side splits its checks across two functions -- one keyed to touched
# scripts, one to touched artefacts -- so the comparison is against their union.
# Which of the two carries a given check does not matter; whether the check runs
# at all does.
_AFTER_SHELL = ("_script_check_after_shell", "_artefact_check_after_shell")

_CHECKS = {
    "_discarded_proof_check",
    "_script_noop_check",
    "_registry_attribute_check",
    "_extra_script_checks",
    "_fourc_deck_write_check",
    "_participant_write_check",
    "_level_index_check",
    "_identical_levels_check",
    "_wrong_level_run_log_check",
    "_early_artefact_check",
}


def _tree():
    return ast.parse(AGENT.read_text())


def _function(name):
    for node in ast.walk(_tree()):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    raise AssertionError(f"{name} is gone from {AGENT.name} — re-point this test")


def _calls(name):
    return {n.func.id for n in ast.walk(_function(name))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def test_the_shell_route_runs_every_check_the_write_route_runs():
    write = _calls("write_file") & _CHECKS
    shell = set().union(*(_calls(f) for f in _AFTER_SHELL)) & _CHECKS
    missing = sorted(write - shell)
    assert not missing, (
        "a script written by shell heredoc escapes these checks entirely: "
        + ", ".join(missing)
        + " — 47% of 4C decks and 6% of participant scripts reach disk that way")


def test_the_shell_route_watches_the_files_those_checks_judge():
    # _fourc_deck_write_check returns "" unless the name ends .yaml/.yml/.dat
    # (workspace_advisor.py:1010), so a route watching only *.py can never
    # hand it a deck no matter which functions it calls.
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "_SCRIPTS"
                for t in node.targets):
            pats = {ast.literal_eval(e) for e in node.value.elts}
            break
    else:
        raise AssertionError("_SCRIPTS is gone — re-point this test")
    for ext in (".yaml", ".yml", ".dat", ".py"):
        assert any(p.endswith(ext) for p in pats), (
            f"the shell route never sees *{ext} files, so the check that "
            f"judges them cannot fire there; watched: {sorted(pats)}")
