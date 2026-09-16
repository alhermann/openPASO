"""A name read where the reply is assembled must be bound before every path reaches it.

MEASURED, in a live round. `_pde_verdicts` was initialised inside the
CONVERGED branch and read where the reply is assembled, which every path
reaches. The first non-converged couple() call therefore returned

    Error executing tool couple: cannot access local variable
    '_pde_verdicts' where it is not associated with a value

and the cell that hit it called couple twice, got that twice, abandoned the
tool and hand-rolled its own coupling driver. The regression suite did not
catch it because the tests written alongside that change asserted on SOURCE
TEXT rather than executing the path.

This walks the AST instead: for the couple() body, every local name that is
read at the function's own statement level must also be assigned at that
level, not only inside some branch.
"""
import ast
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "tools" / "consolidated.py"


def _couple_fn():
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "couple":
            return node
    raise AssertionError("couple() not found")


def _top_level_names(fn):
    """(read_at_top, assigned_at_top) for the function's own statement list."""
    read, assigned = set(), set()
    for stmt in fn.body:
        for n in ast.walk(stmt):
            if isinstance(n, ast.Name):
                if isinstance(n.ctx, ast.Store):
                    # assigned anywhere under a top-level statement still only
                    # counts as unconditional if that statement is not a branch
                    pass
                continue
        if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgts = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            for t in tgts:
                if isinstance(t, ast.Name):
                    assigned.add(t.id)
        elif isinstance(stmt, (ast.If, ast.Try, ast.For, ast.While, ast.With)):
            for n in ast.walk(stmt):
                if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                    read.add(n.id)
    return read, assigned


PDE_LOCALS = ("_pde_verdicts", "_pde_notes")


@pytest.mark.parametrize("name", PDE_LOCALS)
def test_the_reply_locals_are_bound_at_function_scope(name):
    fn = _couple_fn()
    _, assigned = _top_level_names(fn)
    assert name in assigned, (
        f"{name} is read where the reply is assembled but never assigned at the "
        f"function's own statement level -- a non-converged call will raise "
        f"UnboundLocalError, which is exactly what happened in round 64")


@pytest.mark.parametrize("name", PDE_LOCALS)
def test_the_binding_comes_before_the_read(name):
    """Order matters as much as presence: bound late is still unbound early."""
    src = SRC.read_text()
    bind = src.find(f"\n        {name}")
    read = src.find("\n        if _pde_verdicts or _pde_notes:")
    assert bind != -1, f"{name} has no function-scope binding"
    assert read != -1, "the reply-assembly read moved; update this test"
    assert bind < read, f"{name} is bound after it is read"


def test_the_module_still_compiles():
    ast.parse(SRC.read_text())
