"""The isolated shell must be able to start the interpreter that built it.

WHY THIS EXISTS. `_sandboxed_process_argv` puts a tmpfs over $HOME so a cell
cannot read the user's credentials, other checkouts, or past runs, and then
rebinds a fixed list of solver runtime roots. openPASO's own virtual
environment was not on that list -- the list began with a sibling directory
named after this machine's older checkout. So a clone made ANYWHERE UNDER
$HOME, which is exactly what the README tells a reader to make, had its own
`.venv` hidden by that tmpfs, and `run_agent.py` died before the first model
call with:

    bwrap: execvp .../.venv/bin/python: No such file or directory
    openPASO cannot start.
    Problem: McpError: Connection closed

Neither line names the cause. It stayed invisible because a clone OUTSIDE
$HOME -- in /tmp, say -- is not hidden and works.

This test RUNS the sandbox rather than reading its arguments, because the
failure is an exec that argument inspection cannot see: the interpreter is
reached through two symlinks, so a bind can be present and still resolve to
nothing.
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def sandboxed():
    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this machine")
    from langgraph_eval.agent import _sandboxed_process_argv
    return _sandboxed_process_argv


def test_the_sandbox_can_execute_the_interpreter_that_built_it(sandboxed, tmp_path):
    argv = sandboxed(tmp_path, [sys.executable, "-c", "print('reached')"])
    done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, (
        "the isolated shell cannot start openPASO's own Python, so run_agent.py "
        f"would die with 'McpError: Connection closed'.\n{done.stderr[-1500:]}")
    assert "reached" in done.stdout


def test_home_is_still_hidden(sandboxed, tmp_path):
    """The mount that fixes the above must not undo what the tmpfs is for."""
    probe = "import os;print(sorted(os.listdir(os.path.expanduser('~')))[:40])"
    argv = sandboxed(tmp_path, [sys.executable, "-c", probe])
    done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-1500:]
    # HOME is redirected to the cell's own work dir, so the user's real home
    # contents must not be visible through it.
    for private in (".ssh", ".bash_history", ".gitconfig", "Schreibtisch"):
        assert private not in done.stdout, (
            f"{private} is visible inside the sandbox; the tmpfs over $HOME is "
            "no longer doing its job")
