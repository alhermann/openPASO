"""The product CLI must be able to write a file and run a command.

`run_agent.py` passed the MCP solver tools and nothing else -- no `write_file`,
no `run_bash`. Measured on a coupled task: with no filesystem the model used
`run_simulation` as a substitute shell, and because every call creates a fresh
timestamped output directory it rebuilt its participant tree EIGHTEEN times in
eighteen places, calling couple() fifteen times because it never had a stable
tree to couple. Against the evaluation harness's own successful cell:

    run_agent.py      write_file  0   run_bash  0   run_simulation 23  couple 15
    campaign CORRECT  write_file 20   run_bash 20   run_simulation  0  couple  3

So any task needing more than one file was quietly impossible, and the failure
looked like a failure of the task.

These tests run the tools rather than inspecting the wiring, and they run them
WITHOUT bubblewrap, because the product path deliberately does not isolate --
see `_sandboxed_process_argv`'s docstring for why, and for what is given up.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


@pytest.fixture(scope="module")
def factories():
    from langgraph_eval.agent import _bash_tool_for, _read_write_tools_for
    return _bash_tool_for, _read_write_tools_for


def test_the_shell_runs_without_bubblewrap(factories, tmp_path):
    bash_for, _ = factories
    bash = bash_for(tmp_path, isolate=False, budget_note=False)
    out = bash.invoke({"command": "echo reached && pwd"})
    assert "reached" in out
    assert str(tmp_path) in out, "the shell does not start in the working directory"


def test_the_shell_carries_no_campaign_budget_stamp(factories, tmp_path):
    """`[actions spent: N]` is the harness's wall clock. A product run has none."""
    bash_for, _ = factories
    bash = bash_for(tmp_path, isolate=False, budget_note=False)
    out = bash.invoke({"command": "echo plain"})
    assert "actions spent" not in out, out[-300:]
    assert "clock:" not in out, out[-300:]


def test_a_file_written_by_the_model_is_on_disk_and_readable_by_the_shell(
        factories, tmp_path):
    """The whole point: work must survive from one tool call to the next."""
    bash_for, rw_for = factories
    read_file, write_file = rw_for(tmp_path)
    write_file.invoke({"path": "side_A/participant.py",
                       "content": "print('it persisted')\n"})
    assert (tmp_path / "side_A" / "participant.py").is_file()
    assert "it persisted" in read_file.invoke({"path": "side_A/participant.py"})
    bash = bash_for(tmp_path, isolate=False, budget_note=False)
    out = bash.invoke({"command": f"{sys.executable} side_A/participant.py"})
    assert "it persisted" in out, out[-500:]


def test_writes_stay_inside_the_working_directory(factories, tmp_path):
    """The shell cannot be confined by a path check; these two are."""
    _, rw_for = factories
    read_file, write_file = rw_for(tmp_path)
    outside = Path(tempfile.gettempdir()) / "openpaso_escape_probe.txt"
    reply = write_file.invoke({"path": str(outside), "content": "nope"})
    assert not outside.exists(), "write_file escaped the working directory"
    assert "outside" in reply.lower() or "refus" in reply.lower(), reply[:200]


def test_the_campaign_path_still_demands_isolation():
    """The product's relaxation must not reach the evaluation harness."""
    import inspect

    from langgraph_eval import agent as la
    src = inspect.getsource(la._sandboxed_process_argv)
    assert "isolate: bool = True" in src, (
        "isolation must stay the DEFAULT; only the product path opts out")
    assert "bubblewrap is required" in src


def test_run_agent_asks_for_the_tools(tmp_path):
    """A structural check that the CLI actually wires them in."""
    text = (REPO / "run_agent.py").read_text()
    assert "_bash_tool_for(workdir, isolate=False" in text
    assert "_read_write_tools_for(workdir)" in text
    assert "cleanup_sandbox_scratch(workdir)" in text, (
        "scratch is a deterministic digest of the workdir; without cleanup it "
        "persists between runs")
    # The NAME may appear in a comment explaining why it is not used; what
    # must not appear is a CALL.
    assert "_host_tools(" not in text, (
        "_host_tools() drags in spawn_subagent, which resolves a localhost vLLM")


def test_the_cli_still_starts():
    """--help must work: the wiring is inside run(), not import time."""
    done = subprocess.run([sys.executable, str(REPO / "run_agent.py"), "--help"],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr[-800:]
    assert "--workdir" in done.stdout


# ── Telling a run that never started from one that stopped partway ───────────

def test_a_mid_run_failure_is_not_reported_as_a_startup_failure():
    """Both of these were measured saying "openPASO cannot start".

    A coupled task hit the model's own context limit after 104 steps, having
    written every deliverable; a step trial reached its step budget after 4C had
    already produced solver output. Both told the reader the server had not
    started, which sends them to debug the wrong end of the system.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("ra", REPO / "run_agent.py")
    ra = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ra)

    budget, _ = ra._diagnose(RecursionError(
        "Recursion limit of 45 reached without hitting a stop condition"))
    assert "step budget" in budget, budget

    context, fix = ra._diagnose(ValueError(
        "This model's maximum context length is 262144 tokens"))
    assert "context" in context, context
    assert "http" in fix, "a context failure should name where to get a bigger model"

    # and the startup failures must still read as startup failures
    key, _ = ra._diagnose(RuntimeError("401 no auth"))
    assert "key" in key.lower(), key


def test_the_headline_distinguishes_the_two():
    text = (REPO / "run_agent.py").read_text()
    assert "openPASO stopped partway." in text
    assert "started: bool = False" in text, (
        "the caller must be able to say the run had got somewhere")
    assert "started=True" in text, "the catch-all knows the session was entered"
