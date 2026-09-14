"""A timed-out shell command must not leave its solver running.

`run_bash` used to be `subprocess.run(["bash","-lc",cmd], timeout=900)`. On
timeout Python kills its DIRECT child — the shell — and every grandchild
survives, reparented to init. Three DUNE processes launched by round-1 cells
that timed out were found 23-24 hours later still at 100% CPU, having burned
71 CPU-hours between them; they also compete for the machine with whatever
runs next, so one cell's runaway can slow, and eventually time out, another.

The tool now starts the shell in its own session, so the shell and everything
it spawns share a process group that can be signalled as a unit.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "langgraph_eval"))

import agent as A  # noqa: E402

MARKER = "openpaso_pgroup_test_grandchild"


def _alive() -> int:
    r = subprocess.run(["pgrep", "-fc", MARKER], capture_output=True, text=True)
    try:
        return int(r.stdout.strip() or 0)
    except ValueError:
        return 0


def test_kill_group_reaps_the_grandchild(tmp_path):
    # a shell whose CHILD is a long-lived python process, then waits itself
    script = (f"python3 -c \"import time,sys; sys.argv[0]='{MARKER}'; "
              f"time.sleep(300)\" & sleep 300")
    proc = subprocess.Popen(["bash", "-lc", script], cwd=tmp_path,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        deadline = time.time() + 15
        while _alive() == 0 and time.time() < deadline:
            time.sleep(0.3)
        assert _alive() >= 1, "the grandchild never started; test is invalid"

        A._kill_group(proc)

        deadline = time.time() + 15
        while _alive() > 0 and time.time() < deadline:
            time.sleep(0.3)
        assert _alive() == 0, (
            "the grandchild survived the group kill — a timed-out cell would "
            "leave its solver running")
    finally:
        # belt and braces: never leave the test's own processes behind
        try:
            os.killpg(os.getpgid(proc.pid), 9)
        except OSError:
            pass
        subprocess.run(["pkill", "-9", "-f", MARKER], capture_output=True)


def test_bash_tool_starts_a_new_session(tmp_path):
    """The property the kill depends on: the shell leads its own group."""
    tool = A._bash_tool_for(tmp_path)
    out = tool.invoke({"command": "ps -o pgid= -p $$ | tr -d ' '; echo :; "
                                  "ps -o pgid= -p $PPID | tr -d ' '"})
    lines = [ln for ln in out.splitlines() if ln.strip()]
    shell_pgid = lines[0].strip()
    assert shell_pgid and shell_pgid != str(os.getpgid(0)), (
        "the shell shares the runner's process group, so killing it would "
        "signal the runner too and the group kill is unusable")
