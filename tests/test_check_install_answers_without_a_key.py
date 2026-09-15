"""The install check must answer with no API key, and must never echo one.

Someone following the README for the first time installed everything, reached
"Check that it works", and could not find out whether openPASO had seen the
solver they had just installed: every way of listing solvers went through a
model, which costs money or needs an AI app. `check_install.py` is the answer
to that, so the two properties that make it an answer are pinned here -- it
works with no key, and it does not leak a key it happens to find.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "check_install.py"

FAKE_KEY = "sk-or-v1-0123456789abcdefTHISMUSTNOTBEPRINTED"


def _run(env_extra: dict) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("OPENROUTER_API_KEY", None)
    env.update(env_extra)
    return subprocess.run([sys.executable, str(SCRIPT)], cwd=REPO, env=env,
                          capture_output=True, text=True, timeout=900)


def test_it_answers_with_no_key_at_all():
    done = _run({})
    assert done.returncode in (0, 1), done.stderr[-2000:]
    assert "Solvers openPASO can see on this machine" in done.stdout
    # It must say something definite about the install, not merely not crash.
    assert ("Install is working" in done.stdout
            or "No solver is usable" in done.stdout), done.stdout[-2000:]


def test_it_never_prints_the_key_it_finds():
    done = _run({"OPENROUTER_API_KEY": FAKE_KEY})
    everything = done.stdout + done.stderr
    assert FAKE_KEY not in everything, "check_install.py echoed the API key"
    assert "0123456789abcdef" not in everything, (
        "check_install.py printed part of the API key; no part of it may appear")
    assert "key is set" in done.stdout, (
        "it must still confirm that a key was found, without showing it")


def test_it_names_a_command_for_every_solver_it_reports_missing():
    """A red line with no fix is the thing the README was already criticised for."""
    done = _run({})
    missing = [line for line in done.stdout.splitlines()
               if "not installed" in line]
    assert missing, "expected at least one uninstalled solver on a test machine"
    for line in missing:
        assert "to get it:" in line, f"no way out offered: {line.strip()}"
