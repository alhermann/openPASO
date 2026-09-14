"""A killed grading process must still leave the answer keys sealed.

grade_round.py's docstring promised that "a crash or a KeyboardInterrupt cannot
leave them readable", and the mechanism was a `finally` block. A finally block
does not run when the process is killed -- Python's default SIGTERM
disposition terminates immediately -- so the promise held for exceptions and
not for the case that actually happens.

It happened on 2026-08-30: a regrade of seeds 2-11 was launched under a
two-minute harness limit, the limit killed it with SIGTERM part-way through,
the finally block never ran, and the vault was found drwxr-xr-x. Nothing could
read it (the round had finished; no agent process existed) but the custody
claim was false for about two minutes, and nothing announced it.

This test starts a real process that uses the repository shield against a
throwaway vault, kills it the way the harness did, and looks at the directory's
mode afterwards. The real vault is never touched.
"""

from __future__ import annotations

import os
import signal
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GRADE_ROUND = REPO / "campaign3_blind" / "grade_round.py"

VICTIM = """
import os, sys, time
sys.path.insert(0, {gr!r})
import grade_round as GR
GR.arm_reseal_on_signals()
print(GR.shield("unseal"), flush=True)
time.sleep(120)
"""


def _mode(p: Path) -> str:
    return stat.filemode(p.stat().st_mode)


@pytest.fixture
def vault(tmp_path):
    keys = tmp_path / "keys"
    keys.mkdir()
    (keys / "T1").mkdir()
    (keys / "T1" / "key.json").write_text("{}")
    keys.chmod(0o000)
    yield keys
    # pytest cannot remove a chmod-000 directory; leave the tree deletable
    keys.chmod(0o755)


def _start_victim(vault: Path) -> subprocess.Popen:
    env = dict(os.environ, OPENPASO_BLIND_KEYS=str(vault))
    src = VICTIM.format(gr=str(GRADE_ROUND.parent))
    p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(src)],
                         env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True)
    # wait for it to report the unseal, so the kill lands inside the window
    deadline = time.time() + 30
    while time.time() < deadline:
        line = p.stdout.readline()
        if "UNSEALED" in line:
            return p
        if p.poll() is not None:
            raise AssertionError(f"victim died before unsealing: {line}")
    raise AssertionError("victim never reported an unseal")


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP])
def test_the_vault_is_sealed_again_after_a_kill(vault, sig):
    """The case the finally block could not reach."""
    p = _start_victim(vault)
    assert _mode(vault) == "drwx------", "the victim did not actually unseal"
    p.send_signal(sig)
    p.wait(timeout=30)
    assert _mode(vault) == "d---------", (
        f"after {sig!r} the vault is {_mode(vault)}: a killed grading process "
        f"left the answer keys readable, which is the custody claim being "
        f"false without anyone being told")


def test_the_kill_is_still_reported_as_a_kill(vault):
    """Resealing must not disguise the death as a clean exit.

    A handler that swallows the signal would leave a zero status, and a
    watchdog reading exit codes would call an interrupted grading pass
    successful.
    """
    p = _start_victim(vault)
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=30)
    assert p.returncode != 0, (
        f"exit status {p.returncode}: a killed process reported success")
    assert _mode(vault) == "d---------"


def test_the_guard_is_armed_before_the_keys_are_opened(vault):
    """Arming after the unseal would leave a window with no handler."""
    text = GRADE_ROUND.read_text()
    arm = text.index("arm_reseal_on_signals()", text.index("def main"))
    unseal = text.index('shield("unseal")')
    assert arm < unseal, (
        "the signal handlers are installed after the vault is opened, so a "
        "kill in between still leaves the keys readable")
