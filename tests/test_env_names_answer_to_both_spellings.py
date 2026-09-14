"""The rename must not break a process at the far end of an environment variable.

openPASO writes its variables into the environment of a solver participant it
launches, and that participant may be a script written against the old OASIS_*
names. The reverse happens too: an old driver exports OASIS_BLIND_KEYS and the
current code reads OPENPASO_BLIND_KEYS. Both directions have to work, and a
value set explicitly has to win over one inferred from the other spelling --
otherwise the alias would quietly overwrite a deliberate choice.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from core.env_compat import get_env, install_aliases, set_env  # noqa: E402


def test_the_old_name_reaches_code_that_reads_the_new_one():
    env = {"OASIS_CONFIG_JSON": '{"level": 2}'}
    install_aliases(env)
    assert env["OPENPASO_CONFIG_JSON"] == '{"level": 2}'


def test_the_new_name_reaches_a_script_that_reads_the_old_one():
    env = {"OPENPASO_BLIND_KEYS": "/keys"}
    install_aliases(env)
    assert env["OASIS_BLIND_KEYS"] == "/keys"


def test_an_explicitly_set_value_is_not_overwritten_by_its_twin():
    env = {"OASIS_REPO": "/old", "OPENPASO_REPO": "/new"}
    install_aliases(env)
    assert env["OASIS_REPO"] == "/old"
    assert env["OPENPASO_REPO"] == "/new"


def test_variables_that_are_not_ours_are_left_alone():
    env = {"PATH": "/usr/bin", "OPENROUTER_API_KEY": "secret"}
    install_aliases(env)
    assert env == {"PATH": "/usr/bin", "OPENROUTER_API_KEY": "secret"}


def test_calling_it_twice_changes_nothing_the_second_time():
    env = {"OASIS_LEVEL": "3"}
    first = install_aliases(env)
    second = install_aliases(env)
    assert first == ["OPENPASO_LEVEL"]
    assert second == []


def test_set_env_writes_both_spellings_and_get_env_reads_either():
    env = {}
    set_env("OPENPASO_MESH_DIR", "/mesh", env)
    assert env["OASIS_MESH_DIR"] == "/mesh"
    assert get_env("OASIS_MESH_DIR", environ=env) == "/mesh"
    assert get_env("OPENPASO_MESH_DIR", environ=env) == "/mesh"
    assert get_env("OPENPASO_ABSENT", "fallback", environ=env) == "fallback"


def test_the_server_installs_the_aliases_before_it_reads_its_own_guard():
    """server.py reads OPENPASO_NO_FD_GUARD at import; the old name must work."""
    env = dict(os.environ, OASIS_NO_FD_GUARD="1", PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r);"
         "import server, os;"
         "print(os.environ.get('OPENPASO_NO_FD_GUARD'))" % str(REPO / "src")],
        capture_output=True, text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert proc.stdout.strip().splitlines()[-1] == "1"
