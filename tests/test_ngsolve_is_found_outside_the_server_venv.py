"""NGSolve must be findable where it actually is, and verified before use.

`_find_ngsolve_python` used to return `sys.executable` and nothing else, which
made NGSolve the one backend that could not be pointed anywhere. Measured on a
machine with ngsolve installed in two separate interpreters, openPASO reported
"No Python with ngsolve found", and an agent handed an NGSolve task spent its
run trying to install what was already on disk.

Both callers share this function -- check_availability() and run() -- so the
interpreter that answers "is it here?" is the one that does the work. That
property is the point of the test: DUNE shipped the opposite once (issue #40)
and a conda env found by the check was ignored at execution time.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

PROBE = ("from netgen.geom2d import unit_square\n"
         "from ngsolve import Mesh, H1\n"
         "H1(Mesh(unit_square.GenerateMesh(maxh=0.5)), order=1)\n"
         "print('OK')\n")


@pytest.fixture
def finder():
    from backends.ngsolve import backend
    backend._NGSOLVE_PYTHON_CACHE.clear()
    yield backend
    backend._NGSOLVE_PYTHON_CACHE.clear()


def _really_runs(python) -> bool:
    done = subprocess.run([str(python), "-c", PROBE], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=120)
    return done.returncode == 0 and "OK" in done.stdout


def test_what_it_returns_can_actually_run_ngsolve(finder):
    python = finder._find_ngsolve_python()
    if python is None:
        pytest.skip("no interpreter on this machine can run ngsolve")
    assert _really_runs(python), (
        f"{python} was offered as the NGSolve interpreter but cannot build a "
        "mesh and a space with it")


def test_the_check_and_the_run_agree_on_one_interpreter(finder):
    """The defect DUNE shipped once: the check finds it, run() ignores it."""
    import inspect
    from backends.ngsolve.backend import NgsolveBackend
    for method in (NgsolveBackend.check_availability, NgsolveBackend.run):
        src = inspect.getsource(method)
        assert "_find_ngsolve_python" in src, (
            f"{method.__name__} picks its interpreter some other way, so it can "
            "disagree with the other one")
        assert "sys.executable" not in src, (
            f"{method.__name__} falls back to the server's own interpreter")


def test_an_explicit_override_is_verified_not_trusted(finder, monkeypatch):
    """NGSOLVE_PYTHON naming an interpreter without ngsolve must be rejected."""
    bare = sys.executable
    if _really_runs(bare):
        pytest.skip("this interpreter can run ngsolve, so it is not a negative case")
    monkeypatch.setenv("NGSOLVE_PYTHON", bare)
    found = finder._find_ngsolve_python()
    assert found is None or str(found) != bare, (
        "an NGSOLVE_PYTHON that cannot import ngsolve was accepted on trust")


def test_it_resolves_inside_the_isolated_shell(finder, tmp_path):
    """The case that actually bit: HOME is not HOME inside the sandbox.

    `_sandboxed_process_argv` sets HOME to the cell's own work directory, so a
    conda scan under `Path.home()` searches an empty tree. Measured: a run with
    ngsolve present on the machine spent its entire budget calling
    setup_backend(action='install') and pip-installing what was already there.
    The discovered-backends config holds an absolute path that survives the
    tmpfs, which is why it is consulted before any scan.
    """
    import shutil
    import subprocess as sp

    if shutil.which("bwrap") is None:
        pytest.skip("bubblewrap is not installed on this machine")
    if finder._find_ngsolve_python() is None:
        pytest.skip("no interpreter on this machine can run ngsolve")

    from langgraph_eval.agent import _sandboxed_process_argv
    code = ("import sys,logging;logging.disable(logging.INFO);"
            "sys.path.insert(0,'/tmp/openpaso-source/src');"
            "from backends.ngsolve.backend import NgsolveBackend;"
            "print(NgsolveBackend().check_availability()[0].value)")
    argv = _sandboxed_process_argv(tmp_path, [sys.executable, "-c", code],
                                   source_repo=REPO)
    done = sp.run(argv, capture_output=True, text=True, timeout=600)
    assert "available" in done.stdout, (
        "inside the sandbox openPASO cannot see an NGSolve that is installed on "
        f"this machine, so an agent will try to install it.\n{done.stderr[-1200:]}")
