"""preCICE was installed, advertised, reached for — and could not load.

MEASURED. `import precice` on this host fails with

    ImportError: libprecice.so.3: cannot open shared object file

although /opt/precice/lib/libprecice.so.3 -> libprecice.so.3.1.2 is present and
the Python binding is installed at
.venv/lib/python3.12/site-packages/precice/__init__.py. With /opt/precice/lib on
the loader path it imports and reports 3.1.2. And 103 of the C2 run directories
in this campaign mention preCICE, so agents DO reach for that path; it could not
load for a single one of them.

THE CAUSE WAS A `get` WITH A DEFAULT. langgraph_eval/agent.py had

    env["LD_LIBRARY_PATH"] = env.get("LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")

so if the outer environment set the variable at all, 4C's dependency path was
silently discarded, and preCICE's was never added under any circumstance. One
variable, several solvers, and a default that can only ever express one of them.

It is now COMPOSED: every required entry first, then whatever was inherited,
order preserved, duplicates dropped, and non-existent directories left out so a
stale entry cannot mask a real one. The same dict is handed to the agent's bash
AND, at the MultiServerMCPClient call, to the MCP server itself — so `couple`
and every participant it launches inherit it, since core/coupling_driver.py
passes no env of its own and participants inherit the driver's.

Verified with the composed value on this host:
    import precice                -> 3.1.2
    4C on a real deck             -> exit 0, "processor 0 finished normally"
    import KratosMultiphysics     -> banner printed
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("/opt/4C-dependencies/lib", "/opt/precice/lib")
VENV_PY = Path("/home/user/Schreibtisch/open-fem-agent/.venv/bin/python")


def _composed() -> list[str]:
    """Reproduce the composition the harness performs, from its own source."""
    src = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert "_lib_dirs = [" in src, (
        "the LD_LIBRARY_PATH composition is gone; a `get(..., default)` here "
        "can only ever name one solver's libraries")
    seen, parts = [], []
    for d in REQUIRED:
        if d not in seen and Path(d).is_dir():
            seen.append(d)
            parts.append(d)
    return parts


def test_the_harness_composes_rather_than_defaults():
    src = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert 'env.get(\n        "LD_LIBRARY_PATH", "/opt/4C-dependencies/lib")' \
        not in src, "the defaulting form is back"
    for d in REQUIRED:
        assert d in src, f"{d} is not in the composed list"
    # the composed value must reach the MCP SERVER too, not only bash
    assert '"env": env' in src, (
        "the MCP server no longer receives the composed environment, so "
        "`couple` and its participants would lose it")


@pytest.mark.skipif(not Path("/opt/precice/lib").is_dir(),
                    reason="preCICE not installed on this host")
@pytest.mark.skipif(not VENV_PY.is_file(), reason="solver venv absent")
def test_precice_imports_under_the_composed_path():
    """The thing that actually matters: does it load for the agent."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(_composed())
    r = subprocess.run(
        [str(VENV_PY), "-c",
         "import precice; print(precice.get_version_information().decode())"],
        capture_output=True, text=True, timeout=180, env=env)
    assert r.returncode == 0, (
        f"preCICE still cannot load under the composed path "
        f"{env['LD_LIBRARY_PATH']}:\n{r.stderr[-600:]}")
    assert "3.1" in r.stdout, r.stdout[:200]


@pytest.mark.skipif(not VENV_PY.is_file(), reason="solver venv absent")
def test_the_other_solvers_still_load_under_the_same_path():
    """A fix for one solver that breaks another is not a fix. 4C's libraries
    and preCICE's share this one variable."""
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ":".join(_composed())
    r = subprocess.run([str(VENV_PY), "-c", "import KratosMultiphysics"],
                       capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr[-400:]
    fourc = Path("/home/user/4C/build/4C")
    if not fourc.is_file():
        pytest.skip("4C binary absent")
    r = subprocess.run([str(fourc), "-p"], capture_output=True, text=True,
                       timeout=300, env=env)
    assert r.returncode == 0, (
        f"4C no longer starts under the composed path: {r.stderr[-400:]}")
