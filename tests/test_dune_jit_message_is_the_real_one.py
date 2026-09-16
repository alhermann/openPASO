"""What DUNE actually prints when it compiles, checked by making it compile.

THIS TEST REPLACES ONE THAT WAS WRONG, AND THE WAY IT WAS WRONG IS THE POINT.
The earlier version asserted that the prefix `DUNE-INFO:` appears nowhere,
because `grep -rl DUNE-INFO` over an installed dune-fem finds nothing and the
message is built at dune/generator/cmakebuilder.py as
`f"Compiling {pythonName} (new)"`. On that evidence the prefix was stripped from
thirteen files.

The evidence was a proxy. The literal is never written down because it is
ASSEMBLED AT RUNTIME: dune/common/__init__.py does

    logformat = os.environ.get('DUNE_LOG_FORMAT', 'DUNE-%(levelname)s: %(message)s')
    logging.basicConfig(format=logformat, level=loglevel)

so at INFO level a reader sees exactly

    DUNE-INFO: Compiling Scheme (new)

which is what openPASO had documented all along. Grepping the source answered a
different question from the one that mattered, and only running a build showed
it. So this test compiles a form and reads the output.

What it pins, and deliberately not more: the BODY (`Compiling ... (new)`) is
DUNE's own and is what a matcher should key on, because DUNE_LOG_FORMAT can
replace the prefix; and the DEFAULT format does carry `DUNE-INFO`, so text
quoting the whole line is not inventing it.
"""
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_FORM = '''
from dune.alugrid import aluConformGrid
from dune.fem.scheme import galerkin
from dune.fem.space import dglagrange
from dune.grid import cartesianDomain
from dune.ufl import DuneCellDiameter as CellDiameter
from ufl import TestFunction, TrialFunction, dx, grad, inner
grid = aluConformGrid(cartesianDomain([0, 0], [1, 1], [3, 3]), dimgrid=2)
space = dglagrange(grid, order=1)
u, v = TrialFunction(space), TestFunction(space)
h = CellDiameter(space)
galerkin([inner(grad(u), grad(v)) * dx + (__MARK__ / h) * u * v * dx == v * dx],
         solver="gmres")
print("BUILT")
'''


@pytest.fixture(scope="module")
def compile_output():
    """Force ONE cold JIT build and return everything it printed."""
    import random

    from backends.dune.backend import _find_dune_python
    python = _find_dune_python()
    if not python:
        pytest.skip("no interpreter on this machine can run dune")
    body = _FORM.replace("__MARK__", f"{random.random():.12f}")
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "child.py"
        script.write_text(body)
        done = subprocess.run([str(python), str(script)], cwd=tmp,
                              capture_output=True, text=True, timeout=1800)
    blob = done.stdout + done.stderr
    if "BUILT" not in blob:
        pytest.skip(f"the probe form did not build here:\n{blob[-800:]}")
    return blob


def test_the_body_is_dunes_own(compile_output):
    """`Compiling <module> (new)` -- the half that survives any log format."""
    assert re.search(r"Compiling .+ \(new\)", compile_output), (
        "DUNE no longer announces a cold build as `Compiling <module> (new)`; "
        "whatever it prints now is what openPASO should quote.\n"
        + compile_output[-1200:])


def test_the_default_format_really_does_prefix_dune_info(compile_output):
    """The claim the earlier test denied, now taken from the output itself."""
    if os.environ.get("DUNE_LOG_FORMAT"):
        pytest.skip("DUNE_LOG_FORMAT is set here, so the default is not in play")
    assert re.search(r"DUNE-INFO: Compiling .+ \(new\)", compile_output), (
        "the default DUNE_LOG_FORMAT no longer renders `DUNE-INFO: `; openPASO's "
        "served text quotes the whole line, so it would now be quoting something "
        "a reader never sees.\n" + compile_output[-1200:])


def test_the_prefix_is_a_default_not_a_guarantee():
    """Which is why guidance should tell a reader to match the body.

    Read from the installed package rather than remembered: the prefix comes
    from an environment variable with a default, so a user who sets
    DUNE_LOG_FORMAT sees a different line and a matcher keyed to the prefix
    alone would stop working for them.
    """
    from backends.dune.backend import _find_dune_python

    python = _find_dune_python()
    if not python:
        pytest.skip("no interpreter on this machine can run dune")
    probe = ("import dune.common, inspect, pathlib;"
             "print(pathlib.Path(inspect.getfile(dune.common)).read_text())")
    done = subprocess.run([str(python), "-c", probe], stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=120)
    if done.returncode != 0:
        pytest.skip("dune.common is not readable here")
    assert "DUNE_LOG_FORMAT" in done.stdout, (
        "the log format is no longer taken from DUNE_LOG_FORMAT")
    assert "DUNE-%(levelname)s: %(message)s" in done.stdout, (
        "the DEFAULT log format changed; the served text quoting "
        "`DUNE-INFO: Compiling ... (new)` needs re-deriving")
