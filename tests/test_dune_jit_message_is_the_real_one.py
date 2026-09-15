"""The JIT compile line openPASO tells agents to watch for must be DUNE's own.

Thirteen files -- served knowledge, generator guidance, the examples corpus and
a tier-2 fixture's own docstring -- told a reader to look for

    DUNE-INFO: Compiling <module> (new)

That prefix exists nowhere in DUNE. `grep -rl DUNE-INFO` over an installed
dune-fem finds nothing, and the message is built at
dune/generator/cmakebuilder.py as

    compilationInfoMessage = f"Compiling {pythonName} (new)"

with sibling forms "(updated)", "(rebuilding)" and "(rebuilding after
concurrent build)". A symptom search for the invented prefix returns nothing,
so the guidance was unreachable exactly when it was needed: a first DUNE run
compiles for minutes, and the whole point of naming the line is to stop someone
killing it.

This is the same defect class as an NGSolve trap that said "Signal, and it is
one line:" -- guidance keyed to a string the reader will never see.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

SEARCHED = ("--include=*.py", "--include=*.json", "--include=*.md")


def test_the_invented_prefix_is_gone():
    done = subprocess.run(["grep", "-rn", "DUNE-INFO", *SEARCHED, "."],
                          cwd=REPO, capture_output=True, text=True)
    hits = [ln for ln in done.stdout.splitlines() if "/.venv/" not in ln
            and Path(__file__).name not in ln]
    assert not hits, (
        "`DUNE-INFO` is back. DUNE never prints it; the line is "
        '`Compiling <module> (new)`. Offenders:\n  ' + "\n  ".join(hits[:15]))


def test_the_message_we_do_quote_is_dunes_own():
    """Checked against the installed DUNE, not against a memory of it."""
    sys.path.insert(0, str(REPO / "src"))
    from backends.dune.backend import _find_dune_python

    python = _find_dune_python()
    if not python:
        pytest.skip("no interpreter on this machine can run dune")
    probe = ("import os, dune.generator as g;"
             "print(os.path.dirname(g.__file__))")
    where = subprocess.run([str(python), "-c", probe], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
    if where.returncode != 0:
        pytest.skip("dune.generator is not importable here")
    builder = Path(where.stdout.strip()) / "cmakebuilder.py"
    if not builder.is_file():
        pytest.skip("dune's cmakebuilder.py is not where it used to be")
    text = builder.read_text(errors="replace")
    assert re.search(r'f"Compiling \{pythonName\} \(new\)"', text), (
        "DUNE no longer builds its compile line as `Compiling <name> (new)`. "
        "Whatever it prints now is what openPASO should quote.")
