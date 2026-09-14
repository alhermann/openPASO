"""Where the solvers live on THIS machine, without writing down whose machine.

The developer scripts and fixtures under scripts/ and benchmarks/ grep real
solver source trees and run real binaries. Those paths are load-bearing: the
named-key and quoted-diagnostic gates decide whether openPASO invented an input
key by looking for it in the code, and a corpus that cannot be found answers
"not there" to everything, which reads as fabrication. Two gates failed exactly
that way when the paths were first anonymised by hand.

They also used to carry one person's username, which published it and worked on
exactly one computer. Both problems have the same fix: name the environment
variable openPASO already documents, and fall back to the same location under
$HOME. No username in the source, and the same behaviour on the machine it was
written on.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def root(env_var: str, *under_home: str) -> str:
    """`$env_var` if set, else the given path under the user's home directory."""
    named = os.environ.get(env_var, "").strip()
    return named if named else str(Path.home().joinpath(*under_home))


def fourc_binary() -> str:
    return root("FOURC_BINARY", "4C", "build", "4C")


def fourc_root() -> str:
    return root("FOURC_ROOT", "4C")


def sparta_binary() -> str:
    return root("SPARTA_BINARY", "Schreibtisch", "sparta", "src", "spa_serial")


def fenics_python() -> str:
    return root("FENICS_PYTHON", "miniconda3", "envs", "fenics", "bin", "python")


def openpaso_python() -> str:
    """The interpreter running openPASO.

    This was written out as one checkout's `.venv/bin/python`, which is wrong on
    every other checkout including a fresh clone. `sys.executable` is what the
    scripts actually meant, and it is right everywhere.
    """
    return os.environ.get("OPENPASO_PYTHON", "").strip() or sys.executable


# Tokens a data file may carry instead of one machine's absolute path.
_TOKENS = {
    "${OPENPASO_PYTHON}": openpaso_python,
    "${FOURC_BINARY}": fourc_binary,
    "${FOURC_ROOT}": fourc_root,
    "${FENICS_PYTHON}": fenics_python,
    "${SPARTA_BINARY}": sparta_binary,
}


def expand(value):
    """Resolve host-path tokens in a string read from a data file.

    benchmarks/coupling_pairs/fourc_kratos_cht/params.json used to name one
    machine's interpreter and one machine's 4C binary outright. A parameter file
    that hard-codes somebody's home directory works on exactly one computer and
    publishes their username; a token resolves on whichever computer reads it.
    Anything that is not a known token is returned unchanged.
    """
    if not isinstance(value, str):
        return value
    for token, resolve in _TOKENS.items():
        if token in value:
            value = value.replace(token, str(resolve()))
    return value
