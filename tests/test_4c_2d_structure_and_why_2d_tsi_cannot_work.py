"""Two 4C facts that ended runs, both measured here rather than assumed.

C1 — 2-D steady thermoelasticity with 4C on one subdomain — was attempted twice
and reported COULD_NOT_COMPLETE both times, the agent saying 4C's TSI input
format "requires specific section names and element types that differ from
standalone thermal or structural problems". It was right, and openPASO served
nothing about it.

FACT 1. `SOLID` is the THREE-dimensional continuum element. The 2-D one is
`WALL`. Writing `SOLID QUAD4` fails with "Element 'SOLID' does not seem to know
cell type 'quad4'". From 4C's own grammar dump on this build:

    SOLID    HEX8 HEX18 HEX20 HEX27 TET4 TET10 WEDGE6 PYRAMID5 NURBS27
    WALL     QUAD4 QUAD8 QUAD9 TRI3 TRI6 NURBS4 NURBS9

FACT 2. 2-D Thermo_Structure_Interaction cannot work in this build, and the
error does not say so — it says "Unsupported solid element type!". The reason is
in the source: TSI::Utils::ThermoStructureCloneStrategy::set_element_data
accepts ONLY a SolidScatra element and throws otherwise, and SOLIDSCATRA's cell
types are HEX8, HEX27, TET4, TET10, NURBS27 — all three-dimensional. So the
clone step can never succeed in 2-D, whatever else the deck says.

I nearly asserted something false along the way: after seeing SOLID reject
quad4 I was about to conclude 4C has no 2-D solid element at all. Checking the
grammar first showed WALL, which does. The fixture below RUNS.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
import backend_probe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FIX = ROOT / "tests" / "fixtures" / "fourc_2d"
FOURC = backend_probe.fourc_binary()
TSI_SRC = Path(str(backend_probe.fourc_root()) + "/src/tsi/4C_tsi_utils.cpp")


def _grammar():
    from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR
    return re.sub(r"\s+", " ", FOURC_DECK_GRAMMAR)


def test_the_grammar_names_wall_as_the_2d_element():
    g = _grammar()
    assert "WALL QUAD4" in g
    assert "does not seem to know cell type 'quad4'" in g, (
        "the exact error must be quoted, so an agent that hits it can search"
    )


def test_the_grammar_explains_why_2d_tsi_is_impossible():
    g = _grammar()
    assert "2-D THERMO_STRUCTURE_INTERACTION IS NOT AVAILABLE" in g
    assert "Unsupported solid element type" in g
    assert "SOLIDSCATRA's cell types are HEX8" in g, (
        "the reason, not just the symptom — otherwise the agent keeps editing "
        "section names"
    )


def test_it_offers_a_route_that_was_actually_run():
    g = _grammar()
    assert "slab one element thick" in g
    assert "Structure` with WALL elements" in g


def test_it_reaches_the_agent():
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools

    class _MCP:
        def __init__(self):
            self.tools = {}

        def tool(self, *a, **kw):
            def deco(fn):
                self.tools[fn.__name__] = fn
                return fn
            return deco
    mcp = _MCP()
    os.chdir(ROOT)
    register_consolidated_tools(mcp)
    out = mcp.tools["knowledge"](topic="physics", solver="fourc", physics="heat")
    assert "WALL QUAD4" in out and "Unsupported solid element type" in out


@pytest.mark.skipif(not TSI_SRC.is_file(), reason="4C source not present")
def test_the_clone_strategy_still_accepts_only_solidscatra():
    """If 4C gains 2-D TSI, this test fails and the served text must change."""
    src = TSI_SRC.read_text()
    i = src.index("ThermoStructureCloneStrategy::set_element_data")
    body = src[i:i + 1400]
    assert "SolidScatra" in body
    assert "Unsupported solid element type" in body


@pytest.mark.skipif(not FOURC.exists(), reason="4C binary not installed here")
def test_the_wall_deck_really_runs(tmp_path):
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")
    deck = tmp_path / "wall.4C.yaml"
    deck.write_text((FIX / "wall_quad4_runs.4C.yaml").read_text())
    p = subprocess.run([str(FOURC), str(deck), str(tmp_path / "out")],
                       capture_output=True, text=True, timeout=300,
                       env=env, cwd=tmp_path)
    assert p.returncode == 0, f"the served 2-D deck no longer runs: {p.stdout[-500:]}"
    assert "finished normally" in (p.stdout + p.stderr)


@pytest.mark.skipif(not FOURC.exists(), reason="4C binary not installed here")
def test_the_2d_tsi_deck_still_fails_the_way_the_text_says(tmp_path):
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")
    deck = tmp_path / "tsi.4C.yaml"
    deck.write_text((FIX / "tsi_2d_is_impossible.4C.yaml").read_text())
    # stdbuf IS REQUIRED, and that is itself the point. 4C's stdout is block
    # buffered, and MPI_Abort kills the process with the buffer unflushed, so
    # the diagnostic is LOST. Run without it and this deck looks like a silent
    # crash; run with it and 4C names the cause. Measured here: the same deck,
    # same binary, message present only with stdbuf.
    p = subprocess.run(["stdbuf", "-oL", str(FOURC), str(deck),
                        str(tmp_path / "out")],
                       capture_output=True, text=True, timeout=300,
                       env=env, cwd=tmp_path)
    assert p.returncode != 0
    assert "Unsupported solid element type" in (p.stdout + p.stderr), (
        "2-D TSI fails differently now; re-measure before trusting the served "
        "explanation"
    )
