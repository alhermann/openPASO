"""The grammar door serves a TSI slab deck skeleton (sections and syntax, placeholder numbers) and the
skeleton itself passes the lint against the installed binary's grammar: no unknown section, no unknown
or missing material parameter, no condition on an undefined id, no twisted element. Measured need:
round 40's C1 cells spent their budget on TSI deck grammar (7052: 'time budget exhausted debugging 4C
deck syntax') while the door carried only the Scalar_Transport skeleton."""
from __future__ import annotations
import re
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FOURC = Path("/home/user/4C/build/4C")


def _skeleton():
    from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR as G
    i = G.index("THE TSI SLAB DECK GRAMMAR")
    body = G[G.index("  TITLE:", i):]           # the skeleton starts at its first section, not at the prose
    j = body.index("  Run it as")
    lines = [l[2:] if l.startswith("  ") else l for l in body[:j].splitlines()]
    deck = "\n".join(re.sub(r"\s+#.*$", "", l) for l in lines if l.strip())    # only the trailing comments go
    return deck + "\n"


def test_the_door_serves_the_tsi_skeleton_with_its_traps():
    from backends.fourc.deck_grammar import FOURC_DECK_GRAMMAR as G
    for phrase in ('PROBLEMTYPE: "Thermo_Structure_Interaction"', "CLONING MATERIAL MAP", 'COUPVARIABLE: "Temperature"',
                   "SOLIDSCATRA HEX8", "TAG: monitor_reaction", "DESIGN POINT THERMO DIRICH CONDITIONS",
                   "the section says DSURF, the entries say DSURFACE", "never **", "every NUMBER\nis an ARBITRARY PLACEHOLDER"):
        assert phrase in G, phrase


def test_the_skeleton_passes_the_lint_against_the_installed_grammar():
    if not FOURC.is_file():
        pytest.skip("4C binary not on this host")
    from tools.fourc_deck_lint import deck_judgement
    deck = _skeleton()
    assert 'PROBLEMTYPE: "Thermo_Structure_Interaction"' in deck and "STRUCTURE ELEMENTS:" in deck
    findings = deck_judgement(deck)
    assert findings == [], findings


def test_the_skeleton_reaches_the_agent_through_the_coupling_door():
    import tools.coupling_knowledge as CK
    src = Path(CK.__file__).read_text()
    assert "FOURC_DECK_GRAMMAR" in src       # the coupling reply carries the grammar door's text


def test_the_skeleton_runs_in_4c_as_served(tmp_path):
    """The door says 'a deck with these sections runs': measured here, placeholders and all."""
    if not FOURC.is_file():
        pytest.skip("4C binary not on this host")
    import os, subprocess
    deck = _skeleton()
    assert "constant: [1.0]" in deck                      # lowercase keys survive the extraction
    (tmp_path / "skel.4C.yaml").write_text(deck)
    r = subprocess.run(f"stdbuf -oL -eL {FOURC} skel.4C.yaml out", shell=True, cwd=tmp_path, capture_output=True, text=True,
                       timeout=300, env=dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib"))
    log = r.stdout + r.stderr
    assert "finished normally" in log, log[-1500:]
    assert list((tmp_path / "out-vtk-files").glob("structure-*.vtu")) and list((tmp_path / "out-vtk-files").glob("thermo-*.vtu")), \
        sorted(p.name for p in (tmp_path / "out-vtk-files").glob("*"))
    assert list(tmp_path.glob("out-*monitor_dbc.yaml")), sorted(p.name for p in tmp_path.glob("out*"))
