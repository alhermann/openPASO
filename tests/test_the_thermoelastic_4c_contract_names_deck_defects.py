"""The served thermo-elastic 4C contract's finish check names the deck defects
measured on worker trials: a DVOLUME topology keyword (4C accepts it and drops
every load on that id), a volume-wide thermal Dirichlet (the heat equation is
then not solved), a TSI deck without CLONING MATERIAL MAP or COUPVARIABLE
Temperature, a 2-D structural element, and condition ids without topology."""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "data" / "coupling_participants" / "participant_fourc_thermoelastic.py").read_text()


def _check(tmp_path, deck_text, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "deck_U.4C.yaml").write_text(deck_text)
    a = SRC.index("def why_4c_did_not_finish(")
    b = SRC.index("def _diagnose_at_exit(")
    ns = {"CFG": {}, "glob": __import__("glob"), "re": re, "sys": __import__("sys"), "Path": Path}
    exec("import glob, re, sys\nfrom pathlib import Path\n" + SRC[a:b], ns)
    return ns["why_4c_did_not_finish"]("run U")


BAD = '''PROBLEM TYPE:
  PROBLEMTYPE: "Thermo_Structure_Interaction"
STRUCTURAL DYNAMIC:
  DYNAMICTYPE: "Statics"
DESIGN VOL NEUMANN CONDITIONS:
  - E: 1
    NUMDOF: 3
DESIGN VOL THERMO NEUMANN CONDITIONS:
  - E: 2
    NUMDOF: 1
DESIGN VOL THERMO DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 3
    TAG: monitor_reaction
STRUCTURE ELEMENTS:
  - "1 WALL QUAD4 1 2 3 4 MAT 1"
DVOL-NODE TOPOLOGY:
  - "NODE 1 DVOLUME 1"
DNODE-NODE TOPOLOGY:
  - "NODE 5 DNODE 1"
'''


def test_the_measured_defects_are_named(tmp_path, monkeypatch):
    why = _check(tmp_path, BAD, monkeypatch)
    assert "4C DID NOT FINISH (run U)" in why
    assert "DVOLUME" in why and "DNODE, DLINE, DSURFACE, DVOL" in why
    assert "CLONING MATERIAL MAP" in why
    assert 'COUPVARIABLE "Temperature"' in why
    assert "no 2-D thermo-elastic element" in why
    assert "IO/MONITOR STRUCTURE DBC" in why
    assert "DESIGN VOL THERMO DIRICH imposes the temperature" in why
    assert "DESIGN VOL THERMO NEUMANN CONDITIONS names E id(s) 2" in why


def test_a_clean_deck_is_not_accused(tmp_path, monkeypatch):
    good = BAD.replace("DVOLUME", "DVOL").replace("  - E: 2\n    NUMDOF: 1\n", "  - E: 1\n    NUMDOF: 1\n") \
              .replace("DESIGN VOL THERMO DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n", "") \
              .replace('  - "1 WALL QUAD4 1 2 3 4 MAT 1"', '  - "1 SOLIDSCATRA HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear TYPE Undefined"')
    good += 'CLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\nTSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\nIO/MONITOR STRUCTURE DBC:\n  INTERVAL_STEPS: 1\n'
    why = _check(tmp_path, good, monkeypatch)
    for word in ("DVOLUME", "CLONING MATERIAL MAP pairing", "needs COUPVARIABLE", "no 2-D thermo-elastic",
                 "writes nothing without", "imposes the temperature", "names E id(s)"):
        assert word not in why, why
