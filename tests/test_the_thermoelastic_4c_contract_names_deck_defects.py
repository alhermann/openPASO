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


def test_the_served_deck_check_refuses_a_condition_on_an_undefined_e_id(tmp_path):
    """4C runs a deck whose condition E ids have no topology to 'finished normally' with those
    conditions dropped (measured on a worker deck). The served recovery refuses before reading
    any output, naming the ids -- a gate on the agent's own deck, not a deck writer."""
    import json, os, re, subprocess, sys
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "data" / "coupling_participants" / "participant_fourc_thermoelastic.py").read_text()
    BEGIN = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin"; END = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end"
    a = src.index(BEGIN); b = src.index(END, a) + len(END)
    fill = ('hx, hy = (X1 - X0) / NX, (Y1 - Y0) / NY\n'
            'nodes = [(X0 + i * hx, Y0 + j * hy) for j in range(NY + 1) for i in range(NX + 1)]\n'
            'interior = [j * (NX + 1) + NX + 1 for j in range(1, NY)]\nTZ = hx\n'
            'Path("deck_T.4C.yaml").write_text("PROBLEM TYPE:\\n  PROBLEMTYPE: \\"Scalar_Transport\\"\\n'
            'DESIGN SURF NEUMANN CONDITIONS:\\n  - E: 1\\n    NUMDOF: 1\\nDLINE-NODE TOPOLOGY:\\n  - \\"NODE 1 DLINE 2\\"\\n")\n'
            'OUT_T, OUT_U, DECK_U = "out_T", "out_U", "deck_U.4C.yaml"\n')
    (tmp_path / "participant_A.py").write_text(src[:a] + fill + src[b:])
    (tmp_path / "config.json").write_text(json.dumps({"level": 1, "nx": 4, "ny": 4, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,
                                                       "k": 1.0, "lam": 1.0, "mu": 1.0, "beta": 1.0, "iface": "right"}))
    (tmp_path / "imports.json").write_text("{}")
    r = subprocess.run([sys.executable, "participant_A.py"], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode != 0
    assert "DECK CHECK" in r.stderr and "DESIGN SURF NEUMANN CONDITIONS E 1" in r.stderr, r.stderr[-800:]
