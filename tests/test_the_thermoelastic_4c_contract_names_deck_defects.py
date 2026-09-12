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


def test_the_served_lint_names_the_closest_real_section(tmp_path):
    """The served finish lint judges section names by the installed binary's own grammar and names the
    closest known ones (measured worker failure: 'IO/RUNTIME VTK OUTPUT/THERMO')."""
    import json, os, subprocess, sys
    from pathlib import Path
    import pytest
    binp = Path("/home/alexander/4C/build/4C")
    if not binp.is_file():
        pytest.skip("4C binary not on this host")
    src = (Path(__file__).resolve().parents[1] / "data" / "coupling_participants" / "participant_fourc_thermoelastic.py").read_text()
    BEGIN = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin"; END = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end"
    a = src.index(BEGIN); b = src.index(END, a) + len(END)
    fill = ('nodes = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]\ninterior = []\nTZ = 1.0\n'
            'Path("run_u.4C.yaml").write_text("PROBLEM TYPE:\\n  PROBLEMTYPE: \\"Thermo_Structure_Interaction\\"\\n'
            'IO/RUNTIME VTK OUTPUT/THERMO:\\n  OUTPUT_THERMO: true\\nCLONING MATERIAL MAP:\\n  - SRC_FIELD: \\"structure\\"\\n'
            'TSI DYNAMIC/PARTITIONED:\\n  COUPVARIABLE: \\"Temperature\\"\\n")\n'
            'Path("run_u.log").write_text("PROC 0 ERROR in x.cpp, line 546:\\nSection \'IO/RUNTIME VTK OUTPUT/THERMO\' is not a valid section name.\\n")\n'
            'OUT_T, OUT_U, DECK_U = "out_T", "out_U", "run_u.4C.yaml"\n')
    (tmp_path / "participant_A.py").write_text(src[:a] + fill + src[b:])
    (tmp_path / "config.json").write_text(json.dumps({"level": 1, "nx": 1, "ny": 1, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,
                                                       "k": 1.0, "lam": 1.0, "mu": 1.0, "beta": 1.0, "iface": "right",
                                                       "fourc_bin": str(binp), "fourc_ld": "/opt/4C-dependencies/lib"}))
    (tmp_path / "imports.json").write_text("{}")
    r = subprocess.run([sys.executable, "participant_A.py"], cwd=tmp_path, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode != 0
    assert "is not a valid section name" in r.stderr and "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" in r.stderr, r.stderr[-1200:]


def test_the_served_deck_check_refuses_twisted_elements(tmp_path):
    import json, os, subprocess, sys
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "data" / "coupling_participants" / "participant_fourc_thermoelastic.py").read_text()
    BEGIN = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin"; END = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end"
    a = src.index(BEGIN); b = src.index(END, a) + len(END)
    fill = ('nodes = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.0, 0.5), (0.5, 0.5), (1.0, 0.5)]\ninterior = []\nTZ = 0.5\n'
            'Path("deck_T.4C.yaml").write_text("NODE COORDS:\\n" + "".join(f\'  - "NODE {i + 1} COORD {x} {y} 0.0"\\n\' for i, (x, y) in enumerate(nodes))\n'
            '  + "TRANSPORT ELEMENTS:\\n  - \\"1 TRANSP QUAD4 1 2 4 5 MAT 1 TYPE Std\\"\\n  - \\"2 TRANSP QUAD4 2 3 6 5 MAT 1 TYPE Std\\"\\n")\n'
            'OUT_T, OUT_U, DECK_U = "out_T", "out_U", "deck_U.4C.yaml"\n')
    (tmp_path / "participant_A.py").write_text(src[:a] + fill + src[b:])
    (tmp_path / "config.json").write_text(json.dumps({"level": 1, "nx": 2, "ny": 1, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 0.5,
                                                       "k": 1.0, "lam": 1.0, "mu": 1.0, "beta": 1.0, "iface": "right"}))
    (tmp_path / "imports.json").write_text("{}")
    r = subprocess.run([sys.executable, "participant_A.py"], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                       env=dict(os.environ, MPLBACKEND="Agg"))
    assert r.returncode != 0 and "DECK CHECK" in r.stderr and "zero or negative area" in r.stderr and "element 1" in r.stderr, r.stderr[-800:]


def _run_contract_with_deck(tmp_path, deck_lines: str):
    import json, os, subprocess, sys
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "data" / "coupling_participants" / "participant_fourc_thermoelastic.py").read_text()
    BEGIN = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ begin"; END = "# ── SOLVE ─ OASiS DOES NOT SERVE THIS ─ end"
    a = src.index(BEGIN); b = src.index(END, a) + len(END)
    fill = ('hx, hy = (X1 - X0) / NX, (Y1 - Y0) / NY\n'
            'nodes = [(X0 + i * hx, Y0 + j * hy) for j in range(NY + 1) for i in range(NX + 1)]\n'
            'interior = [j * (NX + 1) + NX + 1 for j in range(1, NY)]\nTZ = hx\n'
            'Path("deck_U.4C.yaml").write_text(' + repr(deck_lines) + ')\n'
            'OUT_T, OUT_U, DECK_U = "out_T", "out_U", "deck_U.4C.yaml"\n')
    (tmp_path / "participant_A.py").write_text(src[:a] + fill + src[b:])
    (tmp_path / "config.json").write_text(json.dumps({"level": 1, "nx": 4, "ny": 4, "x0": 0.0, "x1": 1.0, "y0": 0.0, "y1": 1.0,
                                                       "k": 1.0, "lam": 1.0, "mu": 1.0, "beta": 1.0, "iface": "right"}))
    (tmp_path / "imports.json").write_text("{}")
    return subprocess.run([sys.executable, "participant_A.py"], cwd=tmp_path, capture_output=True, text=True, timeout=120,
                          env=dict(os.environ, MPLBACKEND="Agg"))


def test_the_served_deck_check_refuses_a_section_written_twice(tmp_path):
    deck = 'PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nDSURF-NODE TOPOLOGY:\n  - "NODE 1 DSURFACE 1"\nDSURF-NODE TOPOLOGY:\n  - "NODE 2 DSURFACE 1"\n'
    r = _run_contract_with_deck(tmp_path, deck)
    assert r.returncode != 0 and "DECK CHECK" in r.stderr and "DSURF-NODE TOPOLOGY more than once" in r.stderr, r.stderr[-600:]


def test_the_served_deck_check_refuses_a_layer_interleaved_slab_hex(tmp_path):
    nodes = [(0, 0, 0), (0, 0, .1), (1, 0, 0), (1, 0, .1), (1, 1, 0), (1, 1, .1), (0, 1, 0), (0, 1, .1)]   # te4c13 numbering
    coords = "NODE COORDS:\n" + "".join(f'  - "NODE {i + 1} COORD {x} {y} {z}"\n' for i, (x, y, z) in enumerate(nodes))
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\n' + coords
            + 'STRUCTURE ELEMENTS:\n  - "1 SOLIDSCATRA HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear TYPE Undefined"\n')
    r = _run_contract_with_deck(tmp_path, deck)
    assert r.returncode != 0 and "DECK CHECK" in r.stderr and "not a well-formed one-layer HEX8" in r.stderr, r.stderr[-600:]
    assert "ZERO OR NEGATIVE JACOBIAN" in r.stderr
