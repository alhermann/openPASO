"""A 4C side that already ran the binary gets, as its next step, the deck 4C refused -- 4C's own
error lines and the defects OASiS names from the deck text -- never the whole participant again.
The lint is a gate: it reads the agent's decks and consoles and writes nothing."""
from __future__ import annotations
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PART = ('import json\nimp = json.load(open("imports.json"))\n# exports.json LAST: the driver takes its '
        'existence as proof of success.\njson.dump({"values": [1.0], "normal_fluxes": [2.0]}, open("exports.json", "w"))\n')
BAD_DECK = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nTSI DYNAMIC:\n  COUPALGO: "tsi_oneway"\n'
            'TSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\nCLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\n'
            'DESIGN VOL THERMO DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0]\n    FUNCT: [0]\n'
            'DVOLUME-NODE TOPOLOGY:\n  - "NODE 1 DVOLUME 1"\n')
LOG = ("4C banner\n--------\nPROC 0 ERROR\n  Section 'DVOLUME-NODE TOPOLOGY' is not a valid section\n  in file deck_U.4C.yaml\n"
       "MPI_ABORT was invoked\n")


def _w(d: Path, name: str, text: str) -> None:
    d.mkdir(parents=True, exist_ok=True); (d / name).write_text(text)


def test_the_lint_names_the_measured_defect_classes():
    from tools.fourc_deck_lint import lint_deck, fourc_error_lines
    why = " | ".join(lint_deck(BAD_DECK))
    assert "DVOLUME" in why and "VOL THERMO DIRICH" in why and "no *-NODE TOPOLOGY section defines" in why
    assert "DVOLUME-NODE TOPOLOGY' is not a valid section" in fourc_error_lines(LOG)
    assert lint_deck('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  CALCFLUX_BOUNDARY: "diffusive"\n'
                     'SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\nDLINE-NODE TOPOLOGY:\n  - "NODE 1 DLINE 2"\n') == []
    # DSURFACE is the entity word for a SURF condition (measured false positive: a worker deck with
    # "NODE 1 DSURFACE 1" was told its SURF condition E 1 had no topology); ids are per kind
    clean = ('DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\nDESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n'
             'DSURF-NODE TOPOLOGY:\n  - "NODE 1 DSURFACE 1"\nDLINE-NODE TOPOLOGY:\n  - "NODE 1 DLINE 1"\n')
    assert lint_deck(clean) == []
    cross = ('DESIGN SURF NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 1\nDLINE-NODE TOPOLOGY:\n  - "NODE 1 DLINE 1"\n')
    assert any("DESIGN SURF NEUMANN CONDITIONS names E id(s) 1" in w for w in lint_deck(cross))


def test_step_2_is_the_refused_deck_when_4c_already_ran(tmp_path):
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", PART)
    _w(tmp_path / "side_B", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    _w(tmp_path / "side_A", "deck_U.4C.yaml", BAD_DECK)
    _w(tmp_path / "side_A", "deck_U.4C.yaml.log", LOG)
    (tmp_path / "side_A" / "out_T-vtk-files").mkdir()
    (tmp_path / "side_A" / "out_T-vtk-files" / "scatra-00001-0.vtu").write_text("<VTKFile/>")
    r = coupled_ladder(tmp_path)
    assert r["step"] == 2 and "MAKE THE 4C DECK RUN" in r["text"]
    b = r["brief"]
    assert "out_T" in b and "finished" in b and "leave them" not in b   # a finished run on a defective deck is not left alone
    assert "not a valid section" in b                          # 4C's own words
    assert "deck_U.4C.yaml" in b and "DVOLUME" in b and "VOL THERMO DIRICH" in b
    assert "spawn_subagent(role='worker'" in r["text"]


def test_the_participants_own_python_stop_is_in_the_brief(tmp_path):
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", PART)
    _w(tmp_path / "side_B", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    _w(tmp_path / "side_A", "participant_output.log", "banner\nTraceback (most recent call last):\n  File \"participant_A.py\", line 513\n"
                                                      "    deck_u += f'{node_id + {n_nodes}}'\nTypeError: unsupported operand type(s) for +: 'int' and 'set'\n")
    r = coupled_ladder(tmp_path)
    assert r["step"] == 2 and "stopped in Python" in r["text"] and "TypeError" in r["brief"] and "fix that line first" in r["brief"]


def test_no_deck_and_no_console_keeps_the_plain_step_2(tmp_path):
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", PART)
    _w(tmp_path / "side_B", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    r = coupled_ladder(tmp_path)
    assert r["step"] == 2 and "RUN EACH PARTICIPANT STANDALONE" in r["text"]
