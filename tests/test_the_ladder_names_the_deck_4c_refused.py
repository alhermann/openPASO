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


def test_unknown_sections_name_the_closest_known_ones():
    from tools.fourc_deck_lint import unknown_sections
    valid = {f"SECTION {i}" for i in range(120)} | {"THERMAL DYNAMIC/RUNTIME VTK OUTPUT", "IO/RUNTIME VTK OUTPUT/STRUCTURE",
                                                     "STRUCTURE ELEMENTS", "TRANSPORT ELEMENTS", "TITLE"}
    deck = 'TITLE:\n  - "x"\nIO/RUNTIME VTK OUTPUT/THERMO:\n  OUTPUT_THERMO: true\nSOLIDSCATRA ELEMENTS:\n  - "1 SOLIDSCATRA HEX8"\nFUNCT1:\n  - x\n'
    out = unknown_sections(deck, valid, elements={"SOLIDSCATRA", "WALL"})
    assert len(out) == 2
    assert "IO/RUNTIME VTK OUTPUT/THERMO" in out[0] and "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" in out[0]
    assert "SOLIDSCATRA ELEMENTS" in out[1] and "ELEMENT TYPE" in out[1] and "STRUCTURE ELEMENTS" in out[1]
    assert unknown_sections(deck, {"a", "b"}) == []           # no binary, no judgement


def test_the_real_binary_grammar_judges_a_worker_section(tmp_path):
    """With the installed binary, the ladder brief names the invented section and the closest real one
    (measured on te4c9 sample 0: 'IO/RUNTIME VTK OUTPUT/THERMO')."""
    import os, pytest
    from tools.fourc_deck_lint import grammar, unknown_sections
    binp = Path("/home/alexander/4C/build/4C")
    if not binp.is_file():
        pytest.skip("4C binary not on this host")
    g = grammar(str(binp), "/opt/4C-dependencies/lib"); valid = g["sections"]
    assert len(valid) > 100 and "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" in valid and "STRUCTURE ELEMENTS" in valid
    assert "SOLIDSCATRA" in g["elements"] and "TRANSP" in g["elements"]
    e = unknown_sections("SOLIDSCATRA ELEMENTS:\n  - x\n", valid, g["elements"])
    assert len(e) == 1 and "ELEMENT TYPE" in e[0] and "STRUCTURE ELEMENTS" in e[0]
    # measured worker names: the invented family word and the spelled-out acronym
    e = unknown_sections("THERMO STRUCTURE INTERACTION DYNAMIC:\n  x: 1\nDESIGN POINT STRUCTURE DIRICH CONDITIONS:\n  - E: 1\n", valid, g["elements"])
    assert "'TSI DYNAMIC'" in e[0], e[0]
    assert e[1].split("closest known: ")[1].startswith("'DESIGN POINT DIRICH CONDITIONS'"), e[1]
    e = unknown_sections("DESIGN LINE STRUCTURE DIRICH CONDITIONS:\n  - E: 1\n", valid, g["elements"])
    assert e[0].split("closest known: ")[1].startswith("'DESIGN LINE DIRICH CONDITIONS'"), e[0]   # no two-letter acronym hit (LS)
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", PART)
    _w(tmp_path / "side_B", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    _w(tmp_path / "side_A", "config.json", json.dumps({"fourc_bin": str(binp), "fourc_ld": "/opt/4C-dependencies/lib"}))
    _w(tmp_path / "side_A", "run_u.4C.yaml", 'PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nIO/RUNTIME VTK OUTPUT/THERMO:\n  OUTPUT_THERMO: true\n'
                                              'CLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\nTSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\n')
    _w(tmp_path / "side_A", "run_u.log", "PROC 0 ERROR in x.cpp, line 546:\nSection 'IO/RUNTIME VTK OUTPUT/THERMO' is not a valid section name.\n")
    r = coupled_ladder(tmp_path)
    assert r["step"] == 2 and "THERMAL DYNAMIC/RUNTIME VTK OUTPUT" in r["brief"] and "is not a valid section name" in r["brief"]


def test_a_signal_crash_and_a_line_without_an_edge_are_named():
    """Measured on a te4c10 worker deck: the interface DLINE's node ids did not match the element
    numbering, 4C's flux table divided by a zero boundary area and died on a floating point exception
    with no error message at all; the lint names both the crash and the cause."""
    from tools.fourc_deck_lint import fourc_error_lines, lint_deck
    log = ("banner\nNormal fluxes at boundary 'ScaTraFluxCalc' on discretization 'scatra':\n+----+\n| ID | DOF |\n"
           "[kevin:1] *** Process received signal ***\n[kevin:1] Signal: Floating point exception (8)\n"
           "[kevin:1] Signal code: Floating point divide-by-zero (3)\n[kevin:1] [ 0] libc\n")
    said = fourc_error_lines(log)
    assert "Floating point exception" in said and "ScaTraFluxCalc" in said
    deck = ('TRANSPORT ELEMENTS:\n  - "1 TRANSP QUAD4 1 2 5 4 MAT 1 TYPE Std"\n  - "2 TRANSP QUAD4 2 3 6 5 MAT 1 TYPE Std"\n'
            'DLINE-NODE TOPOLOGY:\n  - "NODE 1 DLINE 1"\n  - "NODE 4 DLINE 1"\n  - "NODE 7 DLINE 2"\n  - "NODE 8 DLINE 2"\n')
    why = lint_deck(deck)
    assert any("DLINE 2" in w and "shares no edge" in w for w in why), why
    assert not any("DLINE 1" in w for w in why), why       # nodes 1-4 ARE an edge of element 1


def test_a_console_without_verdict_is_quoted_not_called_missing(tmp_path):
    """Measured on a ladder-loop trial: 4C's console ended in a crash the gate did not recognise, and the
    brief said 'no 4C console found' while the log lay next to the deck. Now its last lines are quoted."""
    from tools.result_audit import coupled_ladder
    _w(tmp_path / "side_A", "participant_A.py", PART)
    _w(tmp_path / "side_B", "participant_B.py", PART)
    _w(tmp_path / "side_B", "exports.json", json.dumps({"values": [1.0], "normal_fluxes": [2.0]}))
    _w(tmp_path / "side_A", "deck_T.4C.yaml", 'PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\n')
    _w(tmp_path / "side_A", "deck_T.4C.yaml.log", "4C banner\nProblem type: Scalar_Transport\n| time loop |\nsomething odd happened here\ncore dumped\n")
    r = coupled_ladder(tmp_path)
    assert r["step"] == 2 and "no 4C console found" not in r["brief"]
    assert "ends without a finish" in r["text"] and "core dumped" in r["brief"]


def test_a_temperature_in_the_structural_dirichlet_family_is_named():
    """Measured on a te4c10 worker deck: a point temperature under DESIGN POINT DIRICH CONDITIONS with
    NUMDOF 1 stops a TSI run with '1 DOFs given but 3 expected in Point Dirichlet boundary condition'."""
    from tools.fourc_deck_lint import lint_deck
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nCLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\n'
            'TSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\n'
            'DESIGN POINT DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.5]\n    FUNCT: [0]\n'
            'DESIGN POINT THERMO DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.5]\n    FUNCT: [0]\n'
            'DNODE-NODE TOPOLOGY:\n  - "NODE 1 DNODE 1"\n')
    why = lint_deck(deck)
    assert any("DESIGN POINT DIRICH CONDITIONS has an entry with NUMDOF 1" in w and "THERMO DIRICH" in w for w in why), why
    assert not any("THERMO DIRICH CONDITIONS has an entry" in w for w in why)


def test_a_flux_calc_line_without_topology_is_named():
    """Measured on a te4c10 worker deck: SCATRA FLUX CALC LINE CONDITIONS on E 2 with no DLINE topology at
    all; 4C stopped with 'DLine 1 not in range [0:0['. A DESIGN-only check let it through."""
    from tools.fourc_deck_lint import lint_deck
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  CALCFLUX_BOUNDARY: "diffusive"\n'
            'SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\nDSURF-NODE TOPOLOGY:\n  - "NODE 1 DSURFACE 1"\n')
    why = lint_deck(deck)
    assert any("SCATRA FLUX CALC LINE CONDITIONS names E id(s) 2" in w for w in why), why


def test_twisted_elements_are_named_with_the_counter_clockwise_rule():
    """Measured on a ladder-loop worker deck: quads written (i, i+1, i+NX, i+NX+1) -- zero area -- and 4C
    said only 'The determinant of the matrix is equal zero or negative!'."""
    from tools.fourc_deck_lint import lint_deck
    coords = "".join(f'  - "NODE {1 + i + 3 * j} COORD {i * 0.5:.1f} {j * 0.5:.1f} 0.0"\n' for j in range(2) for i in range(3))
    # the measured pattern (i, i+1, i+NX, i+NX+1) with NX = 2: (1, 2, 4, 5) crosses itself, signed area 0
    twisted = 'NODE COORDS:\n' + coords + 'TRANSPORT ELEMENTS:\n  - "1 TRANSP QUAD4 1 2 4 5 MAT 1 TYPE Std"\n  - "2 TRANSP QUAD4 2 3 6 5 MAT 1 TYPE Std"\n'
    why = lint_deck(twisted)
    assert any("zero or negative area" in w and "element 1" in w and "counter-clockwise" in w for w in why), why
    good = 'NODE COORDS:\n' + coords + 'TRANSPORT ELEMENTS:\n  - "1 TRANSP QUAD4 1 2 5 4 MAT 1 TYPE Std"\n  - "2 TRANSP QUAD4 2 3 6 5 MAT 1 TYPE Std"\n'
    assert not any("zero or negative area" in w for w in lint_deck(good))


def test_material_parameters_are_judged_by_the_grammar():
    """Measured on a ladder-loop worker deck: MAT_Struct_ThermoStVenantK without YOUNGNUM stopped 4C with
    "Parameter 'YOUNGNUM' not found in container"; the gate names the missing parameter and the material's
    parameter list, an unknown material name gets the closest known ones."""
    import pytest
    from tools.fourc_deck_lint import grammar, material_defects
    binp = Path("/home/alexander/4C/build/4C")
    if not binp.is_file():
        pytest.skip("4C binary not on this host")
    g = grammar(str(binp), "/opt/4C-dependencies/lib"); mats = g["materials"]
    assert "MAT_Struct_ThermoStVenantK" in mats and mats["MAT_Struct_ThermoStVenantK"]["YOUNGNUM"]["required"] is True
    assert mats["MAT_Struct_ThermoStVenantK"]["THERMOMAT"]["required"] is False and "MAT_scatra" in mats and "MAT_Fourier" in mats
    assert mats["MAT_Struct_ThermoStVenantK"]["YOUNG"]["type"] == "vector" and mats["MAT_Fourier"]["CAPA"]["type"] == "double"
    assert list(mats["MAT_Fourier"]) == ["CAPA", "CONDUCT"]
    deck = ('MATERIALS:\n  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n      YOUNG: [1.0]\n      NUE: 0.3\n      DENS: 1\n'
            '      THEXPANS: 1e-5\n      INITTEMP: 0\n      THERMOMAT: 2\n  - MAT: 2\n    MAT_Fourier:\n      CAPA: 1\n      CONDUCT:\n        constant: [1.0]\n'
            '  - MAT: 3\n    MAT_ThermoStVenantK:\n      YOUNG: [1.0]\nCLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\n')
    out = material_defects(deck, mats)
    assert any("MAT_Struct_ThermoStVenantK is missing required parameter(s) YOUNGNUM" in w for w in out), out
    assert not any("MAT_Fourier" in w for w in out), out
    assert any("material 'MAT_ThermoStVenantK' is not in the binary's grammar" in w and "MAT_Struct_ThermoStVenantK" in w for w in out), out


def test_a_vector_parameter_written_as_a_number_and_a_deck_without_vtk_output_are_named():
    """Measured on ladder-loop decks: YOUNG: 787.5 where the grammar says `type: vector` ('Could not match
    this input'), and a TSI run that finished normally without the structure/thermo VTU the recovery reads."""
    import pytest
    from tools.fourc_deck_lint import grammar, material_defects, lint_deck
    binp = Path("/home/alexander/4C/build/4C")
    if not binp.is_file():
        pytest.skip("4C binary not on this host")
    mats = grammar(str(binp), "/opt/4C-dependencies/lib")["materials"]
    deck = ('MATERIALS:\n  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n      YOUNGNUM: 1\n      YOUNG: 787.5\n      NUE: 0.3125\n      DENS: 1\n'
            '      THEXPANS: 1e-5\n      INITTEMP: 0\n      THERMOMAT: 2\nCLONING MATERIAL MAP:\n  - SRC_FIELD: "structure"\n')
    out = material_defects(deck, mats)
    assert any("YOUNG is a vector" in w and "[787.5]" in w for w in out), out
    tsi = 'PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nCLONING MATERIAL MAP:\n  - x\nTSI DYNAMIC/PARTITIONED:\n  COUPVARIABLE: "Temperature"\n'
    why = lint_deck(tsi)
    assert any("no structure VTU" in w for w in why) and any("no thermo VTU" in w for w in why), why
    full = tsi + 'IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\nIO/RUNTIME VTK OUTPUT/STRUCTURE:\n  OUTPUT_STRUCTURE: true\n  DISPLACEMENT: true\nTHERMAL DYNAMIC/RUNTIME VTK OUTPUT:\n  OUTPUT_THERMO: true\n  TEMPERATURE: true\n'
    assert not any("VTU" in w for w in lint_deck(full))
