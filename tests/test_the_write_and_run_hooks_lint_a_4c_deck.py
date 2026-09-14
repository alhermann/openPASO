"""The 4C deck lint fires where the parent already works: on the write of a deck FILE and on the shell
command that runs the binary on one. Every defect is named at once (the binary stops at the first);
nothing is written or completed; a non-deck file and a non-4C command get nothing at all."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tools.workspace_advisor import _fourc_deck_write_check, _fourc_run_check, _fourc_after_shell_check   # noqa: E402
from tools.fourc_deck_lint import run_command_deck, looks_like_deck             # noqa: E402

FOURC = Path("/home/user/4C/build/4C")
BAD = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nIO/RUNTIME VTK OUTPUT/THERMO:\n  OUTPUT_THERMO: true\n'
       'SOLIDSCATRA ELEMENTS:\n  - "1 SOLIDSCATRA HEX8 1 2 3 4 5 6 7 8 MAT 1 KINEM linear TYPE Undefined"\n'
       'MATERIALS:\n  - MAT: 1\n    MAT_Struct_ThermoStVenantK:\n      YOUNGNUM: 1\n'
       'DESIGN VOL NEUMANN CONDITIONS:\n  - E: 1\n    NUMDOF: 3\nDVOLUME-NODE TOPOLOGY:\n  - "NODE 1 DVOLUME 1"\n')
CLEAN = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  CALCFLUX_BOUNDARY: "diffusive"\n'
         'IO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n'
         'MATERIALS:\n  - MAT: 1\n    MAT_scatra:\n      DIFFUSIVITY: 1.0\n'
         'SCATRA FLUX CALC LINE CONDITIONS:\n  - E: 2\nDESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n'
         'DLINE-NODE TOPOLOGY:\n  - "NODE 1 DLINE 1"\n  - "NODE 2 DLINE 2"\n')
STOP = ("Trilinos Version: 0123456789ab (git SHA1)\n----------------\nPROC 0 ERROR in /x/4C_io_input_file.cpp, line 12:\n"
        "Section DVOLUME-NODE TOPOLOGY is unknown\n--------------------------------------------------------------------------\n"
        "MPI_ABORT was invoked on rank 0\n")


def test_a_written_deck_file_gets_every_defect_named_before_any_run(tmp_path):
    out = _fourc_deck_write_check(tmp_path / "side_A" / "slab.4C.yaml", BAD)
    assert out.startswith("\n[write check] 4C DECK slab.4C.yaml: "), out
    assert "defect(s) named by the deck lint before any run" in out
    assert "topology entries use DVOLUME" in out                      # the measured topology-word trap
    assert "names E id(s) 1" in out                                    # the silently dropped condition
    if FOURC.is_file():
        assert "DVOLUME-NODE TOPOLOGY" in out and "DVOL-NODE TOPOLOGY" in out, out   # grammar + closest name
    assert "fix every one, then run" in out


def test_a_clean_deck_gets_the_one_line_that_says_so_and_no_fix(tmp_path):
    if not FOURC.is_file():
        pytest.skip("4C binary not on this host")
    out = _fourc_deck_write_check(tmp_path / "deck.yaml", CLEAN)
    assert "no defect named by the deck lint before any run" in out, out
    assert "not proof it runs" in out


def test_non_decks_and_non_yaml_files_get_nothing(tmp_path):
    assert _fourc_deck_write_check(tmp_path / "participant_A.py", BAD) == ""
    assert _fourc_deck_write_check(tmp_path / "config.yaml", "level: 1\nnx: 8\n") == ""
    assert not looks_like_deck("a: 1\nb: 2\n")


def test_the_shell_command_that_runs_4c_gets_the_decks_defects_and_4cs_own_stop(tmp_path):
    (tmp_path / "side_A").mkdir()
    (tmp_path / "side_A" / "slab.4C.yaml").write_text(BAD)
    cmd = "cd side_A && stdbuf -oL -eL /home/user/4C/build/4C slab.4C.yaml out > run.log 2>&1; tail -20 run.log"
    assert run_command_deck(cmd, tmp_path) == tmp_path / "side_A" / "slab.4C.yaml"
    out = _fourc_run_check(cmd, STOP, tmp_path)
    assert "[run check] 4C DECK slab.4C.yaml:" in out and "topology entries use DVOLUME" in out, out
    assert "[run check] 4C's own stop: Section DVOLUME-NODE TOPOLOGY is unknown" in out, out
    # mpirun in front, absolute deck path, finished run: the console said it all, the lint adds nothing
    (tmp_path / "ok.yaml").write_text(CLEAN)
    quiet = _fourc_run_check(f"mpirun -np 2 /home/user/4C/build/4C {tmp_path}/ok.yaml out",
                             "... 4C finished normally\n", tmp_path)
    if FOURC.is_file():
        assert quiet == "", quiet


def test_commands_that_do_not_run_4c_get_nothing(tmp_path):
    assert _fourc_run_check("python3 side_A/participant_A.py", STOP, tmp_path) == ""
    assert _fourc_run_check("ls -la 4C_output; cat deck.yaml", "x", tmp_path) == ""
    assert run_command_deck("echo 4C deck.yaml", tmp_path) is None      # the deck is not on disk


def test_the_harness_calls_both_hooks_and_carries_no_check_body():
    src = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert "_fourc_deck_write_check(p, content)" in src
    assert "_fourc_run_check(command, out, workdir)" in src
    assert "_fourc_after_shell_check(workdir, _started_at, command)" in src and "def _fourc_after_shell_check" not in src
    assert "def _fourc_deck_write_check" not in src and "def _fourc_run_check" not in src


def test_a_console_redirected_to_a_file_is_read_from_that_file(tmp_path):
    (tmp_path / "side_A").mkdir()
    (tmp_path / "side_A" / "slab.4C.yaml").write_text(BAD)
    (tmp_path / "side_A" / "run_4C.log").write_text(STOP)
    cmd = "cd side_A && stdbuf -oL -eL /home/user/4C/build/4C slab.4C.yaml out > run_4C.log 2>&1"
    out = _fourc_run_check(cmd, "", tmp_path)                       # the shell reply carried nothing
    assert "4C's own stop: Section DVOLUME-NODE TOPOLOGY is unknown" in out, out
    (tmp_path / "side_A" / "run_4C.log").write_text("... processor 0 finished normally\n")
    (tmp_path / "side_A" / "slab.4C.yaml").write_text(CLEAN)
    if FOURC.is_file():
        assert _fourc_run_check(cmd, "", tmp_path) == ""               # finished, clean: nothing to add


GRID = "NODE COORDS:\n" + "".join(f'  - "NODE {1 + 3 * j + i} COORD {0.5 * i:.1f} {0.5 * j:.1f} 0.0"\n' for j in range(3) for i in range(3))


def test_a_condition_on_a_line_inside_the_mesh_is_named_and_a_boundary_line_is_not():
    from tools.fourc_deck_lint import lint_deck
    cond = 'DESIGN LINE DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n'
    inside = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\n' + GRID + cond
              + 'DLINE-NODE TOPOLOGY:\n  - "NODE 2 DLINE 1"\n  - "NODE 5 DLINE 1"\n  - "NODE 8 DLINE 1"\n')
    hits = [f for f in lint_deck(inside) if "INSIDE the mesh" in f]
    assert hits and hits[0].startswith("DLINE 1 (3 nodes) lies on x = 0.5, INSIDE the mesh (x spans 0..1)"), lint_deck(inside)
    on_boundary = inside.replace('"NODE 2 DLINE 1"', '"NODE 1 DLINE 1"').replace('"NODE 5 DLINE 1"', '"NODE 4 DLINE 1"') \
                        .replace('"NODE 8 DLINE 1"', '"NODE 7 DLINE 1"')
    assert not [f for f in lint_deck(on_boundary) if "INSIDE the mesh" in f]
    unused = inside.replace(cond, "")                        # an interior line nothing is filed on is not a defect
    assert not [f for f in lint_deck(unused) if "INSIDE the mesh" in f]


def test_a_console_a_participant_script_wrote_during_the_command_is_named_after_the_shell(tmp_path):
    import os, time
    (tmp_path / "side_A").mkdir()
    (tmp_path / "side_A" / "slab.4C.yaml").write_text(BAD)
    (tmp_path / "side_A" / "slab.4C.yaml.log").write_text(STOP)
    t0 = time.time() - 5                                          # the command started five seconds ago
    out = _fourc_after_shell_check(tmp_path, t0, "python side_A/participant_A.py")
    assert out.startswith("\n[run check] 4C stopped in side_A/slab.4C.yaml.log: Section DVOLUME-NODE TOPOLOGY is unknown"), out
    assert "4C DECK slab.4C.yaml:" in out and "topology entries use DVOLUME" in out, out
    # a console older than the command is not this command's doing
    old = time.time() - 600
    os.utime(tmp_path / "side_A" / "slab.4C.yaml.log", (old, old))
    assert _fourc_after_shell_check(tmp_path, time.time() - 5, "python side_A/participant_A.py") == ""
    # the console of a deck the command ran directly is the run check's, not this one's
    os.utime(tmp_path / "side_A" / "slab.4C.yaml.log", None)
    assert _fourc_after_shell_check(tmp_path, time.time() - 5,
                                    "cd side_A && /home/user/4C/build/4C slab.4C.yaml out > slab.4C.yaml.log 2>&1") == ""


def test_4cs_stop_keeps_the_offending_snippet_after_the_blank_line():
    from tools.fourc_deck_lint import fourc_error_lines
    log = ("PROC 0 ERROR in /x/4C_io_input_spec_builders.cpp, line 633:\nCould not match this input\n\nIO:\n"
           "  VERBOSITY: \"Standard\"\n  RUNTIME VTK OUTPUT:\n    THERMO:\n      OUTPUT_SCALAR: phi_1\n\n\n"
           "against the given input specification. This was the best attempt to match the input:\n\n[!] Candidate group 'IO'\n"
           "--------------------------------------------------------------------------------\n 0# void FourC::Core::x() in lib4C.so\n")
    said = fourc_error_lines(log)
    assert said.startswith("Could not match this input | IO: | VERBOSITY: \"Standard\" | RUNTIME VTK OUTPUT:"), said
    assert "0#" not in said


def test_a_python_power_inside_a_4c_function_is_named_for_every_occurrence():
    from tools.fourc_deck_lint import lint_deck
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi**2*x*sin(pi*y)"\n'
            'FUNCT2:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.01*pi**2*x**2*sin(pi*y)"\nFUNCT3:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi^2*x*sin(pi*y)"\n')
    hits = [f for f in lint_deck(deck) if "uses `**`" in f]
    assert len(hits) == 2, lint_deck(deck)
    assert "2*pi**2*x*sin(pi*y)" in hits[0] and "write `^`" in hits[0]


def test_a_finished_run_whose_field_dwarfs_its_data_is_named(tmp_path):
    import numpy as np
    meshio = pytest.importorskip("meshio")
    from tools.fourc_deck_lint import field_scale_findings
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nFUNCT1:\n  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "2*pi^2*x*sin(pi*y)"\n'
            'DESIGN POINT DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 1\n    ONOFF: [1]\n    VAL: [0.8]\n    FUNCT: [0]\n')
    (tmp_path / "out-vtk-files").mkdir()
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    cells = [("quad", np.array([[0, 1, 3, 2]]))]
    meshio.write(tmp_path / "out-vtk-files" / "scatra-00001-0.vtu", meshio.Mesh(pts, cells, point_data={"phi_1": np.full(4, 8.5e13)}))
    hits = field_scale_findings(deck, tmp_path)
    assert len(hits) == 1 and hits[0].startswith("4C FIELD SCALE: phi_1 peaks at 8.5e+13") and "is 0.8" not in hits[0], hits
    assert "largest number the deck prescribes (VAL entries, FUNCT constants) is 2" in hits[0]
    meshio.write(tmp_path / "out-vtk-files" / "scatra-00001-0.vtu", meshio.Mesh(pts, cells, point_data={"phi_1": np.array([0.0, 0.8, 0.0, 0.79])}))
    assert field_scale_findings(deck, tmp_path) == []
    # the run hook carries it for a finished run
    (tmp_path / "slab.4C.yaml").write_text(deck)
    meshio.write(tmp_path / "out-vtk-files" / "scatra-00001-0.vtu", meshio.Mesh(pts, cells, point_data={"phi_1": np.full(4, 8.5e13)}))
    out = _fourc_run_check("stdbuf -oL /home/user/4C/build/4C slab.4C.yaml out", "... processor 0 finished normally\n", tmp_path)
    assert "[run check] 4C FIELD SCALE: phi_1 peaks at 8.5e+13" in out, out


def _slab(order):
    nodes = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, .1), (1, 0, .1), (1, 1, .1), (0, 1, .1)]
    coords = "NODE COORDS:\n" + "".join(f'  - "NODE {i + 1} COORD {x} {y} {z}"\n' for i, (x, y, z) in enumerate(nodes))
    return ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\n' + coords
            + 'STRUCTURE ELEMENTS:\n  - "1 SOLIDSCATRA HEX8 ' + " ".join(map(str, order)) + ' MAT 1 KINEM linear TYPE Undefined"\n')


def test_a_scrambled_slab_hex_is_named_and_a_proper_one_is_not():
    from tools.fourc_deck_lint import _bad_hex8_slabs
    assert _bad_hex8_slabs(_slab([1, 2, 3, 4, 5, 6, 7, 8])) == []                      # bottom CCW, then the top layer
    scrambled = _bad_hex8_slabs(_slab([1, 5, 2, 6, 3, 7, 4, 8]))                        # layer-interleaved, the te4c13 shape
    assert len(scrambled) == 1 and "first four nodes do not lie on one layer" in scrambled[0], scrambled
    clockwise = _bad_hex8_slabs(_slab([1, 4, 3, 2, 5, 8, 7, 6]))
    assert clockwise and "runs clockwise" in clockwise[0], clockwise
    assert "ZERO OR NEGATIVE JACOBIAN" in clockwise[0]


def test_a_parameter_written_at_the_top_level_is_named_as_not_a_section():
    if not FOURC.is_file():
        pytest.skip("4C binary not on this host")
    from tools.fourc_deck_lint import deck_judgement
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nSCALAR TRANSPORT DYNAMIC:\n  TIMEINTEGR: "Stationary"\n'
            'CALCFLUX_BOUNDARY: "diffusive"\nIO/RUNTIME VTK OUTPUT:\n  INTERVAL_STEPS: 1\n')
    hits = [f for f in deck_judgement(deck) if "top-level key `CALCFLUX_BOUNDARY" in f]
    assert hits and "belongs INSIDE its section" in hits[0], deck_judgement(deck)
    assert not [f for f in deck_judgement(CLEAN) if "top-level key" in f]


YAML_STOP = ("Trilinos Version: f4d64271518 (git SHA1)\nTotal number of MPI ranks: 1\n\n\n=================\nERROR: could not find ':' colon after key\n"
             "204:40: 2 TRANSP QUAD4 2 3 12 11 MAT 1 TYPE Std  (size=39)\n                                               ^  (cols 40-40)\n"
             " 0# FourC::(anonymous namespace)::throw_on_yaml_parse_error(char const*) in lib4C.so\n\n=================\n\n"
             "--------------------------------------------------------------------------\nMPI_ABORT was invoked on rank 0\n")


def test_the_yaml_readers_stop_is_extracted_and_the_unquoted_row_named():
    from tools.fourc_deck_lint import fourc_error_lines, lint_deck
    said = fourc_error_lines(YAML_STOP)
    assert said.startswith("ERROR: could not find ':' colon after key | 204:40: 2 TRANSP QUAD4 2 3 12 11 MAT 1 TYPE Std"), said
    assert "0#" not in said
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nNODE COORDS:\n  - "NODE 1 COORD 0 0 0"\n  - NODE 2 COORD 1 0 0\n'
            'TRANSPORT ELEMENTS:\n  - 1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std\n  - "2 TRANSP QUAD4 2 3 12 11 MAT 1 TYPE Std"\n')
    hits = [f for f in lint_deck(deck) if "not quoted YAML strings" in f]
    assert len(hits) == 1 and "2 table row(s)" in hits[0] and "first, in NODE COORDS: `- NODE 2 COORD 1 0 0`" in hits[0], lint_deck(deck)
    assert not [f for f in lint_deck(CLEAN) if "not quoted" in f]


def test_an_element_naming_an_undefined_node_is_named():
    from tools.fourc_deck_lint import lint_deck
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Scalar_Transport"\nNODE COORDS:\n  - "NODE 1 COORD 0 0 0"\n  - "NODE 2 COORD 1 0 0"\n'
            '  - "NODE 3 COORD 1 1 0"\n  - "NODE 4 COORD 0 1 0"\nTRANSPORT ELEMENTS:\n  - "1 TRANSP QUAD4 1 2 3 4 MAT 1 TYPE Std"\n'
            '  - "17 TRANSP QUAD4 2 27 28 3 MAT 1 TYPE Std"\n')
    hits = [f for f in lint_deck(deck) if "NODE COORDS does not define" in f]
    assert len(hits) == 1 and "element 17 -> node(s) 27 28" in hits[0] and "cannot find node 27" in hits[0], lint_deck(deck)
    assert not [f for f in lint_deck(CLEAN) if "does not define" in f and "element" in f]


def test_a_yaml_structure_slip_is_named_with_its_line_before_any_run():
    from tools.fourc_deck_lint import lint_deck
    deck = ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\nDESIGN POINT DIRICH CONDITIONS:\n  - E: 1\n   NUMDOF: 3\n'
            '    ONOFF: [1, 1, 1]\n')
    hits = [f for f in lint_deck(deck) if "not valid YAML" in f]
    assert len(hits) == 1 and "line 5" in hits[0] and "NUMDOF: 3" in hits[0] and "ERROR: parse error 5:" in hits[0], lint_deck(deck)
    assert not [f for f in lint_deck(CLEAN) if "not valid YAML" in f]
    assert not [f for f in lint_deck(BAD) if "not valid YAML" in f]


def test_the_after_shell_check_pairs_a_console_with_its_own_deck(tmp_path):
    import os, time
    side = tmp_path / "side_A"; side.mkdir()
    (side / "test_deck.yaml").write_text(CLEAN)                       # an older probe deck
    old = time.time() - 600; os.utime(side / "test_deck.yaml", (old, old))
    (side / "deck_U.4C.yaml").write_text(BAD)                          # the deck run_U.log belongs to
    (side / "run_U.log").write_text(STOP)
    out = _fourc_after_shell_check(tmp_path, time.time() - 5, "cd side_A && python3 participant_A.py")
    assert "4C DECK deck_U.4C.yaml:" in out and "test_deck.yaml" not in out, out


def _pinned_slab(all_pinned: bool):
    nodes = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, .1), (1, 0, .1), (1, 1, .1), (0, 1, .1)]
    coords = "NODE COORDS:\n" + "".join(f'  - "NODE {i + 1} COORD {x} {y} {z}"\n' for i, (x, y, z) in enumerate(nodes))
    surf = [1, 4, 5, 8] + ([2, 3, 6, 7] if all_pinned else [])
    return ('PROBLEM TYPE:\n  PROBLEMTYPE: "Thermo_Structure_Interaction"\n' + coords
            + 'DESIGN SURF DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [1, 1, 1]\n    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]\n'
            + 'DESIGN VOL DIRICH CONDITIONS:\n  - E: 1\n    NUMDOF: 3\n    ONOFF: [0, 0, 1]\n    VAL: [0, 0, 0]\n    FUNCT: [0, 0, 0]\n'
            + 'DSURF-NODE TOPOLOGY:\n' + "".join(f'  - "NODE {n} DSURFACE 1"\n' for n in surf)
            + 'DVOL-NODE TOPOLOGY:\n' + "".join(f'  - "NODE {n} DVOL 1"\n' for n in range(1, 9)))


def test_dirichlet_on_every_node_is_named_and_a_plane_strain_pin_is_not():
    from tools.fourc_deck_lint import lint_deck
    hits = [f for f in lint_deck(_pinned_slab(True)) if "DIRICHLET PINS EVERY NODE" in f]
    assert len(hits) == 1 and "displacement field (8 of 8 nodes" in hits[0] and "res-norm 0" in hits[0], lint_deck(_pinned_slab(True))
    assert not [f for f in lint_deck(_pinned_slab(False)) if "PINS EVERY NODE" in f]       # u_z = 0 everywhere is plane strain


def test_the_harness_surfaces_a_workers_submission_audit_to_the_parent():
    src = (ROOT / "langgraph_eval" / "agent.py").read_text()
    assert "the worker wrote RESULT.txt; the write check on it reported" in src
    assert src.index("_rt_before = ") < src.index('config={"recursion_limit": 40}')      # measured before the worker runs
