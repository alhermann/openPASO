"""The 4C deck lint fires where the parent already works: on the write of a deck FILE and on the shell
command that runs the binary on one. Every defect is named at once (the binary stops at the first);
nothing is written or completed; a non-deck file and a non-4C command get nothing at all."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tools.workspace_advisor import _fourc_deck_write_check, _fourc_run_check   # noqa: E402
from tools.fourc_deck_lint import run_command_deck, looks_like_deck             # noqa: E402

FOURC = Path("/home/alexander/4C/build/4C")
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
    cmd = "cd side_A && stdbuf -oL -eL /home/alexander/4C/build/4C slab.4C.yaml out > run.log 2>&1; tail -20 run.log"
    assert run_command_deck(cmd, tmp_path) == tmp_path / "side_A" / "slab.4C.yaml"
    out = _fourc_run_check(cmd, STOP, tmp_path)
    assert "[run check] 4C DECK slab.4C.yaml:" in out and "topology entries use DVOLUME" in out, out
    assert "[run check] 4C's own stop: Section DVOLUME-NODE TOPOLOGY is unknown" in out, out
    # mpirun in front, absolute deck path, finished run: the console said it all, the lint adds nothing
    (tmp_path / "ok.yaml").write_text(CLEAN)
    quiet = _fourc_run_check(f"mpirun -np 2 /home/alexander/4C/build/4C {tmp_path}/ok.yaml out",
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
    assert "def _fourc_deck_write_check" not in src and "def _fourc_run_check" not in src


def test_a_console_redirected_to_a_file_is_read_from_that_file(tmp_path):
    (tmp_path / "side_A").mkdir()
    (tmp_path / "side_A" / "slab.4C.yaml").write_text(BAD)
    (tmp_path / "side_A" / "run_4C.log").write_text(STOP)
    cmd = "cd side_A && stdbuf -oL -eL /home/alexander/4C/build/4C slab.4C.yaml out > run_4C.log 2>&1"
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
