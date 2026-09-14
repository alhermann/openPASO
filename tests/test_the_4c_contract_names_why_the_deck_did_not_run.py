"""The served 4C participant contract, when the run left no VTU, names the deck
defect instead of dying on an index error.

Measured on 2026-09-10 (three 27B worker trials, three different aborts): an
invented section name, a condition entry without `E:`, and a topology section
written twice. Each is mechanical, each aborts 4C on read, and the served
recovery used to fail afterwards with `list index out of range` -- the one line
an agent cannot act on. The served block below the hole now reads the captured
console log and lints the deck, and stops with the cause spelled out.
"""
import glob
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def _finish_check_block() -> str:
    from tools import consolidated as C
    text = C._coupling_participant_script("fourc")
    assert "4C DID NOT FINISH" in text, "the served 4C contract lost its finish check"
    # the diagnosis is a function defined ABOVE the hole (so it can also run at
    # exit); exec its definition and run it the way the post-hole check does
    a = text.index("import atexit, glob, sys")
    b = text.index("def _diagnose_at_exit", a)
    return text[a:b] + "\nraise SystemExit(why_4c_did_not_finish())\n"


DECK = """TITLE:
  - my deck
NODE COORDS:
  - "NODE 1 COORD 0.0 0.0 0.0"
  - "NODE 2 COORD 1.0 0.0 0.0"
DNODE-NODE TOPOLOGY:
  - "NODE 1 DNODE 1"
DESIGN POINT DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 1
  - E: 7
    NUMDOF: 1
  - NUMDOF: 1
DNODE-NODE TOPOLOGY:
  - "NODE 2 DNODE 2"
"""

LOG = """some banner
PROC 0 ERROR in 4C_io_input_file.cpp, line 492:
Section 'DNODE-NODE TOPOLOGY' is defined more than once.
------------------
"""


def _run_block(tmp_path: Path, deck: str, log: str, cfg: dict) -> str:
    os.chdir(tmp_path)
    (tmp_path / "deck.4C.yaml").write_text(deck)
    if log:
        (tmp_path / "run.log").write_text(log)
    with pytest.raises(SystemExit) as exc:
        exec(_finish_check_block(), {"glob": glob, "CFG": cfg, "__name__": "served"})
    return str(exc.value)


def test_the_check_names_the_duplicate_the_missing_id_and_the_orphan_e_id(tmp_path):
    msg = _run_block(tmp_path, DECK, LOG, {})
    assert msg.startswith("4C DID NOT FINISH")
    assert "defined more than once" in msg              # 4C's own line, lifted from the log
    assert "written more than once" in msg and "DNODE-NODE TOPOLOGY" in msg
    assert "without `E: <id>`" in msg                   # the third entry has no E
    assert "E id(s) 7" in msg                           # E: 7 has no DNODE 7 anywhere


def test_without_a_log_the_check_still_says_what_to_capture(tmp_path):
    msg = _run_block(tmp_path, "TITLE:\n  - x\n", "", {})
    assert "stdbuf -oL -eL" in msg and "read that log from the top" in msg


@pytest.mark.skipif(not Path("/home/alexander/4C/build/4C").is_file(), reason="no 4C binary here")
def test_with_the_binary_an_invented_section_is_named_from_4c_p(tmp_path):
    deck = DECK.replace("DESIGN POINT DIRICH CONDITIONS", "THERMAL FLUX CALC LINE CONDITIONS")
    msg = _run_block(tmp_path, deck, "", {"fourc_bin": "/home/alexander/4C/build/4C",
                                          "fourc_ld": "/opt/4C-dependencies/lib"})
    assert "`4C -p`" in msg and "THERMAL FLUX CALC LINE CONDITIONS" in msg
    assert "FUNCT" not in msg                            # FUNCT<n> sections are pattern-named, never flagged


def test_it_is_inert_once_a_vtu_exists(tmp_path):
    from tools import consolidated as C
    text = C._coupling_participant_script("fourc")
    a = text.index("# ── DID 4C FINISH? ── the check defined above the hole"); b = text.index("# The VTU name is", a)
    os.chdir(tmp_path)
    (tmp_path / "out-vtk-files").mkdir()
    (tmp_path / "out-vtk-files" / "scatra-00001-0.vtu").write_text("<VTKFile/>")
    exec(text[a:b], {"glob": glob, "why_4c_did_not_finish": lambda: "SHOULD NOT RUN", "__name__": "served"})   # no SystemExit


def test_the_diagnosis_also_runs_at_exit_when_the_worker_stops_early():
    """A worker that raises its own 'solver failed' before the post-hole check
    never reached the diagnosis. The served head registers it at exit."""
    from tools import consolidated as C
    text = C._coupling_participant_script("fourc")
    assert "atexit.register(_diagnose_at_exit)" in text
    assert text.index("atexit.register(_diagnose_at_exit)") < text.index("openPASO DOES NOT SERVE THIS")


def test_the_4c_contract_takes_either_role_from_its_config():
    """Round 8 (2026-09-10): the tasks that put 4C on the Dirichlet side found a
    Neumann-only scaffold in the copy-now block and hand-rolled the handshake in
    all six runs. The scaffold now reads its role from config.json; both roles
    were executed against a manufactured solution before this test existed."""
    from tools import consolidated as C
    t = C._coupling_participant_script("fourc")
    assert 'SIDE = CFG.get("side", "neumann")' in t
    assert "def partner_value(" in t and "def partner_flux(" in t
    assert '"values": (vals if SIDE == "neumann" else [])' in t
    assert 'if SIDE == "neumann" and _chk_qin.size' in t
    assert 'if SIDE == "dirichlet" and _chk_qin.shape' in t
    assert "DESIGN POINT DIRICH per INTERIOR" in t          # the Dirichlet bullet survives the lean view
    assert "the recovered interface flux does not match the load you" in t   # Neumann load-consistency check
