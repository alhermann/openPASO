"""Round 29: four cells coupled three refined levels and handed in run logs that were all the level-1
console; the grader read an unchanged mesh. The audit compares each run log's NDOF with the side's captured
per-level console and names the file to copy."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _w(d: Path, name: str, text: str) -> None:
    d.mkdir(parents=True, exist_ok=True); (d / name).write_text(text)


def test_a_run_log_copied_from_another_level_is_named_with_the_right_source(tmp_path):
    from tools.result_audit import wrong_level_run_log_findings
    for k, n in ((1, 54), (2, 187), (3, 693)):
        _w(tmp_path / "side_A", f"participant_output_level{k}.log", f"4C banner\nFinalised step 1 / 1 | time 1.0 | dt 1.0 | nlniter 1 | wct 0.1\nNDOF = {n}\n")
        _w(tmp_path, f"run_level{k}_A.log", "4C banner\nNDOF = 54\n")          # every level got level 1's console
    out = wrong_level_run_log_findings(tmp_path)
    assert len(out) == 2, out
    assert "run_level2_A.log carries NDOF 54" in out[0]["finding"] and "level 2 says NDOF 187" in out[0]["finding"]
    assert "it is level 1's console" in out[0]["finding"] and "side_A/participant_output_level2.log" in out[0]["finding"]
    # matching logs are silent
    _w(tmp_path, "run_level2_A.log", "4C banner\nNDOF = 187\n"); _w(tmp_path, "run_level3_A.log", "4C banner\nNDOF = 693\n")
    assert wrong_level_run_log_findings(tmp_path) == []


def test_a_run_log_of_the_agents_own_words_names_the_console_copy_to_use(tmp_path):
    """Round 46, C1 7191: three run logs were the parent's summaries; the driver's per-level console copies sat in
    the side directories and the parent called the defect unfixable. The finding names the file to copy."""
    from tools.result_audit import audit
    (tmp_path / "side_B").mkdir()
    (tmp_path / "side_B" / "participant_output_level1.log").write_text("iteration: 5\n--- stdout ---\n[2026-09-12 04:04:07.373] [info] Extract basic topology: 384->384\nNDOF = 351\n")
    (tmp_path / "run_level1_B.log").write_text("Level 1 side B: FEniCSx ran fine, converged.\nNDOF = 351\n")
    (tmp_path / "run_level1_A.log").write_text("Trilinos Version: f4d64271518 (git SHA1)\nFinalised step 1 / 1 | time 1.0 | dt 1.0 | nlniter 2 | wct 0.1\nNDOF = 243\n")
    (tmp_path / "residual_level1.csv").write_text("iteration,residual\n1,0.5\n2,0.1\n3,0.01\n4,1e-7\n")
    fs = [f for f in audit(tmp_path).get("findings", []) if "run log run_level1_B.log" == f.get("sequence")]
    assert fs, audit(tmp_path).get("findings")
    assert "NO LINE ANY SOLVER EMITS" in fs[0]["finding"]
    assert str(tmp_path / "side_B" / "participant_output_level1.log") in fs[0]["finding"], fs[0]["finding"]
    assert "copy that file over this one" in fs[0]["finding"]
