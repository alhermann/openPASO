"""The moment a per-level run log is written, the write check names it when it is another level's console
(rounds 29-33: six refined three-level couplings graded as an unchanged mesh for run logs written last,
from one level, unaudited)."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _w(d: Path, name: str, text: str) -> None:
    d.mkdir(parents=True, exist_ok=True); (d / name).write_text(text)


def test_the_write_check_fires_only_for_the_wrong_file(tmp_path):
    from tools.workspace_advisor import _wrong_level_run_log_check
    for k, n in ((1, 54), (2, 187)):
        _w(tmp_path / "side_A", f"participant_output_level{k}.log", f"4C banner\nNDOF = {n}\n")
    _w(tmp_path, "run_level2_A.log", "4C banner\nNDOF = 54\n")
    out = _wrong_level_run_log_check(tmp_path, tmp_path / "run_level2_A.log")
    assert out.startswith("\n[write check] RUN LOG FROM THE WRONG LEVEL: run_level2_A.log carries NDOF 54"), out
    assert "side_A/participant_output_level2.log" in out
    _w(tmp_path, "run_level1_A.log", "4C banner\nNDOF = 54\n")
    assert _wrong_level_run_log_check(tmp_path, tmp_path / "run_level1_A.log") == ""
    assert _wrong_level_run_log_check(tmp_path, tmp_path / "solution_level1_A.csv") == ""
