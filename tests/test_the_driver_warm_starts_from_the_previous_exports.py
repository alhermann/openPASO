"""A coupling run in work directories that already hold exports.json (the previous
mesh level's converged interface state) seeds its first imports from them and
says so; the relaxation still starts fresh at iteration 1. Measured need: the
development cells reached level 2 or 3 and ran out of wall clock on couplings
that started every level from nothing."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOY = '''import json
from pathlib import Path
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = 0.0
for d in imp.values():
    other = float((d.get("values") or [0.0])[0])
mine = COEF * other + OFFSET
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1, "coordinates": [[0.5, 0.0]],
    "values": [mine], "normal_fluxes": [0.0]}))
'''


def _pair(tmp_path):
    from core.coupling_driver import Participant
    parts = []
    for name, coef, off in (("A", 0.6, 1.0), ("B", 0.7, 0.2)):
        d = tmp_path / f"side_{name}"
        d.mkdir(exist_ok=True)
        (d / "part.py").write_text(TOY.replace("COEF", str(coef)).replace("OFFSET", str(off)))
        parts.append(Participant(name=name, command=[sys.executable, "part.py"], work_dir=d,
                                 imports_from=["B" if name == "A" else "A"]))
    return parts


def test_a_second_run_in_the_same_directories_starts_from_the_first_ones_answer(tmp_path):
    from core.coupling_driver import run_coupling
    parts = _pair(tmp_path)
    cold = run_coupling(parts, max_iter=200, tol=1e-9, accelerator="aitken", probe=False)
    assert cold.converged and not any("warm start" in n for n in cold.notes)
    assert all((p.work_dir / "exports.json").is_file() for p in parts)
    warm = run_coupling(parts, max_iter=200, tol=1e-9, accelerator="aitken", probe=False)
    assert warm.converged
    assert any("warm start" in n and "A, B" in n for n in warm.notes), warm.notes
    assert len(warm.history) < len(cold.history), (len(cold.history), len(warm.history))
    # the two runs agree on the fixed point
    a_cold = json.loads((parts[0].work_dir / "exports.json").read_text())["values"][0]
    assert abs(a_cold - (1.0 + 0.6 * 0.2) / (1 - 0.6 * 0.7)) < 1e-6


def test_a_stale_or_empty_export_does_not_seed(tmp_path):
    from core.coupling_driver import run_coupling
    parts = _pair(tmp_path)
    (parts[0].work_dir / "exports.json").write_text("{not json")
    (parts[1].work_dir / "exports.json").write_text(json.dumps({"field_name": "u", "n_points": 0, "coordinates": [], "values": [], "normal_fluxes": []}))
    r = run_coupling(parts, max_iter=200, tol=1e-9, accelerator="aitken", probe=False)
    assert r.converged and not any("warm start" in n for n in r.notes)
