"""The coupling tool keeps each side's captured solver console per level.

Measured on four development cells: participant_output.log holds only the latest
level's console, the agents wrote every level's run log from it at the end, and
every level carried the finest level's DOF count. When the call names a level
(iface_level, or a history_path like residual_level<k>.csv) the same console is
also kept as participant_output_level<k>.log next to exports.json."""
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TOY = '''import json, sys
from pathlib import Path
imp = json.loads(Path("imports.json").read_text()) if Path("imports.json").is_file() else {}
other = 0.0
for d in imp.values():
    v = d.get("values") or [0.0]
    other = float(v[0])
mine = {A}
print("toy solver console: NDOF = 7")
Path("exports.json").write_text(json.dumps({"field_name": "u", "n_points": 1, "coordinates": [[0.5, 0.0]],
    "values": [mine], "normal_fluxes": [0.0]}))
'''


class _CollectingMCP:
    def __init__(self):
        self.tools = {}

    def tool(self, *a, **k):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


@pytest.fixture(scope="module")
def couple_tool():
    os.chdir(ROOT)
    from core.registry import load_all_backends
    load_all_backends()
    from tools.consolidated import register_consolidated_tools
    mcp = _CollectingMCP()
    register_consolidated_tools(mcp)
    return mcp.tools["couple"]


def test_each_side_keeps_its_console_for_the_named_level(tmp_path, couple_tool, monkeypatch):
    import asyncio, inspect
    monkeypatch.chdir(tmp_path)
    for name, expr in (("A", "0.3 * other + 1.0"), ("B", "0.5 * other + 0.2")):
        d = tmp_path / f"side_{name}"
        d.mkdir()
        (d / "part.py").write_text(TOY.replace("{A}", expr))
    parts = [{"name": "A", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_A"), "imports_from": ["B"]},
             {"name": "B", "command": [sys.executable, "part.py"], "work_dir": str(tmp_path / "side_B"), "imports_from": ["A"]}]
    r = couple_tool(participants=json.dumps(parts), max_iter=60, tol=1e-8, critic_approved=True, iface_level=2, probe=False)
    if inspect.iscoroutine(r):
        r = asyncio.run(r)
    out = json.loads(str(r))
    assert out.get("converged"), {k: out.get(k) for k in ("error", "reason", "why", "history", "verdict")}
    for name in ("A", "B"):
        per_level = tmp_path / f"side_{name}" / "participant_output_level2.log"
        assert per_level.is_file(), f"side {name}: no per-level console copy"
        assert "toy solver console: NDOF = 7" in per_level.read_text()
    assert not (tmp_path / "side_A" / "participant_output_level1.log").exists()
