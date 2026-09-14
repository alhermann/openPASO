"""Campaign MCP calls must share one server process.

The critic registry is held in memory by the openPASO server.  A review submitted
on one stdio connection cannot authorize a run made on a newly spawned server.
This test exercises the same transport boundary as the campaign: submit a
review, then redeem its token on a small real solve through one session.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "langgraph_eval"))


DECK = """
import skfem
from skfem import Basis, BilinearForm, ElementTriP1, LinearForm, asm, condense, solve
from skfem.helpers import dot, grad
import sys as _sys; from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parent))
import backend_probe  # noqa: E402

mesh = skfem.MeshTri().refined(2)
basis = Basis(mesh, ElementTriP1())

@BilinearForm
def stiffness(u, v, w):
    return dot(grad(u), grad(v))

@LinearForm
def load(v, w):
    return v

matrix = asm(stiffness, basis)
rhs = asm(load, basis)
field = solve(*condense(matrix, rhs, D=basis.get_dofs()))
mesh.save("solution.vtu", {"u": field})
"""

FINDINGS = (
    "Checked the weak form, homogeneous boundary constraints, P1 space, "
    "source sign, mesh resolution, and output field. The setup is suitable "
    "for this transport-level critic-registry test."
)


def _text(result) -> str:
    """Return the JSON text from a converted MCP tool result."""
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        return "".join(
            str(block.get("text", "")) if isinstance(block, dict) else str(block)
            for block in result
        )
    return str(result)


def test_critic_review_survives_until_the_run(tmp_path, monkeypatch):
    pytest.importorskip("langchain_mcp_adapters")
    import agent

    session_factory = getattr(agent, "openpaso_mcp_tools_session", None)
    assert session_factory is not None, (
        "the campaign exposes only connection-per-call MCP tools; add one "
        "persistent-session context for the lifetime of an agent run"
    )
    marker = "MCP_CROSS_CELL_MARKER_836db1"
    credential = "OPENROUTER_SECRET_MARKER_3e68f2"
    monkeypatch.setenv("OPENROUTER_API_KEY", credential)
    sensitive_names = (
        "GIT_ASKPASS",
        "VSCODE_GIT_ASKPASS_EXTRA_ARGS",
        "VSCODE_GIT_ASKPASS_MAIN",
        "VSCODE_GIT_ASKPASS_NODE",
        "SSH_AUTH_SOCK",
        "XAUTHORITY",
    )
    for name in sensitive_names:
        monkeypatch.setenv(name, credential)
    sibling = tmp_path.parent / "adjacent_cell" / "solution.txt"
    sibling.parent.mkdir()
    sibling.write_text(marker)

    async def exercise():
        async with session_factory(tmp_path) as tools:
            by_name = {tool.name: tool for tool in tools}
            assert set(by_name) == set(agent.CAMPAIGN_MCP_TOOL_ALLOWLIST)
            assert not ({"coupled_solve", "transfer_field", "setup_backend",
                         "reload_catalog", "rediscover_backends"} & set(by_name))
            cross_cell = await by_name["run_simulation"].ainvoke({
                "solver": "skfem",
                "input_content": (
                    "import json, os\n"
                    "from pathlib import Path\n"
                    "try:\n"
                    f"    print(Path({str(sibling)!r}).read_text())\n"
                    "except OSError:\n"
                    "    print('sibling-blocked')\n"
                    f"names = {sensitive_names!r}\n"
                    "leaked = [name for name in names if name in os.environ]\n"
                    "json.dump(leaked, open('sensitive_env_names.json', 'w'))\n"
                ),
                "job_name": "cross_cell_probe",
            })
            assert marker not in _text(cross_cell)
            assert credential not in _text(cross_cell)
            reports = list(tmp_path.rglob("sensitive_env_names.json"))
            assert len(reports) == 1
            assert json.loads(reports[0].read_text()) == []

            review = json.loads(_text(
                await by_name["submit_critic_review"].ainvoke({
                    "solver": "skfem",
                    "setup": DECK,
                    "findings": FINDINGS,
                })))
            assert review["accepted"] is True

            return json.loads(_text(await by_name["run_simulation"].ainvoke({
                "solver": "skfem",
                "input_content": DECK,
                "job_name": "persistent_critic",
                "critic_token": review["critic_token"],
            })))

    result = asyncio.run(exercise())

    assert result["trustworthy_result"] is True, result["critic_review"]
    assert "token redeemed" in result["critic_review"]


def test_mcp_couple_persists_native_participant_output(tmp_path):
    pytest.importorskip("langchain_mcp_adapters")
    import agent

    runtime_python = str(backend_probe.openpaso_python())
    participants = []
    for name, marker in (("A", "NATIVE_A_SIGNATURE = 17"),
                         ("B", "NATIVE_B_SIGNATURE = 23")):
        work = tmp_path / name
        work.mkdir()
        (work / "run.py").write_text(
            "import json\n"
            f"print({marker!r})\n"
            "json.dump({'field_name': 'u', 'coordinates': [[0.0]], "
            "'values': [1.0]}, open('exports.json', 'w'))\n")
        participants.append({
            "name": name,
            "command": [runtime_python, "run.py"],
            "work_dir": str(work),
            "imports_from": ["B" if name == "A" else "A"],
        })

    async def exercise():
        async with agent.openpaso_mcp_tools_session(tmp_path) as tools:
            couple = next(tool for tool in tools if tool.name == "couple")
            valid = json.loads(_text(await couple.ainvoke({
                "participants": json.dumps(participants),
                "max_iter": 4,
                "tol": 1e-9,
                "probe": False,
                "history_path": str(tmp_path / "residual_level1.csv"),
            })))
            escaped = json.loads(_text(await couple.ainvoke({
                "participants": json.dumps(participants),
                "max_iter": 4,
                "tol": 1e-9,
                "probe": False,
                "history_path": str(tmp_path.parent / "outside.csv"),
            })))
            return valid, escaped

    try:
        result, escaped = asyncio.run(exercise())
    finally:
        agent.cleanup_sandbox_scratch(tmp_path)

    assert result["converged"] is True
    history = (tmp_path / "residual_level1.csv").read_text().splitlines()
    assert history[0] == "iteration,interface_residual"
    assert len(history) >= 2
    assert all("nan" not in row.lower() and "inf" not in row.lower()
               for row in history[1:])
    assert result["history_file"]["rows_written"] == len(history) - 1
    assert result["history_file"]["driver_iterations"] == result["iterations"]
    assert result["history_file"]["nonfinite_omitted"] == 1
    assert "outside this task" in escaped["history_file"]["error"]
    assert not (tmp_path.parent / "outside.csv").exists()
    for name, marker in (("A", "NATIVE_A_SIGNATURE = 17"),
                         ("B", "NATIVE_B_SIGNATURE = 23")):
        log = Path(result["participant_output_logs"][name])
        assert log.is_file()
        assert marker in log.read_text()