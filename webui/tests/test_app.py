"""Smoke tests for the WebUI.

Run from repo root: .venv-lg/bin/pytest webui/tests -v

These tests use FastAPI's TestClient (HTTP) and websockets for the
streamed channel. No vLLM, no GPU; the runner uses the mock LLM so
the end-to-end spawn_subagent chain is exercised in <1 s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pytest
from fastapi.testclient import TestClient

from webui import app as webui_app
from webui import config, files, sessions, viz


@pytest.fixture
def client():
    return TestClient(webui_app.app)


# ───────────────────────────────────────────────────────────────────
# Config endpoints
# ───────────────────────────────────────────────────────────────────
def test_models(client):
    r = client.get("/api/models").json()
    assert any(m["id"].startswith("qwen2.5-") for m in r["models"])
    assert r["default"] in [m["id"] for m in r["models"]]


def test_the_fake_model_is_never_offered_to_a_person(client):
    """The mock answers one canned turn and runs no solver.

    It was the default, and a fabricated run was indistinguishable from real
    work: same layout, same "Answer" heading, same "finished". It stays for
    these tests and must never appear in the interface again."""
    ids = [m["id"] for m in client.get("/api/models").json()["models"]]
    assert "mock" not in ids, (
        "the fake model is selectable again; a person can produce a result "
        "that looks real and computed nothing")


def test_the_solver_count_is_measured_not_asserted(client):
    """The first screen used to state nine solvers from a hardcoded array.

    It must come from the registry, and when the check cannot run it must say
    so rather than guess."""
    r = client.get("/api/solvers").json()
    assert "ok" in r and "solvers" in r
    if r["ok"]:
        assert all("status" in s and "name" in s for s in r["solvers"])


def test_mcp_servers(client):
    r = client.get("/api/mcp_servers").json()
    assert any(s["id"] == "openpaso" for s in r["servers"])


def test_modes(client):
    r = client.get("/api/modes").json()
    assert set(r["modes"]) == {"plan", "accept", "autonomous"}


# ───────────────────────────────────────────────────────────────────
# Sandbox file safety
# ───────────────────────────────────────────────────────────────────
def test_sandbox_traversal_escape_blocked(client):
    r = client.get("/api/files", params={"rel": "../../etc/passwd"})
    assert r.status_code in (403, 422)


def test_list_root(client):
    r = client.get("/api/files").json()
    assert r["exists"]
    assert isinstance(r["entries"], list)


def test_classify():
    p = config.SANDBOX_ROOT / "x" / "out.vtu"
    assert files.classify(p) == "vtk"
    p2 = config.SANDBOX_ROOT / "x" / "data.csv"
    assert files.classify(p2) == "table"


# ───────────────────────────────────────────────────────────────────
# Viz / parameter extraction
# ───────────────────────────────────────────────────────────────────
def test_extract_params():
    src = (
        "import numpy as np\n"
        "N = 32\n"
        "dt = 0.005\n"
        "nu = 1.0e-3\n"
        "L = 10.0  # length\n"
        "alpha=2  # inline\n"
        "name = 'no'\n"
        "L = 99  # duplicate ignored\n"
    )
    params = viz.extract_params(src)
    names = [p["name"] for p in params]
    assert names == ["N", "dt", "nu", "L", "alpha"]
    assert all("min" in p and "max" in p for p in params)


# ───────────────────────────────────────────────────────────────────
# Sessions round-trip
# ───────────────────────────────────────────────────────────────────
def test_session_lifecycle(client):
    n = client.post("/api/sessions", json={"model": "mock"}).json()
    sid = n["id"]
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["id"] == sid
    assert got["model"] == "mock"
    listed = client.get("/api/sessions").json()["sessions"]
    assert any(s["id"] == sid for s in listed)
    d = client.delete(f"/api/sessions/{sid}").json()
    assert d["deleted"]


# ───────────────────────────────────────────────────────────────────
# Runner end-to-end with mock LLM.
#
# We exercise the agent flow directly via runner.build_agent_for_session
# + runner.stream_turn rather than through TestClient's WebSocket. The
# TestClient's anyio-driven WS loop deadlocks against asyncio.to_thread
# inside the gated spawn_subagent tool; the runner itself works fine
# under plain asyncio, which is what the live server uses.
# ───────────────────────────────────────────────────────────────────
import asyncio
import tempfile
from contextlib import asynccontextmanager

from webui.runner import (ApprovalGate, build_agent_for_session,
                          open_agent_for_session, stream_turn)


def _run_turn(*, mode, prompt, approve_first=False):
    seen, events = [], []

    async def main():
        async def emitter(e):
            seen.append(e["type"])
            events.append(e)

        gate = ApprovalGate()

        def get_mode():
            return mode

        with tempfile.TemporaryDirectory() as d:
            agent = build_agent_for_session(
                model="mock", mcp_on=False, workdir=Path(d),
                emitter=emitter, get_mode=get_mode, gate=gate)

            turn = asyncio.create_task(stream_turn(
                agent=agent, user_text=prompt, emitter=emitter))

            if approve_first:
                # Wait until the first pending call appears, then approve.
                while turn.done() is False:
                    pending = next((e for e in events
                                    if e["type"] == "tool_call_pending"), None)
                    if pending:
                        gate.resolve(pending["call_id"], True)
                        break
                    await asyncio.sleep(0.05)
            return await turn

    final = asyncio.run(main())
    return final, seen, events


def test_runner_mock_run_accept():
    final, seen, _ = _run_turn(
        mode="accept", prompt="Plan a Poisson MMS demo. Use the critic.")
    assert "subagent_spawned" in seen, (
        f"expected spawn_subagent in {set(seen)}")
    assert "subagent_returned" in seen
    assert "done" in seen
    assert "critic approved" in final.lower()


def test_runner_mock_plan_mode_pending_then_approve():
    final, seen, events = _run_turn(
        mode="plan", prompt="Begin.", approve_first=True)
    pending = [e for e in events if e["type"] == "tool_call_pending"]
    assert pending, "no tool_call_pending event in plan mode"
    # After approval the call must have executed (tool_result) and the
    # turn must have terminated (done).
    assert any(e["type"] == "tool_result" for e in events)
    assert "done" in seen, f"missing done in {set(seen)}"


def test_webui_keeps_mcp_context_open_for_agent_lifetime(monkeypatch):
    import agent as langgraph_agent

    lifecycle = []

    @asynccontextmanager
    async def fake_mcp_session(_workdir):
        lifecycle.append("opened")
        try:
            yield []
        finally:
            lifecycle.append("closed")

    monkeypatch.setattr(
        langgraph_agent, "openpaso_mcp_tools_session", fake_mcp_session)

    async def main():
        async def emitter(_event):
            return None

        with tempfile.TemporaryDirectory() as directory:
            async with open_agent_for_session(
                    model="mock", mcp_on=True, workdir=Path(directory),
                    emitter=emitter, get_mode=lambda: "accept",
                    gate=ApprovalGate()) as built:
                assert built is not None
                assert lifecycle == ["opened"]
            assert lifecycle == ["opened", "closed"]

    asyncio.run(main())


def test_webui_cancels_active_turn_before_closing_agent():
    from webui.app import WSSession

    lifecycle = []

    class Context:
        async def __aexit__(self, *_args):
            lifecycle.append("context-closed")

    async def main():
        session = object.__new__(WSSession)
        session.agent = object()
        session.agent_context = Context()

        async def turn():
            try:
                await asyncio.Event().wait()
            finally:
                lifecycle.append("turn-stopped")

        session.turn_task = asyncio.create_task(turn())
        await asyncio.sleep(0)
        await session.close_agent()
        assert session.turn_task is None
        assert session.agent is None
        assert session.agent_context is None

    asyncio.run(main())
    assert lifecycle == ["turn-stopped", "context-closed"]
