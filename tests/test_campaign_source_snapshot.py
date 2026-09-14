"""Blind runs execute one immutable, identified source build."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "campaign3_blind"))

import run_blind as R  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def test_snapshot_is_committed_content_and_dirty_source_is_refused(tmp_path):
    repo = tmp_path / "repo"
    files = {
        "src/server.py": "BUILD = 'committed'\n",
        "data/catalog.json": "{}\n",
        "langgraph_eval/agent.py": "PROMPT = 'committed'\n",
        "campaign3_blind/host_hygiene.py": "HYGIENE = 'committed'\n",
        "campaign3_blind/phase.py": "PHASE = 'committed'\n",
        "campaign3_blind/run_blind.py": "RUNNER = 'committed'\n",
        "scripts/blind_keys.py": "CUSTODY = 'committed'\n",
        "benchmarks/spent_problem_oracle.py": "EXACT = 'developer only'\n",
        "tests/test_spent_problem.py": "EXPECTED = 'developer only'\n",
    }
    for relative, content in files.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=openPASO Test", "-c",
         "user.email=openpaso-test@example.invalid", "commit", "-qm", "fixture")

    build = R.prepare_source_snapshot(repo, tmp_path / "snapshots")
    snapshot = Path(build["snapshot_path"])
    assert (snapshot / "src/server.py").read_text() == files["src/server.py"]
    assert not (snapshot / "benchmarks").exists()
    assert not (snapshot / "tests").exists()
    assert build["git_commit"] == _git(repo, "rev-parse", "HEAD")
    assert len(build["source_sha256"]) == 64

    (snapshot / "src/server.py").write_text("BUILD = 'tampered cache'\n")
    with pytest.raises(R.SourceBuildError, match="integrity"):
        R.prepare_source_snapshot(repo, tmp_path / "snapshots")
    (snapshot / "src/server.py").write_text(files["src/server.py"])

    (repo / "src/server.py").write_text("BUILD = 'changed live'\n")
    assert (snapshot / "src/server.py").read_text() == files["src/server.py"]
    with pytest.raises(R.SourceBuildError, match="uncommitted"):
        R.prepare_source_snapshot(repo, tmp_path / "snapshots")


def test_dune_baseline_is_reused_and_corruption_is_rebuilt(tmp_path, monkeypatch):
    interpreter = tmp_path / "dune-python"
    interpreter.write_text("fixture")
    builds = 0

    def fake_run(args, **kwargs):
        nonlocal builds
        if "import importlib.metadata" in args[-1]:
            return subprocess.CompletedProcess(
                args, 0, stdout="Python fixture\ndune-fem fixture\n", stderr="")
        builds += 1
        generated = Path(kwargs["env"]["DUNE_PY_DIR"]) / "dune-py"
        generated.mkdir()
        (generated / "neutral-module.so").write_bytes(b"neutral-cache")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(R.subprocess, "run", fake_run)
    cache = tmp_path / "cache"
    first = R.prepare_dune_cache_baseline(interpreter, cache)
    second = R.prepare_dune_cache_baseline(interpreter, cache)
    assert first == second and builds == 1

    (Path(first["baseline_path"]) / "neutral-module.so").write_bytes(b"changed")
    repaired = R.prepare_dune_cache_baseline(interpreter, cache)
    assert builds == 2
    assert repaired["dune_cache_sha256"] == first["dune_cache_sha256"]

    manifest = cache / f"{repaired['runtime_fingerprint']}.json"
    manifest.write_text("not json")
    R.prepare_dune_cache_baseline(interpreter, cache)
    assert builds == 3


def test_population_lock_refuses_a_second_source_hash(tmp_path):
    lock = tmp_path / "population.json"
    first = {"git_commit": "a" * 40, "source_sha256": "1" * 64,
             "dune_cache_sha256": "3" * 64,
             "dune_runtime_fingerprint": "4" * 64,
             "problems_root": "/problems",
             "problems_sha256": "5" * 64}
    same = dict(first)
    changed = {"git_commit": "b" * 40, "source_sha256": "2" * 64}

    R.lock_population_source(lock, first)
    R.lock_population_source(lock, same)
    with pytest.raises(R.SourceBuildError, match="already locked"):
        R.lock_population_source(lock, changed)


def test_cell_lock_refuses_a_duplicate_process(tmp_path):
    run = tmp_path / "runs" / "FE1_27b_MCP_seed1"
    first = R.acquire_cell_lock(run)
    try:
        with pytest.raises(R.SourceBuildError, match="already running"):
            R.acquire_cell_lock(run)
    finally:
        first.close()
    second = R.acquire_cell_lock(run)
    second.close()


def test_problem_draw_hash_detects_mid_population_mutation(tmp_path):
    problems = tmp_path / "problems"
    task = problems / "FE1" / "task.txt"
    task.parent.mkdir(parents=True)
    task.write_text("first task\n")
    expected = R._tree_sha256(problems)
    assert R.verify_problem_draw(problems, expected) == expected

    task.write_text("changed task\n")
    with pytest.raises(R.SourceBuildError, match="problem draw changed"):
        R.verify_problem_draw(problems, expected)


def test_live_trajectory_stream_closes_explicitly_and_idempotently(tmp_path):
    path = tmp_path / "trajectory_live.txt"
    live = R.TrajLiveLog(path)
    live.note("before close")

    live.close()
    live.close()

    assert live._f.closed
    assert path.read_text() == "before close\n"
