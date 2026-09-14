#!/usr/bin/env python
"""Campaign-3 blind runner.

Executes one agent per (problem, model, condition, seed) on a BLIND task and
records everything needed for cost and energy accounting. It deliberately does
NOT grade: grading happens afterwards, offline, with grade_blind.py, which is
the only component that opens a sealed key.

Safety properties:
  * the answer keys are sealed (unreadable) while this runs — the runner
    refuses to start otherwise;
  * every run gets a fresh directory and an existing ledger is never
    overwritten (a relaunch resumes and fills gaps);
  * tokens in/out, wall time and tool calls are recorded per run;
  * the live trajectory is written as it happens, so a crash or timeout cannot
    destroy the evidence the leak audit needs.

Usage:
  ../../open-fem-agent/.venv-lg/bin/python run_blind.py \
      --model 27b --conditions BARE MCP --problems B1 B2 D1 --seed 0
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from core.env_compat import install_aliases as _install_env_aliases
_install_env_aliases()   # OASIS_* and OPENPASO_* name the same variable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

SOURCE_ARCHIVE_PATHS = (
    "src",
    "data",
    "langgraph_eval/agent.py",
    "campaign3_blind/host_hygiene.py",
    "campaign3_blind/phase.py",
    "campaign3_blind/run_blind.py",
    "scripts/blind_keys.py",
)


class SourceBuildError(RuntimeError):
    """The agent-facing build cannot be pinned reproducibly."""


def prepare_source_snapshot(repo: Path, cache: Path) -> dict:
    """Materialize and identify the committed agent-facing source build."""
    repo = repo.resolve()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "--",
         *SOURCE_ARCHIVE_PATHS],
        cwd=repo, capture_output=True, text=True)
    if status.returncode != 0:
        raise SourceBuildError(status.stderr.strip() or "git status failed")
    if status.stdout.strip():
        raise SourceBuildError(
            "uncommitted agent-facing source would make the build mutable:\n"
            + status.stdout.rstrip())

    commit_result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True)
    if commit_result.returncode != 0:
        raise SourceBuildError(
            commit_result.stderr.strip() or "cannot resolve git HEAD")
    commit = commit_result.stdout.strip()

    cache.mkdir(parents=True, exist_ok=True)
    archive_fd, archive_name = tempfile.mkstemp(
        prefix=".source-", suffix=".tar", dir=cache)
    os.close(archive_fd)
    archive = Path(archive_name)
    try:
        with archive.open("wb") as stream:
            made = subprocess.run(
                ["git", "archive", "--format=tar", commit, "--",
                 *SOURCE_ARCHIVE_PATHS],
                cwd=repo, stdout=stream, stderr=subprocess.PIPE)
        if made.returncode != 0:
            raise SourceBuildError(
                made.stderr.decode(errors="replace").strip()
                or "git archive failed")
        with archive.open("rb") as stream:
            source_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        tree_sha = _tar_tree_sha256(archive)

        target = cache / source_sha
        manifest = target / "SOURCE_BUILD.json"
        if not target.exists():
            building = Path(tempfile.mkdtemp(prefix=".extract-", dir=cache))
            try:
                with tarfile.open(archive, "r") as packed:
                    packed.extractall(building, filter="data")
                (building / "SOURCE_BUILD.json").write_text(json.dumps({
                    "git_commit": commit,
                    "source_sha256": source_sha,
                    "tree_sha256": tree_sha,
                    "paths": list(SOURCE_ARCHIVE_PATHS),
                }, indent=2) + "\n")
                try:
                    building.rename(target)
                except (FileExistsError, OSError) as exc:
                    # LOSING THE RACE IS NORMAL; ENOTEMPTY IS HOW IT LOOKS.
                    #
                    # Renaming a directory onto an EXISTING NON-EMPTY directory
                    # raises OSError(errno 39, ENOTEMPTY) on Linux, not
                    # FileExistsError, so catching only FileExistsError made
                    # every cell but the first crash on a cold cache. Measured:
                    # launching six NG1 cells at once, five died with
                    # "[Errno 39] Directory not empty: .extract-XXXX ->
                    # <sha>" before a single paid call, and one survived.
                    # A parallel round could therefore never populate a new
                    # snapshot — the reproducibility mechanism was unusable in
                    # exactly the case it exists for.
                    #
                    # The loser's copy is discarded and the winner's tree is
                    # verified below, so this is safe. A real failure — no
                    # space, no permission — leaves no target, and is re-raised.
                    if not target.is_dir():
                        shutil.rmtree(building, ignore_errors=True)
                        raise SourceBuildError(
                            f"could not install the source snapshot at "
                            f"{target}: {type(exc).__name__}: {exc}") from exc
                    shutil.rmtree(building, ignore_errors=True)
            except Exception:
                shutil.rmtree(building, ignore_errors=True)
                raise

        try:
            recorded = json.loads(manifest.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SourceBuildError(
                f"source snapshot {target} has no valid manifest: {exc}") from exc
        if (recorded.get("git_commit") != commit or
                recorded.get("source_sha256") != source_sha or
                recorded.get("tree_sha256") != tree_sha or
                _tree_sha256(target, ignore={"SOURCE_BUILD.json"}) != tree_sha):
            raise SourceBuildError(
                f"source snapshot cache integrity failure at {target}")
        return {
            "git_commit": commit,
            "source_sha256": source_sha,
            "snapshot_path": str(target),
        }
    finally:
        archive.unlink(missing_ok=True)


def verify_source_snapshot(snapshot: Path) -> dict:
    """Verify an extracted build still matches its recorded committed tree."""
    manifest = snapshot / "SOURCE_BUILD.json"
    try:
        recorded = json.loads(manifest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceBuildError(
            f"source snapshot {snapshot} has no valid manifest: {exc}") from exc
    actual = _tree_sha256(snapshot, ignore={"SOURCE_BUILD.json"})
    if actual != recorded.get("tree_sha256"):
        raise SourceBuildError(
            f"source snapshot changed after creation: expected "
            f"{recorded.get('tree_sha256')}, found {actual}")
    return recorded


def lock_population_source(lock_path: Path, source_build: dict) -> None:
    """Bind one model/phase/seed population to exactly one source hash."""
    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    guard = lock_path.with_suffix(lock_path.suffix + ".lock")
    with guard.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        expected = {
            "source_git_commit": source_build["git_commit"],
            "source_sha256": source_build["source_sha256"],
            "dune_cache_sha256": source_build.get("dune_cache_sha256"),
            "dune_runtime_fingerprint": source_build.get(
                "dune_runtime_fingerprint"),
            "problems_root": source_build.get("problems_root"),
            "problems_sha256": source_build.get("problems_sha256"),
        }
        if lock_path.exists():
            try:
                recorded = json.loads(lock_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise SourceBuildError(
                    f"population build lock is unreadable: {lock_path}: "
                    f"{exc}") from exc
            if recorded != expected:
                raise SourceBuildError(
                    f"population is already locked to "
                    f"{recorded.get('source_git_commit')} / "
                    f"{recorded.get('source_sha256')}; current build is "
                    f"{expected['source_git_commit']} / "
                    f"{expected['source_sha256']}. Use a new seed or archive "
                    f"and deliberately remove {lock_path.name}; never mix the "
                    f"two builds in one population.")
            return
        tmp = lock_path.with_suffix(lock_path.suffix + ".tmp")
        tmp.write_text(json.dumps(expected, indent=2) + "\n")
        tmp.replace(lock_path)


def acquire_cell_lock(run_dir: Path):
    """Refuse a duplicate process for the same cell before files can collide."""
    import fcntl

    lock_dir = run_dir.parent / ".cell_locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    handle = (lock_dir / f"{run_dir.name}.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise SourceBuildError(
            f"another process is already running {run_dir.name}") from exc
    return handle


def _tree_sha256(root: Path, ignore: set[str] | None = None) -> str:
    """Hash the SOURCE of a snapshot tree, not the bytecode Python leaves in it.

    IMPORTING FROM THE SNAPSHOT WRITES INTO IT. CPython caches compiled modules
    next to their source, so the moment an agent's tool process imports openPASO
    from the mounted snapshot, `src/tools/__pycache__/*.pyc` appears inside the
    tree this function hashes. The post-run `verify_source_snapshot` then finds
    a different digest and reports

        SourceSnapshotIntegrity: source snapshot changed after creation

    on a run where nothing was tampered with. Measured: every snapshot that has
    actually been used carries exactly the .pyc files written during its own
    runs -- 2 in the C2 build a874d802, 2 in the NG1 build 8300bc8e -- while
    snapshots that were built and never run are clean. So EVERY run through the
    reproducible-build path ends flagged, including NG1 runs that graded
    CORRECT, and the flag means the opposite of what it says.

    Bytecode is a derived artefact of the very source already in the digest, so
    excluding it weakens nothing: a changed .py changes the hash whether or not
    its .pyc is counted. What is excluded is named here rather than guessed at
    a call site, so creation and verification cannot drift apart.
    """
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda p: str(p.relative_to(root))):
        relative_text = str(path.relative_to(root))
        if relative_text in (ignore or set()):
            continue
        parts = path.relative_to(root).parts
        if "__pycache__" in parts or path.suffix in (".pyc", ".pyo"):
            continue
        relative = relative_text.encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        if path.is_symlink():
            digest.update(b"L" + os.readlink(path).encode())
        elif path.is_file():
            digest.update(b"F")
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        elif path.is_dir():
            digest.update(b"D")
    return digest.hexdigest()


def _tar_tree_sha256(archive: Path) -> str:
    """Hash archive members in the same form as their extracted tree."""
    digest = hashlib.sha256()
    with tarfile.open(archive, "r") as packed:
        members = sorted(packed.getmembers(), key=lambda m: m.name.rstrip("/"))
        for member in members:
            relative = member.name.rstrip("/").encode()
            digest.update(len(relative).to_bytes(4, "big"))
            digest.update(relative)
            if member.issym():
                digest.update(b"L" + member.linkname.encode())
            elif member.isfile():
                digest.update(b"F")
                stream = packed.extractfile(member)
                if stream is None:
                    raise SourceBuildError(
                        f"cannot read {member.name} from source archive")
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            elif member.isdir():
                digest.update(b"D")
            else:
                raise SourceBuildError(
                    f"unsupported source archive member: {member.name}")
    return digest.hexdigest()


def verify_problem_draw(root: Path, expected_sha256: str | None) -> str:
    """Return the draw hash or refuse when its public bytes have changed."""
    actual = _tree_sha256(root.resolve())
    if not expected_sha256 or actual != expected_sha256:
        raise SourceBuildError(
            "public problem draw changed after the population was locked: "
            f"expected {expected_sha256}, found {actual}")
    return actual


def prepare_dune_cache_baseline(interpreter: Path, cache: Path) -> dict:
    """Build one task-neutral DUNE JIT cache for private per-cell copies."""
    import fcntl

    probe = subprocess.run(
        [str(interpreter), "-c",
         "import importlib.metadata,sys; "
         "print(sys.version); "
         "print(importlib.metadata.version('dune-fem'))"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        raise SourceBuildError(
            "cannot identify the DUNE runtime: " + probe.stderr[-500:])
    fingerprint = hashlib.sha256(
        (str(interpreter.resolve()) + "\n" + probe.stdout).encode()).hexdigest()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / fingerprint
    manifest = cache / f"{fingerprint}.json"
    lock_path = cache / f"{fingerprint}.lock"

    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.is_dir():
            try:
                recorded = json.loads(manifest.read_text())
            except (OSError, json.JSONDecodeError):
                recorded = {}
            tree_sha = _tree_sha256(target)
            if recorded.get("tree_sha256") == tree_sha:
                return {"baseline_path": str(target),
                        "dune_cache_sha256": tree_sha,
                        "runtime_fingerprint": fingerprint}
            shutil.rmtree(target)
        manifest.unlink(missing_ok=True)

        building = cache / f".build-{fingerprint}"
        shutil.rmtree(building, ignore_errors=True)
        building.mkdir()
        env = os.environ.copy()
        env["DUNE_PY_DIR"] = str(building)
        try:
            made = subprocess.run(
                [str(interpreter), "-c",
                 "from dune.grid import structuredGrid; "
                 "grid=structuredGrid([0,0],[1,1],[2,2]); "
                 "assert grid.size(0) == 4"],
                env=env, capture_output=True, text=True, timeout=900)
            if made.returncode != 0:
                raise SourceBuildError(
                    "neutral DUNE cache build failed: "
                    + (made.stdout + made.stderr)[-1500:])
            generated = building / "dune-py"
            if not generated.is_dir():
                raise SourceBuildError(
                    f"DUNE created no cache under {building}")
            generated.rename(target)
            tree_sha = _tree_sha256(target)
            manifest.write_text(json.dumps({
                "runtime_fingerprint": fingerprint,
                "tree_sha256": tree_sha,
                "probe": "structuredGrid([0,0],[1,1],[2,2])",
            }, indent=2) + "\n")
            return {"baseline_path": str(target),
                    "dune_cache_sha256": tree_sha,
                    "runtime_fingerprint": fingerprint}
        finally:
            shutil.rmtree(building, ignore_errors=True)

_envf = ROOT / ".env"
if _envf.exists():
    for _l in _envf.read_text().splitlines():
        if "=" in _l and not _l.lstrip().startswith("#"):
            k, v = _l.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# The agent definition (both arms) MUST come from the checkout this campaign
# lives in. The default used to be a hard-coded path to a DIFFERENT checkout
# that exists on this machine, so any launch not going through the one
# untracked driver script silently ran an uncommitted agent while the
# pre-registration attested to this tree. Self-locate, and refuse a mismatch.
REPO = Path(os.environ.get("OPENPASO_REPO", str(ROOT))).resolve()
if REPO != ROOT.resolve():
    sys.exit(
        f"REFUSING TO RUN — OPENPASO_REPO points at {REPO}, but this campaign "
        f"lives in {ROOT.resolve()}. The agent definition, the knowledge under "
        f"test and the pre-registration must all come from one checkout. "
        f"Unset OPENPASO_REPO or set it to {ROOT.resolve()}.")
sys.path.insert(0, str(REPO / "langgraph_eval"))
import agent as _agent                                            # noqa: E402
from agent import build_bare_agent, build_mcp_agent               # noqa: E402
from langchain_openai import ChatOpenAI                           # noqa: E402
from langchain_core.callbacks import (UsageMetadataCallbackHandler,  # noqa: E402
                                      BaseCallbackHandler)

OR_MODELS = {
    "27b": "qwen/qwen3.5-27b",
    "122b": "qwen/qwen3.5-122b-a10b",
    "397b": "qwen/qwen3.5-397b-a17b",
}

# LangGraph counts one node execution per step, so a tool-calling agent burns
# roughly two per tool call. Named because the prompt quotes it.
#
# THERE MUST BE ONE BUDGET, AND IT IS THE CLOCK. At 250 the step cap bound
# FIRST and silently: round 2 opened with NG1-BARE cut off at 124 calls having
# used 30% of its 45 minutes — with a COMPLETE submission already on disk —
# and NG2-BARE cut off at 66% while still solving. Measured cost across the
# first eight runs of that round is 6.5-29 s per call (one outlier at 102),
# so 2700 s buys roughly 90-400 calls. A 500-call ceiling therefore sits
# above what the clock can ever reach, which is the point: the agent is told
# 45 minutes, and 45 minutes is what actually stops it.
#
# In round 1 the cap almost never bound because the models quit voluntarily at
# ~37% of budget. Telling them their budget fixed that and promptly exposed
# the second, unstated limit underneath.
RECURSION_LIMIT = 1000

# EVERY PATH HERE IS CHECKED AT PREFLIGHT — see assert_environment_is_real().
#
# This block used to interpolate f"{REPO}/.venv/bin/python" for NGSolve and
# scikit-fem. When the runner was made to self-locate, REPO became this
# checkout, which has no .venv, and every agent was handed an interpreter that
# does not exist. Ten of round 2's first twenty-one runs hit it; one openPASO run
# died 8 calls in with a finished solver script it could not execute. Telling
# an agent a tool is at a path where it is not is the same defect as promising
# it source we do not serve — it reads as an instruction and burns the budget.
_NGSOLVE_SKFEM_PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
_ENV_PATHS = {
    "NGSolve & scikit-fem": _NGSOLVE_SKFEM_PY,
    "FEniCSx/dolfinx": "/home/alexander/miniconda3/envs/fenics/bin/python",
    "DUNE-fem": "/home/alexander/miniconda3/envs/dune-py313/bin/python",
    "Kratos Multiphysics (and gmsh)": _NGSOLVE_SKFEM_PY,
    "4C binary": "/home/alexander/4C/build/4C",
    "FEBio binary": "/home/alexander/FEBio/bin/febio4",
    "deal.II build tree (DEAL_II_DIR)": "/home/alexander/dealii/build",
}

ENVIRON = (
    "\nENVIRONMENT: NGSolve & scikit-fem -> "
    f"{_NGSOLVE_SKFEM_PY} ; "
    "FEniCSx/dolfinx -> /home/alexander/miniconda3/envs/fenics/bin/python ; "
    "DUNE-fem -> /home/alexander/miniconda3/envs/dune-py313/bin/python ; "
    f"Kratos Multiphysics (and gmsh) -> {_NGSOLVE_SKFEM_PY} ; "
    "deal.II -> build C++ with cmake using DEAL_II_DIR=/home/alexander/dealii/build "
    "(run with LD_LIBRARY_PATH=/opt/4C-dependencies/lib) ; "
    "4C binary -> /home/alexander/4C/build/4C ; "
    "FEBio binary -> /home/alexander/FEBio/bin/febio4 .\n"
)


# An interpreter that EXISTS but cannot import its backend is the same defect
# as a path that does not exist — the agent is told a tool is there, tries it,
# and spends its budget working around it. This bit twice: ofa-v2/.venv for
# NGSolve/scikit-fem (did not exist) and /usr/bin/python3 for Kratos (exists,
# is Python 3.8, and this Kratos is built for 3.12). Two coupled runs died
# believing the second one; the runs that scored found Kratos by searching the
# filesystem themselves.
_IMPORT_CHECKS = {
    "NGSolve & scikit-fem": (_NGSOLVE_SKFEM_PY, "import ngsolve, skfem"),
    "Kratos Multiphysics (and gmsh)": (_NGSOLVE_SKFEM_PY,
                                       "import KratosMultiphysics"),
    "FEniCSx/dolfinx": ("/home/alexander/miniconda3/envs/fenics/bin/python",
                        "import dolfinx"),
    # `import dune.fem` IS NOT A CHECK THAT DUNE WORKS. It succeeded in
    # dune-fem-env while the very first real use — structuredGrid, which is
    # JIT-compiled — died on a stale cache:
    #   undefined symbol: PyThreadState_GetUnchecked
    # The preflight passed, the round started, and every DUNE cell would have
    # failed for an environment reason recorded as a model failure. Exercise
    # the JIT path, which is where DUNE actually breaks.
    "DUNE-fem": ("/home/alexander/miniconda3/envs/dune-py313/bin/python",
                 "from dune.grid import structuredGrid; "
                 "structuredGrid([0,0],[1,1],[2,2])"),
}


def assert_backends_importable() -> None:
    """Every interpreter we name must actually import what we claim it has."""
    bad = []
    for what, (interp, stmt) in _IMPORT_CHECKS.items():
        r = subprocess.run([interp, "-c", stmt], capture_output=True,
                           text=True, timeout=300)
        if r.returncode != 0:
            bad.append(f"{what}: {interp} cannot `{stmt}` "
                       f"({(r.stderr or '').strip().splitlines()[-1][:90]})")
    if bad:
        sys.exit("REFUSING TO RUN — an interpreter we advertise cannot import "
                 "its backend:\n  - " + "\n  - ".join(bad))


def assert_environment_is_real() -> None:
    """Refuse to run if anything ENVIRON names does not exist.

    An environment string is a promise to the agent. A wrong path in it does
    not fail loudly — the agent reads it, tries it, and spends its budget
    working around a tool it was told it had.
    """
    missing = [f"{what} -> {p}" for what, p in _ENV_PATHS.items()
               if not Path(p).exists()]
    if missing:
        sys.exit("REFUSING TO RUN — the ENVIRONMENT block promises paths that "
                 "do not exist:\n  - " + "\n  - ".join(missing))

_USAGE_CB = UsageMetadataCallbackHandler()


class TruncationWatch(BaseCallbackHandler):
    """Records replies the output cap cut short.

    A truncated reply is not a model failure and must never be graded as one:
    it is the harness taking the pen out of the agent's hand mid-word. Counted
    per run and surfaced in the ledger so it can be re-run rather than scored.
    """

    def __init__(self):
        self.hits = 0

    def on_llm_end(self, response, **kw):
        try:
            for gen in (response.generations or []):
                for g in gen:
                    info = getattr(g, "generation_info", None) or {}
                    if info.get("finish_reason") == "length":
                        self.hits += 1
        except Exception:
            pass


_TRUNC_CB = TruncationWatch()


def _usage_totals():
    ti = sum(int(u.get("input_tokens", 0) or 0) for u in _USAGE_CB.usage_metadata.values())
    to = sum(int(u.get("output_tokens", 0) or 0) for u in _USAGE_CB.usage_metadata.values())
    return ti, to


# Every request reserves its max output from the SAME 262144-token window the
# history lives in. Unset, the provider reserved its own default of 65536 —
# a quarter of the window, gone whether or not the model wrote a single token,
# which is what killed FC2 BARE seed3 at 196609 input tokens. The arms are not
# equally exposed: the openPASO arm accumulates faster because one knowledge call
# can return ~23k tokens, so an unnecessarily large reservation costs openPASO
# more turns than it costs bare. Measured over 128 round-3 runs, output is 839
# tokens per call on average and 2480 in the worst whole-run average, so 16384
# is roughly six times the worst case observed and frees 49152 tokens of input.
# A cap can truncate mid-sentence, so TruncationWatch below makes that loud
# rather than letting a half-written tool call be graded as the agent's answer.
_MAX_OUT = 16384


def _or_llm(size, *, temperature, seed):
    return ChatOpenAI(base_url="https://openrouter.ai/api/v1",
                      api_key=os.environ["OPENROUTER_API_KEY"],
                      model=OR_MODELS[size], temperature=temperature, seed=seed,
                      max_tokens=_MAX_OUT,
                      timeout=600, max_retries=30,
                      callbacks=[_USAGE_CB, _TRUNC_CB])


_agent._llm = _or_llm

# MATCHED CASE-INSENSITIVELY. The provider returns lowercase
# "error code: 504" while this list carried "Error code: 5", so a gateway
# failure went unrecognised and was booked as a normal run: FC2 BARE died
# after 10 tool calls to a 504 and was graded FAILED, charging OUR outage to
# the model — and to the BARE arm, which inflates the measured uplift. Same
# defect class as the evidence patterns that were lowercased on one side only.
_INFRA_ERRS = ("APIConnectionError", "Connection error", "UnicodeDecodeError",
               "InternalServerError", "empty response",
               "error code: 5", "error code: 429", "status code: 5",
               "RateLimit", "rate limit", "ReadTimeout", "ServiceUnavailable",
               "BadGateway", "Timeout error", "overloaded",
               # our own output cap, not the model's doing — see TruncationWatch
               "OutputTruncated",
               # The MCP transport dying is OUR infrastructure, not a model
               # result. Seen 2026-08-21: a discovery probe let 4C print its
               # ASCII banner into the server's stdout — the JSON-RPC channel —
               # and the stream corruption surfaced as an ExceptionGroup from
               # the client's TaskGroup. Unlisted, that MCP-arm run was booked
               # as a model failure.
               "ExceptionGroup", "JSONRPCMessage", "json_invalid",
               # 402 Insufficient credits. The account ran dry partway through
               # round 5 seed 7 and 18 runs died with ZERO tool calls and zero
               # wall time — they never reached the model. Unlisted, they were
               # booked as model results, 10 of them against the openPASO arm, so
               # an unpaid invoice would have read as a capability gap.
               "error code: 402", "Insufficient credits")


# The provider refuses a request whose input exceeds its window. That is the
# agent having accumulated too much history, not an outage — a different thing
# from both a timeout and an infrastructure fault, and worth counting per arm
# because the arms are not equally exposed to it.
_CONTEXT_ERRS = ("input length", "context length", "maximum context",
                 "context_length_exceeded", "too many tokens",
                 "reduce the length")


def _is_context_exhausted(err: str) -> bool:
    if not err:
        return False
    low = err.lower()
    return any(s in low for s in _CONTEXT_ERRS)


def _is_infra(err: str) -> bool:
    """True when the run died of OUR infrastructure, not the model's work."""
    if not err:
        return False
    low = err.lower()
    return any(s.lower() in low for s in _INFRA_ERRS)


class TrajLiveLog(BaseCallbackHandler):
    """Append tool activity to disk as it happens, so a crash or timeout
    cannot destroy the trajectory the leak audit depends on."""

    def __init__(self, path):
        self._f = open(path, "a", buffering=1, encoding="utf-8", errors="replace")

    # A SILENT 600-CHARACTER CUT MAKES THIS LOG LIE ABOUT WHAT WAS SERVED.
    #
    # The cut is right — a full trajectory with every payload would be tens of
    # megabytes per run — but it was invisible, and the log is the only record
    # of what the agent was told. A prepare_simulation payload of 30,099
    # characters was written as ~600, so grepping the trajectory for a phrase
    # openPASO definitely served returns nothing and reads as "the agent never
    # saw it".
    #
    # That is not hypothetical: on 2026-08-30 I reported that a served
    # primitive "was never consulted" by FB2_27b_MCP_seed15 because its
    # distinctive strings were absent from this file. The run had in fact
    # called prepare_simulation(febio, viscoelasticity) and been served the
    # whole thing. The conclusion was drawn from the truncation, not from the
    # run.
    #
    # So the marker now states what was dropped. A reader who greps and finds
    # nothing can see whether they were reading a complete record or a stub,
    # and `served_bytes` gives the size to check against the tool's own output.
    _CUT = 600

    def _clip(self, text: str) -> str:
        if len(text) <= self._CUT:
            return text
        return (f"{text[:self._CUT]}"
                f"\n  [... TRUNCATED BY THE LOGGER: {len(text)} chars served, "
                f"{self._CUT} recorded. Absence of a phrase below this point "
                f"is NOT evidence it was not served.]")

    def on_tool_start(self, serialized, input_str, **kw):
        try:
            self._f.write(f"TOOL_CALL {(serialized or {}).get('name')} | "
                          f"{self._clip(str(input_str))}\n")
        except Exception:
            pass

    def on_tool_end(self, output, **kw):
        try:
            text = str(output)
            self._f.write(f"TOOL_RESULT served_bytes={len(text)} "
                          f"{self._clip(text)}\n")
        except Exception:
            pass

    def note(self, msg):
        """Record a harness action in the same stream as the tool activity,
        so a reader of the trajectory can tell the agent's moves from ours."""
        try:
            self._f.write(f"{msg}\n")
        except Exception:
            pass

    def close(self) -> None:
        """Flush and close this cell's trajectory stream; safe to call twice."""
        try:
            self._f.close()
        except Exception:
            pass


sys.path.insert(0, str(REPO / "src"))
from blind_eval import keyvault as _kv                             # noqa: E402
from host_hygiene import (readable_worked_answers,               # noqa: E402
                          _materially_useful, _worked_handshake,
                          is_quarantinable_scratch)


def keys_are_sealed() -> bool:
    """The runner must never execute while the answer keys are readable.

    The previous implementation returned True when ``keys/`` was MISSING or
    EMPTY, so a deleted keys tree read as sealed and the campaign would have
    started with nothing to grade against.  Absence is not a seal.  The real
    check lives in the openPASO repo, under version control and under test, and is
    imported rather than restated here -- Amendment 1 declared this fixed while
    the runner still carried the broken copy and never imported the fix.
    """
    return _kv.is_sealed(_keys_dir())


def _keys_dir():
    """The ONE answer-key location, shared with grade_blind_v2.

    The custody preflight checked HERE/"keys", which does not exist in this
    checkout — the keys deliberately live outside the repository. Result: the
    preflight failed on exists=False in every configuration, including the
    correct one. Same defect class as the grader loading its evidence gate
    from another worktree: two components resolving the same thing two ways.
    OPENPASO_BLIND_KEYS is the single authority, exactly as in the grader.
    """
    import os as _os
    v = _os.environ.get("OPENPASO_BLIND_KEYS")
    return Path(v) if v else HERE / "keys"


_WROTE_RE = re.compile(r"wrote \d+ chars to (/[^\s\"']+)")

# AN AGENT WRITES WITH BASH, NOT ONLY WITH write_file.
#
# _WROTE_RE matches the write_file tool's confirmation line and nothing else, so
# a directory an agent created with `mkdir`/`python -c` was invisible to the
# quarantine. Measured: C2_27b_MCP_seed72's agent built /tmp/coupled_heat_final
# -- 12 two-sided interface levels and 18 native solver artefacts, a complete
# worked coupled answer -- its transcript names the path TEN times, and none of
# them in the "wrote N chars to" form. The next seed's preflight then refused to
# start, correctly, because the readable-answers gate found it; but the loop
# stalled instead of healing itself.
#
# This regex therefore collects EVERY absolute path an agent's transcript
# mentions under a scatter root. That is deliberately loose, because precision
# does not come from the pattern here -- it comes from the materiality test
# applied afterwards: a candidate is only quarantined if it actually holds a
# worked coupled answer. The docstring below already warned that "bash is not"
# confined; this closes it.
_PATH_RE = re.compile(r"(/(?:tmp|home/alexander)/[A-Za-z0-9_./+-]{2,})")


def _quarantine_stray_scratch() -> list:
    """Move aside exactly what PRIOR RUNS wrote outside their sandbox.

    Driven by the recorded transcripts, not by a filesystem sweep. The first
    version of this searched /tmp and $HOME for campaign-shaped artefacts and
    a dry run showed it would have moved 1043 directories — including this
    session's own scratch and unrelated pytest trees. Precision matters more
    than reach here: the only directories that can contaminate a run are the
    ones an earlier run actually created, and every one of those is named in
    that run's trajectory.

    Never deletes; moves into runs_quarantine/stray_scratch/ and returns what
    it moved.
    """
    dest_root = HERE / "runs_quarantine" / "stray_scratch"
    repo, here = REPO.resolve(), HERE.resolve()
    protected = {repo, here}
    configured_repo = os.environ.get("OPENPASO_REPO")
    if configured_repo:
        protected.add(Path(configured_repo).expanduser().resolve())
    roots = {Path("/tmp").resolve(), Path.home().resolve(),
             (Path.home() / "Schreibtisch").resolve()}
    targets, moved = set(), []
    bash_candidates: set = set()
    # Quarantined runs count too: their run directory was moved aside, but
    # whatever they wrote into /tmp or $HOME is still sitting there.
    trajectories = list((HERE / "runs").glob("*/work/trajectory*.txt")) + \
        list((HERE / "runs_quarantine").glob("**/work/trajectory*.txt"))
    for traj in trajectories:
        try:
            text = traj.read_text(errors="replace")
        except OSError:
            continue
        seen_paths = {m.group(1) for m in _WROTE_RE.finditer(text)}
        # Bash-created scratch: collect the TOP-LEVEL directory name as a
        # STRING here and probe the filesystem once per unique candidate below.
        #
        # Probing inside this loop walked the disk once per regex match per
        # transcript -- thousands of tree walks over ~2,000 transcripts, minutes
        # per call, in a function that runs at every seed's preflight. String
        # work first, filesystem work once.
        for m in _PATH_RE.finditer(text):
            parts = m.group(1).rstrip("/.,;:'\")").split("/")
            if len(parts) > 2 and parts[1] == "tmp":
                bash_candidates.add("/tmp/" + parts[2])
            elif len(parts) > 4 and parts[1:3] == ["home", "alexander"]:
                bash_candidates.add("/home/alexander/" + parts[3])
        for raw in sorted(seen_paths):
            p = Path(raw)
            try:
                rp = p.resolve()
            except OSError:
                continue
            if repo in rp.parents or rp == repo:
                continue                      # inside the campaign: fine
            if "claude-" in str(rp):
                continue                      # this session's own scratch
            # The directory to move is the highest ancestor still under a
            # scatter root — /tmp/coupled_heat, not /tmp/coupled_heat/level1.
            top = None
            for anc in list(rp.parents):
                if anc in roots:
                    break
                top = anc
            # THE PATH MAY ALREADY *BE* THE DIRECTORY TO MOVE.
            #
            # This loop assumed the matched path is a FILE somewhere below the
            # scatter directory, and walked up to the highest ancestor that is
            # not itself a root. When the match is already top-level -- e.g.
            # /tmp/coupled_heat_final -- the first parent IS /tmp, so it breaks
            # immediately, `top` stays None, and the tree is silently skipped.
            # Measured: that is why the widened path collection still quarantined
            # 0 trees while the readable-answers gate was flagging exactly that
            # directory.
            if top is None and rp.parent in roots:
                top = rp
            if top is not None and top.exists() and top.resolve() not in roots:
                targets.add(top)
    # ONE probe per unique candidate, gated by MATERIALITY rather than by the
    # regex: a candidate is quarantined only if it actually holds a worked
    # coupled answer. That is what keeps a loose path pattern precise.
    for raw in sorted(bash_candidates):
        d = Path(raw)
        if not d.is_dir() or d in roots or "claude-" in raw:
            continue
        try:
            names = [f.name for f in d.iterdir() if f.is_file()]
        except OSError:
            continue
        if _materially_useful(d, names) or _worked_handshake(d):
            targets.add(d)

    for d in sorted(targets):
        if not d.exists() or not is_quarantinable_scratch(d, protected):
            continue
        dest = dest_root / d.name
        n = 1
        while dest.exists():
            n += 1
            dest = dest_root / f"{d.name}__{n}"
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(d), str(dest))
            moved.append(f"{d} -> {dest.relative_to(HERE)}")
        except OSError as e:
            print(f"[preflight] could not quarantine {d}: {e}")
    return moved


def _problems_root() -> Path:
    """The question sheets, from the same place the grader takes them.

    Mirrors campaign3_blind/grading/loading.problems_dir(): OPENPASO_BLIND_PROBLEMS
    if set, else campaign3_blind/problems. Two independent notions of "where the
    task text lives" is how a runner ends up serving one contract while the
    grader enforces another.
    """
    env = os.environ.get("OPENPASO_BLIND_PROBLEMS")
    return Path(env) if env else (HERE / "problems")


def preflight_or_die(problems: list) -> None:
    """Every custody control, executed, before a single paid run starts.

    One cell run with the keys readable makes every cell's blindness arguable
    afterwards, and there is no way to prove afterwards that they were sealed.
    So this is a hard gate, it runs the checks rather than asserting them, and
    it refuses on anything it cannot verify.
    """
    import subprocess
    keys = _keys_dir()
    assert_environment_is_real()
    assert_backends_importable()

    failures = []

    if shutil.which("bwrap") is None:
        failures.append(
            "bubblewrap is unavailable, so shell and MCP solver processes "
            "cannot be isolated from sibling runs")

    # A SEALED PRIMARY SAYS NOTHING ABOUT A COPY BESIDE IT.
    #
    # Measured: keys/ was d--------- while keys_backup_20260816/ next to it was
    # drwxr-xr-x and held 32 plaintext key.json files -- every live problem ID,
    # 28 of them with `exact_solution` -- created 2026-08-14, before every run
    # in this campaign. The seal check reported SEALED the whole time, and it
    # was telling the truth about the one directory it was asked about. The
    # shielding script only ever knew about "$D/keys".
    siblings = _kv.unsealed_sibling_key_stores(keys)
    if siblings:
        failures.append(
            "a COPY of the answer keys is readable beside the sealed one, so "
            "the campaign's blindness cannot be shown:\n  - "
            + "\n  - ".join(f"{d} ({state})" for d, state in siblings))

    if not keys_are_sealed():
        failures.append(f"answer keys at {keys} are not sealed "
                        f"(exists={keys.exists()})")
    else:
        probe = _kv.verify_unreadable(keys)
        if not probe.get("sealed"):
            failures.append(f"the seal did not survive an executed read: {probe}")

    plain = [p for p in keys.rglob("*.json")
             if not p.name.endswith(".enc")] if keys.is_dir() else []
    if plain:
        failures.append(f"{len(plain)} plaintext key file(s) still on disk")

    for src in (HERE / "build_problems.py", HERE / "build_extra.py",
                HERE / "build_coupled.py"):
        if src.exists() and os.access(src, os.R_OK):
            failures.append(f"{src.name} is readable and holds hidden fields "
                            f"as literals two directories above the agent's "
                            f"workdir")

    man = REPO / "data" / "blind_key_commitment.json"
    if not man.is_file():
        failures.append(f"no key commitment at {man}: without it, nobody can "
                        f"check the solutions were fixed before the runs")

    # The exposure sweep, executed, over the whole agent-reachable tree.
    sweep = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "blind_keys.py"), "exposure",
         "--root", str(HERE)],
        capture_output=True, text=True, timeout=600)
    if sweep.returncode != 0:
        failures.append("exposure sweep found readable solution-bearing files:\n"
                        + sweep.stdout[-1500:])

    # PRIOR RUNS' LEAVINGS ARE READABLE BY THIS RUN.
    #
    # write_file is confined to the sandbox now, but bash is not, and round 1
    # scattered participants, RESULT.txt files and solution CSVs across /tmp
    # and $HOME. Two runs then READ them: C2-MCP picked up a previous
    # attempt's RESULT.txt and CSVs in /tmp/coupled_heat, and C4-MCP read
    # C6's deal.II source out of $HOME. That is one cell's work leaking into
    # another cell's measurement.
    #
    # Quarantined by EVIDENCE, never by name — a directory is moved only if
    # it actually contains campaign-shaped artefacts — and moved, never
    # deleted, so nothing is lost.
    # A WORKED ANSWER READABLE ON THE HOST IS A CUSTODY FAILURE TOO.
    worked = readable_worked_answers()
    if worked:
        failures.append(
            "worked coupled answers are readable outside the campaign; an "
            "agent greps this machine and WILL find them (measured: one bare "
            "coupled run did). Move or seal them, then re-run:\n  - "
            + "\n  - ".join(
                f"{d} ({lv} two-sided levels, {nat} native solver artefact(s)"
                + (f", scripts {scr}" if scr else "") + ")"
                for d, (lv, nat, scr) in worked[:10]))

    stray = _quarantine_stray_scratch()
    if stray:
        print(f"[preflight] quarantined {len(stray)} directory(ies) of prior "
              f"agent scratch so this run cannot read them:")
        for s in stray[:12]:
            print(f"             {s}")

    # A coupled task whose intended path has never executed measures the path.
    ready = HERE / "path_readiness.json"
    if ready.is_file():
        rd = json.loads(ready.read_text()).get("path_verified", {})
        # AN ID THE FILE HAS NEVER HEARD OF IS UNREADY, NOT READY.
        #
        # This was `rd.get(p) is False`. For a problem the file does not
        # mention, rd.get returns None, and `None is False` is False — so the
        # id was treated as VERIFIED. The gate only ever caught an id
        # explicitly recorded as unwalked.
        #
        # That makes the control a no-op for exactly the case it exists for: a
        # NEW problem, whose path has by definition never run. Renaming the
        # coupled set — which a rebalanced matrix does — would have silently
        # disabled it for every cell while the preflight printed a pass. The
        # file's own opening principle is "a coupled task whose intended
        # execution path has never run measures the path, not the agent".
        #
        # Absent is now unready. Add the id with `true` once its path has
        # actually been walked, which is the only thing that should silence it.
        unready = [p for p in problems if rd.get(p) is not True]
        if unready and not os.environ.get("OPENPASO_SKIP_PATH_CHECK"):
            failures.append(
                f"no throwaway non-blind run has been recorded through the "
                f"intended coupling path for {unready}. A tool bug there reads "
                f"as agent failure and is charged to the arm under test. Run "
                f"one per task, discard the result, and record it in "
                f"path_readiness.json (or set OPENPASO_SKIP_PATH_CHECK to "
                f"override deliberately).")

    if failures:
        sys.exit("REFUSING TO RUN — custody preflight failed:\n  - "
                 + "\n  - ".join(failures))
    print(f"[preflight] custody controls verified for {len(problems)} "
          f"problem(s); keys sealed and unreadable by execution")


def per_cell_exposure_check(run_dir: Path) -> None:
    """After every cell, sweep the tree the tooling just wrote.

    Amendment 1 found seven plaintext copies of live solutions in scratch space
    AFTER the design declared custody solved -- sub-agent transcripts that had
    quoted key files, a keys snapshot, its __pycache__.  Custody has to cover
    the space tooling writes, not only the directory the design names, and a
    sweep that runs once at the start cannot see what a run creates.
    """
    import subprocess
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "blind_keys.py"), "exposure",
         "--root", str(run_dir)],
        capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        (run_dir / "EXPOSURE_ALERT.txt").write_text(r.stdout)
        print(f"  !! EXPOSURE in {run_dir.name}: a solution is readable from "
              f"the run tree; see EXPOSURE_ALERT.txt", flush=True)


def run_one(pid: str, model: str, cond: str, seed: int, timeout_s: int) -> dict:
    run_dir = HERE / "runs" / f"{pid}_{model}_{cond}_seed{seed}"
    ledger = run_dir / "ledger.json"
    if ledger.exists():
        print(f"[{pid} {model} {cond} s{seed}] SKIP: ledger exists", flush=True)
        return json.loads(ledger.read_text())

    try:
        cell_lock = acquire_cell_lock(run_dir)
    except SourceBuildError as exc:
        sys.exit(f"REFUSING DUPLICATE CELL: {exc}")

    expected_problems = os.environ.get("OPENPASO_PROBLEMS_SHA256")
    try:
        verify_problem_draw(_problems_root(), expected_problems)
    except SourceBuildError as exc:
        sys.exit(f"REFUSING TO RUN — {exc}")

    work = run_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    # THE RUNNER MUST READ THE SAME PROBLEMS ROOT THE GRADER READS.
    #
    # This was hardcoded to HERE/"problems" while the grader's loading.py
    # resolves OPENPASO_BLIND_PROBLEMS. A fresh draw goes into its own root
    # (build_balanced --problems-root, so a spent instance is never
    # overwritten), so a hardcoded runner serves the OLD task text and the
    # grader grades against the NEW key -- agent and grader disagreeing about
    # what was asked, which is the exact class of defect this harness has hit
    # repeatedly.
    _task_path = _problems_root() / pid / "task.txt"
    task = _task_path.read_text(encoding="utf-8")
    # WHICH CONTRACT WAS THIS RUN ACTUALLY ASKED TO MEET?
    #
    # The ledger recorded condition, model, seed, tokens and wall time -- and
    # nothing about the QUESTION. So no run could be tied afterwards to the task
    # text it was served, and a grader holding a newly drawn key had no way to
    # show the agent had been given the matching contract. That is not a
    # hypothetical: the problems root was hardcoded here while the grader
    # resolved OPENPASO_BLIND_PROBLEMS, so runner and grader could silently read
    # different contracts, and a fresh draw would have been graded against text
    # the agent never saw.
    #
    # The sha256 is of the exact bytes served. It costs nothing and it is the
    # only thing that can settle the question later.
    import hashlib as _hl
    _task_meta = {"problems_root": str(_problems_root()),
                  "task_path": str(_task_path),
                  "task_sha256": _hl.sha256(task.encode()).hexdigest(),
                  "source_git_commit": os.environ.get(
                      "OPENPASO_SOURCE_GIT_COMMIT"),
                  "source_sha256": os.environ.get("OPENPASO_SOURCE_SHA256"),
                  "problems_sha256": expected_problems,
                  "dune_cache_sha256": os.environ.get(
                      "OPENPASO_DUNE_CACHE_SHA256")}
    # STATE THE BUDGET. Round 1: 13 of 14 coupled openPASO runs stopped
    # VOLUNTARILY at a mean of 37% of the wall budget (floor 11.8%, 24 calls,
    # zero solver runs), and 47 statements across those transcripts invoke
    # "time constraints" or invent hour estimates — a deadline the model
    # could not observe, because nothing in the prompt or any tool result
    # mentioned one. Cells that quit had converged couplings and diagnosed
    # bugs in hand. An agent that cannot see its budget guesses, and guesses
    # low. Both arms get the identical sentence.
    budget = (
        f"\nBUDGET: you have {timeout_s // 60} minutes of wall-clock for this "
        f"task. That is the only limit that will stop you — the tool-call "
        f"ceiling ({RECURSION_LIMIT // 2}) sits above what the clock can "
        f"reach. No other deadline exists. Nothing is gained by stopping "
        f"early: if you are making progress, keep working. Write "
        f"COULD_NOT_COMPLETE only when you have genuinely exhausted what you "
        f"can do, not when the task looks long.\n")
    prompt = task + ENVIRON + budget

    _USAGE_CB.usage_metadata.clear()
    _TRUNC_CB.hits = 0
    ag = (build_bare_agent(size=model, seed=seed, workdir=work)
          if cond == "BARE" else None)

    # Hand the agent a clock. Both arms: _bash_tool_for is shared, and knowing
    # the time is not an openPASO capability. Two C9 runs threw away 59% of their
    # budget while stating they had run out of it.
    _agent._DEADLINE = (time.time() + timeout_s, float(timeout_s))

    live = TrajLiveLog(work / "trajectory_live.txt")
    t0, err, final, n_calls = time.time(), None, None, 0
    conts = 0

    # A TURN THAT ENDS WITHOUT A TOOL CALL IS NOT A RUN THAT FINISHED.
    #
    # LangGraph's ReAct loop ends the moment a reply carries no tool call. That
    # is correct when the agent is done and wrong when the pen was taken out of
    # its hand, and the two are indistinguishable from the loop's side. Measured
    # over all 813 runs of this campaign, 112 (13.8%) ended with error=null and
    # no RESULT.txt on disk — and the arms are NOT equally exposed: 79 of 405
    # openPASO runs (19.5%) against 33 of 408 bare (8.1%), because the openPASO arm
    # accumulates history faster (one knowledge call can return ~23k tokens) and
    # carries a larger tool schema in every request. Every uplift this campaign
    # has reported so far was measured with the openPASO arm silently losing one
    # run in five to this.
    #
    # The stopped runs were not agents giving up. Their last words were
    # "Now let me run the coupling again:", "First, let me create the FEBio
    # participant for subdomain A:" — mid-task, with 30-70% of the wall clock
    # unspent, one of them having just gotten its second participant to run.
    # Two shapes appear: an empty final message (the provider returned nothing)
    # and a text-only one that announces the next action and never emits it.
    # TruncationWatch sees neither — finish_reason is not "length" — so this
    # went unrecorded through seven rounds.
    #
    # So the wall clock becomes the only thing that ends a run, exactly as the
    # prompt already promises the agent it is. The nudge is deliberately empty
    # of everything but the contract: it names the missing file, states the time
    # left, and repeats that COULD_NOT_COMPLETE is the honest way out. It says
    # nothing about the physics, the method, or the state of the work, and both
    # arms get the identical text — this is the harness enforcing its own
    # deliverable, not a hint. It will help openPASO more than bare, because the
    # defect hurt openPASO more than bare.
    _CONT_MAX = 8
    _NUDGE = (
        "Your last turn ended without a tool call, and {f} does not exist yet, "
        "so this task is not finished. You have {m} minutes of wall-clock left "
        "and nothing else will stop you. Continue from exactly where you "
        "stopped. If you have genuinely exhausted what you can do, write {f} "
        "containing COULD_NOT_COMPLETE and one line saying why.")

    async def _drive(agent):
        nonlocal conts
        answer = work / "RESULT.txt"
        deadline = time.time() + timeout_s
        state = {"messages": [("user", prompt)]}
        while True:
            left = deadline - time.time()
            if left <= 5:
                raise asyncio.TimeoutError
            out = await asyncio.wait_for(
                agent.ainvoke(
                    state, config={"recursion_limit": RECURSION_LIMIT,
                                   "callbacks": [live]}),
                timeout=left)
            msgs = (out or {}).get("messages", []) if isinstance(out, dict) else []
            last = msgs[-1] if msgs else None
            # Deliverable on disk, or the step cap hit (which the block below
            # turns into an error) — either way the run is over.
            if answer.exists() or conts >= _CONT_MAX or getattr(last, "tool_calls", None):
                return out
            body = getattr(last, "content", "") or ""
            if isinstance(body, str) and "need more steps" in body:
                return out
            conts += 1
            live.note(f"[harness] turn ended with no tool call and no "
                      f"RESULT.txt; continuation {conts}/{_CONT_MAX}")
            state = {"messages": list(msgs) + [
                ("user", _NUDGE.format(f="RESULT.txt",
                                       m=max(1, int((deadline - time.time()) // 60))))]}

    async def _invoke():
        if ag is not None:
            return await _drive(ag)
        async with build_mcp_agent(
                size=model, seed=seed, workdir=work) as mcp_agent:
            return await _drive(mcp_agent)

    try:
        final = asyncio.run(_invoke())
    except asyncio.TimeoutError:
        err = f"TimeoutError: exceeded {timeout_s}s"
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        live.close()
        _agent.cleanup_sandbox_scratch(work)

    # A RUN THAT HIT THE STEP CAP IS NOT A RUN THAT FINISHED. LangGraph does
    # not raise here — it returns a normal state whose last message is
    # "Sorry, need more steps to process this request." Round 1 recorded three
    # such runs with error=null, indistinguishable in the ledger from an agent
    # that chose to stop, which is exactly the confusion the budget analysis
    # then had to unpick by hand.
    if err is None and isinstance(final, dict):
        _msgs = final.get("messages", [])
        _last = getattr(_msgs[-1], "content", "") if _msgs else ""
        if isinstance(_last, str) and "need more steps" in _last:
            err = f"RecursionLimit: exhausted {RECURSION_LIMIT} graph steps"

    try:
        msgs = (final or {}).get("messages", []) if isinstance(final, dict) else []
        lines = []
        for m in msgs:
            for tc in (getattr(m, "tool_calls", None) or []):
                lines.append(f"TOOL_CALL {tc.get('name')}")
                n_calls += 1
            c = getattr(m, "content", "") or ""
            if c:
                lines.append(str(c)[:1500])
        (work / "trajectory.txt").write_text("\n".join(lines), errors="replace")
    except Exception:
        pass
    # A TIMED-OUT RUN STILL HAS A TRANSCRIPT. `final` is None on timeout, so
    # the block above wrote nothing and trajectory.txt was 0 bytes for exactly
    # the runs whose evidence matters most — every timeout of round 1. The
    # live log was on disk the whole time; promote it.
    _traj = work / "trajectory.txt"
    _live = work / "trajectory_live.txt"
    if (not _traj.exists() or _traj.stat().st_size == 0) and _live.exists():
        try:
            _traj.write_text(
                "[reconstructed from trajectory_live.txt: the run did not "
                "exit cleanly, so no final message list existed]\n"
                + _live.read_text(errors="replace"), errors="replace")
        except Exception:
            pass
    if n_calls == 0 and (work / "trajectory_live.txt").exists():
        n_calls = sum(1 for ln in (work / "trajectory_live.txt")
                      .read_text(errors="replace").splitlines()
                      if ln.startswith("TOOL_CALL "))

    tin, tout = _usage_totals()
    rec = dict(problem=pid, model=model, condition=cond, seed=seed,
               wall_s=round(time.time() - t0, 1), tool_calls=n_calls,
               tokens_in=tin, tokens_out=tout, error=err,
               truncated_replies=_TRUNC_CB.hits, continuations=conts,
               graded=False, note="grading is offline: run grade_blind.py")
    if _TRUNC_CB.hits and not err:
        # The cap cut a reply short. Whatever the agent produced after that is
        # an artefact of our configuration, so the run is flagged for re-run
        # instead of being graded as the agent's own work. Only when the run
        # carried no other error: a timeout that also truncated is still a
        # timeout, and overwriting it here would lose the real cause.
        rec["outcome"] = "INVALID_INFRA"
        rec["error"] = (f"OutputTruncated: {_TRUNC_CB.hits} reply(ies) hit the "
                        f"{_MAX_OUT}-token output cap")
    if _is_infra(err):
        rec["outcome"] = "INVALID_INFRA"
    elif _is_context_exhausted(err):
        # NOT infrastructure, and not a timeout: the agent filled the context
        # window and the provider refused the request. That is the agent's own
        # accumulation — this harness never trims history — so the run counts
        # as a model result, but it is labelled distinctly because it is a
        # different failure from running out of clock, and because the arms
        # are not equally exposed: the openPASO arm receives much larger tool
        # responses (a single knowledge call can return ~23k tokens). Measured
        # so far: 1 occurrence in 109 runs, in the BARE arm, so no bias yet —
        # but it is worth counting per arm at every tier.
        rec["outcome"] = "CONTEXT_EXHAUSTED"
    try:
        verify_source_snapshot(Path(os.environ["OPENPASO_SOURCE_SNAPSHOT"]))
    except (KeyError, SourceBuildError) as exc:
        rec["outcome"] = "INVALID_INFRA"
        rec["error"] = f"SourceSnapshotIntegrity: {exc}"
    try:
        verify_problem_draw(_problems_root(), expected_problems)
    except SourceBuildError as exc:
        rec["outcome"] = "INVALID_INFRA"
        rec["error"] = f"ProblemDrawIntegrity: {exc}"
    rec.update(_task_meta)
    ledger.write_text(json.dumps(rec, indent=2))
    print(f"[{pid} {model} {cond} s{seed}] done  calls={n_calls} "
          f"tok={tin}/{tout} {rec['wall_s']}s"
          + (f"  ERR {err[:60]}" if err else ""), flush=True)
    # Custody is checked per cell, not once at launch: a run creates scratch
    # that did not exist when the campaign started.
    per_cell_exposure_check(run_dir)
    if not keys_are_sealed():
        sys.exit(f"HALTING AFTER {run_dir.name}: the answer keys became "
                 f"readable during the run. Every later cell would be "
                 f"unarguably non-blind.")
    cell_lock.close()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(OR_MODELS))
    ap.add_argument("--conditions", nargs="+", required=True,
                    choices=["BARE", "MCP"])
    ap.add_argument("--problems", nargs="+", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--phase", required=True,
                    choices=["development", "evaluation"],
                    help="development: refine knowledge, run post-mortems, "
                         "check convergence. evaluation: the paper's numbers — "
                         "requires a freeze marker and held-out instances.")
    ap.add_argument("--timeout", type=int, default=2700,
                    help="per-run wall-clock cap in seconds")
    a = ap.parse_args()

    try:
        source_build = prepare_source_snapshot(
            REPO, HERE / ".source_snapshots")
    except SourceBuildError as exc:
        sys.exit(f"REFUSING TO RUN — source build is not immutable:\n{exc}")
    os.environ["OPENPASO_SOURCE_SNAPSHOT"] = source_build["snapshot_path"]
    os.environ["OPENPASO_SOURCE_GIT_COMMIT"] = source_build["git_commit"]
    os.environ["OPENPASO_SOURCE_SHA256"] = source_build["source_sha256"]
    sys.path.insert(0, str(Path(source_build["snapshot_path"]) / "src"))
    # flush=True: this line identifies WHICH BUILD a live round is on, and
    # block-buffered stdout held it back until process exit -- a launch at
    # 10:38 on 4bd716d7 showed 997f53e1 (the previous round's line) as the
    # newest entry in the cell log for its whole lifetime, which reads as a
    # stale snapshot when it is only a stale buffer.
    print(f"[source build] {source_build['git_commit'][:12]} "
          f"sha256={source_build['source_sha256'][:16]}... (immutable)",
          flush=True)

    preflight_or_die(a.problems)

    # Which PHASE is this? Development runs against spent instances are fine and
    # expected; an EVALUATION run must not grade a problem the knowledge was
    # tuned on. Nothing in this runner used to distinguish them, so the whole
    # protection rested on somebody remembering — and a rule that depends on
    # memory is not a rule. `--phase evaluation` demands a freeze marker and
    # refuses any instance that matches a development one by id or by the
    # fingerprint of its task text, so a rename cannot launder it.
    if a.phase == "evaluation":
        import phase as _phase
        try:
            _phase.assert_evaluation_is_clean(_problems_root())
        except _phase.EvaluationNotCleanError as exc:
            sys.exit(f"REFUSING TO RUN AN EVALUATION: {exc}")
        overlap = sorted(set(a.problems) & set(_phase.DEVELOPMENT))
        if overlap:
            sys.exit("REFUSING TO RUN AN EVALUATION on development instances "
                     f"{overlap}: the knowledge was shaped against these, so "
                     f"grading them measures the tuning and not the tool.")
    else:
        print("[phase: development] results are for post-mortems and "
              "convergence checking, NOT for the paper's evaluation table.")

    try:
        dune_build = prepare_dune_cache_baseline(
            Path(_IMPORT_CHECKS["DUNE-fem"][0]),
            HERE / ".source_snapshots" / "dune-cache")
    except SourceBuildError as exc:
        sys.exit(f"REFUSING TO RUN — DUNE cache baseline failed:\n{exc}")
    os.environ["OPENPASO_DUNE_CACHE_BASELINE"] = dune_build["baseline_path"]
    os.environ["OPENPASO_DUNE_CACHE_SHA256"] = dune_build["dune_cache_sha256"]
    print(f"[DUNE cache] sha256="
          f"{dune_build['dune_cache_sha256'][:16]}... "
          f"(neutral, private copy per cell)")

    population_build = dict(
        source_build,
        dune_cache_sha256=dune_build["dune_cache_sha256"],
        dune_runtime_fingerprint=dune_build["runtime_fingerprint"],
        problems_root=str(_problems_root().resolve()),
        problems_sha256=_tree_sha256(_problems_root().resolve()),
    )
    os.environ["OPENPASO_PROBLEMS_SHA256"] = \
        population_build["problems_sha256"]
    population_lock = (HERE / "runs" /
                       f".source_build_{a.phase}_{a.model}_seed{a.seed}.json")
    try:
        lock_population_source(population_lock, population_build)
    except SourceBuildError as exc:
        sys.exit(f"REFUSING TO RUN — mixed source population:\n{exc}")

    for pid in a.problems:
        for cond in a.conditions:
            run_one(pid, a.model, cond, a.seed, a.timeout)


if __name__ == "__main__":
    main()
