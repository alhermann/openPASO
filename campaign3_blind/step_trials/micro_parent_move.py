"""Micro-test of the PARENT's first moves on a coupled task: the real harness agent (same system prompt, same
tools, same OASiS server) is run for at most K tool calls and we record what it did -- did it hand step 1 to a
worker sub-agent, did it copy the served contract, or did it start writing the participant itself?
Usage: micro_parent_move.py <n_samples> <tag> [K=12] [problem=C3]   (env as the campaign launch script)"""
import os, sys, json, time, asyncio, glob
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get("OASIS_REPO", "/home/alexander/Schreibtisch/ofa-v2"))
sys.path.insert(0, str(REPO / "langgraph_eval")); sys.path.insert(0, str(REPO / "campaign3_blind"))
import agent as A                                    # noqa: E402
try:
    from run_blind import ENVIRON, RECURSION_LIMIT   # noqa: E402
except Exception as exc:                             # noqa: BLE001
    sys.exit(f"cannot import run_blind for ENVIRON: {exc}")

# THE SAME DUNE CACHE BASELINE THE CAMPAIGN CELLS GET. Without it the sandbox's empty
# DUNE_PY_DIR configures dune-py from scratch, cmake finds ~/.local's dune-common (bound)
# whose script dir (~/.local/share/dune) is hidden, and every JIT compile dies -- measured
# on the first orchestrator-rule sample: the worker copied the contract and lost to cmake.
from run_blind import prepare_dune_cache_baseline, _IMPORT_CHECKS   # noqa: E402
_dune_build = prepare_dune_cache_baseline(Path(_IMPORT_CHECKS["DUNE-fem"][0]),
                                          REPO / "campaign3_blind" / ".source_snapshots" / "dune-cache")
os.environ["OASIS_DUNE_CACHE_BASELINE"] = _dune_build["baseline_path"]
os.environ["OASIS_DUNE_CACHE_SHA256"] = _dune_build["dune_cache_sha256"]
N = int(sys.argv[1]); TAG = sys.argv[2]; K = int(sys.argv[3]) if len(sys.argv) > 3 else 12
PID = sys.argv[4] if len(sys.argv) > 4 else "C3"
task = (Path(os.environ["OASIS_BLIND_PROBLEMS"]) / PID / "task.txt").read_text(encoding="utf-8")
timeout_s = 2700
budget = (f"\nBUDGET: you have {timeout_s // 60} minutes of wall-clock for this task. That is the only limit that will "
          f"stop you — the tool-call ceiling ({RECURSION_LIMIT // 2}) sits above what the clock can reach. No other "
          f"deadline exists. Nothing is gained by stopping early: if you are making progress, keep working. Write "
          f"COULD_NOT_COMPLETE only when you have genuinely exhausted what you can do, not when the task looks long.\n")
prompt = task + ENVIRON + budget

async def one(seed: int):
    work = HERE / "parent_runs" / f"{TAG}_{PID}_s{seed}" / "work"
    work.mkdir(parents=True, exist_ok=True)
    A._ACTIONS_USED = 0
    t0 = time.time()
    async with A.build_mcp_agent(size="27b", seed=seed, workdir=work) as ag:
        try:
            out = await asyncio.wait_for(ag.ainvoke({"messages": [("user", prompt)]},
                                                    config={"recursion_limit": 2 * K + 2}), timeout=1500)
        except asyncio.TimeoutError:
            out = None
    calls = []
    for m in (out or {}).get("messages", []) if isinstance(out, dict) else []:
        for tc in (getattr(m, "tool_calls", None) or []):
            a = tc.get("args") or {}
            calls.append({"name": tc.get("name"), "args": {k: (str(v)[:160]) for k, v in a.items()}})
    roles = [c["args"].get("role") for c in calls if c["name"] == "spawn_subagent"]
    first_worker = next((i for i, c in enumerate(calls) if c["name"] == "spawn_subagent" and c["args"].get("role") == "worker"), None)
    first_own_write = next((i for i, c in enumerate(calls) if c["name"] == "write_file" and c["args"].get("path", "").endswith(".py")), None)
    contract_files = [p for p in glob.glob(str(work / "**/*.py"), recursive=True) if "EXPORT SELF-CHECK" in Path(p).read_text(errors="replace")]
    rec = {"seed": seed, "calls": [c["name"] for c in calls], "n_calls": len(calls), "spawn_roles": roles,
           "first_worker_at": first_worker, "first_own_py_write_at": first_own_write,
           "contract_copied": bool(contract_files), "secs": round(time.time() - t0, 1),
           "detail": calls}
    A.cleanup_sandbox_scratch(work)
    return rec

results = []
for i in range(N):
    r = asyncio.run(one(7000 + i))
    results.append(r)
    print(f"[{TAG}] seed {r['seed']}: calls={r['n_calls']} roles={r['spawn_roles']} worker_at={r['first_worker_at']} "
          f"own_py_at={r['first_own_py_write_at']} contract={r['contract_copied']} {r['secs']}s\n   {r['calls']}", flush=True)
    (HERE / f"results_parent_{TAG}.json").write_text(json.dumps(results, indent=1))
w = sum(1 for r in results if r["first_worker_at"] is not None); c = sum(1 for r in results if r["contract_copied"])
print(f"[{TAG}] worker spawned within {K} calls: {w}/{N} ; contract copied: {c}/{N}")
