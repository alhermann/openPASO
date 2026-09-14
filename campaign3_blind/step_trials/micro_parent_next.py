"""Dedicated wording test: after level 1 of a coupled run is complete on disk, does the 27B PARENT's next tool
call become the one-call mesh sequence (couple_levels) when audit_results hands it the ladder brief?
Two brief variants (A = the round-25 wording, B = the current one), N samples each, single completion.
Usage: python micro_parent_next.py <n> [tag]. Neutral task text (not a campaign task)."""
import os, re, sys, json, time
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-v2/src")
from core.instructions import INSTRUCTIONS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TAG = sys.argv[2] if len(sys.argv) > 2 else "pnext"
MODEL = "qwen/qwen3.5-27b"
HERE = Path(__file__).resolve().parent
TASK = """Two codes share the rectangle (0, 2) x (0, 1), split at x = 1: side A (DUNE-fem) on the left, side B (4C) on the right.
Steady heat conduction -div(k grad T) + c T = f with the source and boundary data given below; Dirichlet-Neumann partitioning.
Deliver THREE mesh levels (h = 1/8, 1/16, 1/32): per level a field file per side (temperature at the prescribed probe
points), an interface file per side, a coupling history file named history_level<k>.csv, and a run log per side with
the solver console and an NDOF line. Then a summary file. Budget: 45 minutes of wall clock.
State so far: both participant scripts run standalone; level 1 is coupled (history_level1.csv, 9 iterations to 1e-7);
both level-1 field files, interface files and run logs are written. 31 minutes remain."""
A_OLD = """LADDER STEP 6 OF 6 -- LEVELS [1] ARE COMPLETE ON DISK. If your task prescribes more levels, couple level 2; otherwise write the summary.
HAND THIS STEP TO A SUB-AGENT AS IS: spawn_subagent(role='worker', task="If the task prescribes a level 2: ONE couple_levels(participants=..., levels=[...], history_pattern=<your task's per-level history file name with 1>) call runs every remaining level (it hands each side its level's nx, ny in the environment, doubling every cell count to halve h, warm-starts each level from the previous one, and writes each residual_level<k>.csv and participant_output_level<k>.log); otherwise, per level: in BOTH ./config.json set level=2 AND halve h -- double every cell count (nx, ny) -- because the served participants mesh from nx and ny and take level as a label (measured: three runs changed only the label, coupled three identical meshes and were graded on an unchanged NDOF); then call couple(participants=..., history_path=<absolute path of this level's residual-history file>) and iterate until it converges; then do steps 4 and 5 for it. If not: write the summary file naming ONLY files that exist, then run audit_results(work_dir). CHECK: the audit reports no missing level, no invented name and no missing field file."). The step ends when its check passes on disk; the sub-agent reports the exact error otherwise. Judge nothing about the whole task -- only this step."""
B_NEW = """LADDER STEP 6 OF 6 -- LEVELS [1] ARE COMPLETE ON DISK. If your task prescribes more levels, run them ALL in ONE call: couple_levels(participants=<the same list you passed to couple>, levels=[2, ...], history_pattern='<the task's per-level history file name with {k} in place of the level>'). Otherwise write the summary.
HAND THIS STEP TO A SUB-AGENT AS IS: spawn_subagent(role='worker', task="If the task prescribes a level 2: call couple_levels(participants=<the same list you passed to couple>, levels=[2, ...every further level the task prescribes], history_pattern='<the task's per-level history file name with {k} in place of the level number>') ONCE. It runs every remaining level in that single call: each side gets its level's nx, ny in the environment (every cell count doubled to halve h), each level warm-starts from the previous one, and each level's history file and participant_output_level<k>.log are written. Measured: per-level couple() calls cost ten calls a level and six proven couplings never reached level 3 that way. Then do steps 4 and 5 for every new level. If the task prescribes no further level: write the summary file naming ONLY files that exist, then run audit_results(work_dir). CHECK: the audit reports no missing level, no invented name and no missing field file."). The step ends when its check passes on disk; the sub-agent reports the exact error otherwise. Judge nothing about the whole task -- only this step."""
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = INSTRUCTIONS + "\n\nHost-side tools (also available): run_bash, read_file, write_file, web_search, spawn_subagent."
def ask(brief, seed):
    r = client.chat.completions.create(model=MODEL, temperature=0.7, seed=seed, max_tokens=600,
        extra_body={"reasoning": {"max_tokens": 1500}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK + "\n\naudit_results(work_dir) just returned:\n\n" + brief +
                   "\n\nState your NEXT tool call exactly as you would issue it (one call, with its arguments), and nothing else."}])
    return (r.choices[0].message.content or "").strip()
out = {}
for name, brief in (("A_old", A_OLD), ("B_new", B_NEW)):
    hits = 0; answers = []
    for i in range(N):
        a = ask(brief, 9000 + i)
        first = re.search(r"(couple_levels|couple|spawn_subagent|audit_results|write_file|run_bash)\s*\(", a)
        call = first.group(1) if first else "?"
        # a spawn_subagent whose task text names couple_levels counts as the one-call sequence
        uses_levels = "couple_levels" in a
        hits += int(uses_levels); answers.append({"first_call": call, "couple_levels_named": uses_levels, "text": a[:400]})
        print(f"[{TAG}] {name} sample {i}: first call = {call}; couple_levels named = {uses_levels}", flush=True)
    out[name] = {"couple_levels_named": hits, "n": N, "answers": answers}
    print(f"[{TAG}] {name}: couple_levels named in {hits}/{N}", flush=True)
(HERE / f"results_{TAG}.json").write_text(json.dumps(out, indent=1))
