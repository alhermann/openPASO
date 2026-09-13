"""Dedicated wording test: after level 1 converges and the couple() lead warns that a hand-written side keeps no
level-tagged output, does the 27B PARENT's next call preserve this level's outputs (copy/rename to level-tagged names,
or add the served dump block) BEFORE coupling the next level?  N samples, one completion each. Neutral task text."""
import os, re, sys, json, time
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-v2/src")
from core.instructions import INSTRUCTIONS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TAG = sys.argv[2] if len(sys.argv) > 2 else "ptag"
LEAD = Path(sys.argv[3]).read_text()
MODEL = "qwen/qwen3.5-27b"
HERE = Path(__file__).resolve().parent
TASK = """Two codes share the rectangle (0, 1.5) x (0, 1), split at x = 0.625: side A (4C) on the left, side B (FEniCSx) on the right.
Steady thermo-elasticity, Dirichlet-Neumann partitioning, temperature and displacement exchanged at the interface.
Deliver THREE mesh levels (h = 1/10, 1/20, 1/40): per level a field file per side at the prescribed probe points, an interface
file per side, a coupling history file named history_level<k>.csv, and a run log per side with the solver console and an NDOF line.
Then a summary file. Budget: 45 minutes of wall clock.
State so far: side A's participant is your own script (it writes its 4C decks and runs the binary with the output prefix 'out');
side B's is the served contract. You just called couple() for level 1 with history_path=history_level1.csv and it converged in 14 iterations to 5e-7. 30 minutes remain."""
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = INSTRUCTIONS + "\n\nHost-side tools (also available): run_bash, read_file, write_file, web_search, spawn_subagent(role, task, context)."
out = []
for i in range(N):
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, temperature=0.7, seed=9400 + i, max_tokens=900,
        extra_body={"reasoning": {"max_tokens": 1500}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK + "\n\ncouple() just returned, its what_to_fix_next field reads:\n\n" + LEAD +
                   "\n\nState your NEXT tool call exactly as you would issue it (one call, with its arguments), and nothing else."}])
    a = (r.choices[0].message.content or "").strip()
    first = re.search(r"(couple_levels|couple|spawn_subagent|audit_results|write_file|run_bash|read_file)\s*\(", a)
    call = first.group(1) if first else "?"
    preserves = bool(re.search(r"level1|level_1|_L1|level-1|field_level|interface_level|cp |copy|rename|mv ", a)) and call != "couple_levels" or ("copy" in a.lower() and "level" in a.lower())
    out.append({"first_call": call, "preserves_level1": preserves, "secs": round(time.time() - t0), "text": a[:700]})
    print(f"[{TAG}] sample {i}: first call = {call}; preserves level-1 outputs = {preserves} | {out[-1]['secs']}s", flush=True)
print(f"[{TAG}] preserves in {sum(o['preserves_level1'] for o in out)}/{N}", flush=True)
(HERE / f"results_{TAG}.json").write_text(json.dumps(out, indent=1))
