"""Dedicated wording test for the reordered converged lead (dc500a8e): after level 1 converges and couple() returns
this lead, does the 27B PARENT's next tool call become couple_levels for the remaining levels (rather than
deliverable scripts or another couple())?  N samples, one completion each. Neutral task text, not a campaign task."""
import os, re, sys, json, time
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-v2/src")
from core.instructions import INSTRUCTIONS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TAG = sys.argv[2] if len(sys.argv) > 2 else "plevels"
LEAD = Path(sys.argv[3]).read_text() if len(sys.argv) > 3 else ""
MODEL = "qwen/qwen3.5-27b"
HERE = Path(__file__).resolve().parent
TASK = """Two codes share the rectangle (0, 2) x (0, 1), split at x = 1: side A (DUNE-fem) on the left, side B (4C) on the right.
Steady heat conduction -div(k grad T) + c T = f with the source and boundary data given below; Dirichlet-Neumann partitioning.
Deliver THREE mesh levels (h = 1/8, 1/16, 1/32): per level a field file per side (temperature at the prescribed probe
points), an interface file per side, a coupling history file named history_level<k>.csv, and a run log per side with
the solver console and an NDOF line. Then a summary file. Budget: 45 minutes of wall clock.
State so far: both participant scripts (the served contracts, filled) run standalone; you just called couple() for level 1
with history_path=history_level1.csv and it converged in 9 iterations to 1e-7. 33 minutes remain."""
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = INSTRUCTIONS + "\n\nHost-side tools (also available): run_bash, read_file, write_file, web_search, spawn_subagent(role, task, context)."
out = []
for i in range(N):
    t0 = time.time()
    r = client.chat.completions.create(model=MODEL, temperature=0.7, seed=9300 + i, max_tokens=900,
        extra_body={"reasoning": {"max_tokens": 1500}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK + "\n\ncouple() just returned, its what_to_fix_next field reads:\n\n" + LEAD +
                   "\n\nState your NEXT tool call exactly as you would issue it (one call, with its arguments), and nothing else."}])
    a = (r.choices[0].message.content or "").strip()
    first = re.search(r"(couple_levels|couple|spawn_subagent|audit_results|write_file|run_bash|read_file)\s*\(", a)
    call = first.group(1) if first else "?"
    levels_first = call == "couple_levels" or (call == "spawn_subagent" and "couple_levels" in a)
    out.append({"first_call": call, "levels_first": levels_first, "secs": round(time.time() - t0), "text": a[:600]})
    print(f"[{TAG}] sample {i}: first call = {call}; levels first = {levels_first} | {out[-1]['secs']}s", flush=True)
hits = sum(o["levels_first"] for o in out)
print(f"[{TAG}] levels first in {hits}/{N}", flush=True)
(HERE / f"results_{TAG}.json").write_text(json.dumps(out, indent=1))
