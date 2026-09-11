"""Dedicated gate test: does OASiS's ladder brief for a REFUSED 4C DECK (4C's own error lines + the deck
defects OASiS names from the deck text + the runs that finished) get the 27B worker from a failed
participant to a passing one?  Compared against the plain repair the step trial already does (stderr tail).
Usage: python micro_4c_deck_repair.py <failed_script.py> [tag] [extra_served_file,...] [rounds=2]
The failed script is re-run in a fresh directory (the trial's config/imports), the side directory is read by
OASiS's gate (tools.result_audit._fourc_deck_state), and the worker gets exactly what a real worker gets:
the served text, its own script, and the ladder brief. Graded by the trial's grader (manufactured solution)."""
import os, sys, json, shutil, subprocess, time
from pathlib import Path
sys.argv, _argv = [sys.argv[0]], sys.argv           # the trial module reads argv at import; give it none
import micro_4c_thermoelastic as T                   # TASK, CONFIG, IMPORTS, run_participant, extract, served, MODEL, PY
sys.argv = _argv
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-v2/src")
from tools.result_audit import _fourc_deck_state
from openai import OpenAI
HERE = Path(__file__).resolve().parent
FAILED = Path(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else "repair"
served = T.served
for _extra in (sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] else []):
    served += "\n\n# ---- reply of another OASiS door the agent can call ----\n" + Path(_extra).read_text()

def reproduce(script: Path, d: Path) -> str:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy(script, d / "participant_A.py")
    (d / "config.json").write_text(json.dumps(T.CONFIG)); (d / "imports.json").write_text(json.dumps(T.IMPORTS))
    env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg")
    r = subprocess.run([T.PY, "participant_A.py"], cwd=d, capture_output=True, text=True, timeout=900, env=env)
    return (r.stderr or r.stdout or "")[-1500:]

ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
work = HERE / f"repair_work_{TAG}"
side = work / "side_A"
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for 4C.\n\n" + served
code = FAILED.read_text()
log = []
ok, info = False, ""
for rnd in range(1, ROUNDS + 1):
    t0 = time.time()
    tail = reproduce(Path(HERE / f"participant_{TAG}_round{rnd - 1}.py") if rnd > 1 else FAILED, side)
    state = _fourc_deck_state(side, work)
    if not state:
        print(f"[{TAG}] round {rnd}: the gate found no deck and no 4C console in the side directory; stopping"); break
    brief = (f"LADDER STEP 2 OF 6 -- MAKE THE 4C DECK RUN in side_A/participant_A.py: {state['what']}\n"
             f"HAND THIS STEP TO A SUB-AGENT AS IS: spawn_subagent(role='worker', task=\"In the directory of "
             f"side_A/participant_A.py: {state['brief']} Then run the participant again with its interpreter. CHECK: "
             f"./exports.json appears beside the script with finite values and the script exited 0.\")")
    print(f"[{TAG}] round {rnd} gate brief ({len(brief)} chars): {state['what']}", flush=True)
    resp = client.chat.completions.create(model=T.MODEL, temperature=0.7, seed=7000 + rnd, max_tokens=32000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": T.TASK},
                  {"role": "assistant", "content": "```python\n" + code + "```"},
                  {"role": "user", "content": "That script was run exactly as specified and did NOT pass. Its run ended with:\n\n"
                   + tail + "\n\nYou are the worker for exactly this one step. audit_results(work_dir) on the working "
                   "directory returned:\n\n" + brief + "\n\nHand in the corrected COMPLETE participant_A.py as one "
                   "```python fenced block (keep every served line; change only what the step names)."}])
    code = T.extract(resp.choices[0].message.content or "")
    (HERE / f"participant_{TAG}_round{rnd}.py").write_text(code)
    ok, info = T.run_participant(code)
    log.append({"round": rnd, "ok": ok, "info": info[:600], "what": state["what"], "secs": round(time.time() - t0)})
    print(f"[{TAG}] round {rnd}: {'PASS' if ok else 'FAIL'} | kept self-check={'EXPORT SELF-CHECK' in code} | {round(time.time() - t0)}s | {info[:300]}", flush=True)
    if ok:
        break
print(f"[{TAG}] gate loop: {'PASS' if ok else 'FAIL'} after {len(log)} round(s)", flush=True)
(HERE / f"results_{TAG}.json").write_text(json.dumps({"failed": str(FAILED), "ok": ok, "rounds": log}, indent=1))
