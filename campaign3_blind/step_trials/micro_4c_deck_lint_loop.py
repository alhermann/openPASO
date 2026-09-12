"""Dedicated gate test for the 4C RUN HOOK (tools.workspace_advisor._fourc_run_check): a 27B worker writes a
Python script that produces a Scalar_Transport deck for a manufactured heat problem and the trial runs 4C on
it, exactly as a parent does through run_bash.  Arm 'hook': each failed round is fed what the hook appends to
the shell reply (every deck defect the lint names + 4C's own stop line).  Arm 'console': each failed round is
fed only what 4C's console said (the parent's situation before the hook).  PASS = 4C finished, the VTU exists,
max|phi_1 - x*sin(pi*y)| / 0.8 < 0.05.   Usage: python micro_4c_deck_lint_loop.py <hook|console> [n_seeds=3] [rounds=4]
Nothing here is a benchmark key: the problem is manufactured for this trial."""
import os, re, sys, json, shutil, subprocess, time
from pathlib import Path
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-v2/src")
from tools.fourc_deck_lint import fourc_error_lines
from tools.workspace_advisor import _fourc_run_check
from openai import OpenAI
import numpy as np
import meshio

ARM = sys.argv[1]; NSEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3; ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
assert ARM in ("hook", "console")
HERE = Path(__file__).resolve().parent
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
FOURC = "/home/alexander/4C/build/4C"; LD = "/opt/4C-dependencies/lib"
CMD = f"stdbuf -oL -eL {FOURC} slab_T.4C.yaml out > run_4C.log 2>&1"
served = (HERE / "served_fourc_facts.txt").read_text() + "\n\n" + (HERE / "served_fourc_grammar.txt").read_text()
YS = [j / 10 for j in range(11)]
TABLE = "\n".join(f"    y = {y:.1f}: T = {0.8 * np.sin(np.pi * y):.10f}" for y in YS)
TASK = f"""You are the worker for ONE step: make the 4C side of a coupled heat problem run standalone.
DOMAIN: the rectangle (0, 0.8) x (0, 1). Steady heat conduction, k = 2:  -div(k grad T) = f_T  with
    f_T = 2*pi**2*x*sin(pi*y)      (as written; convert to 4C's function syntax yourself)
OUTER BOUNDARY (x = 0, y = 0, y = 1): T = 0.
INTERFACE (x = 0.8): impose the partner's temperature at the 11 interface nodes (given per node, y from 0 to 1 in steps of 0.1):
{TABLE}
MESH: structured 8 x 10 four-node quadrilaterals (h = 0.1, 99 nodes), Scalar_Transport (steady: one step to a stationary state is fine).
OUTPUT: runtime VTK output of the field every step, and the consistent boundary heat flux on the interface line.
The 4C binary is {FOURC} and needs LD_LIBRARY_PATH={LD}; it is run as `{CMD}` in the directory of the deck.
Write ONE Python script gen_deck.py (numpy available) that, run in an empty directory, WRITES the complete deck to ./slab_T.4C.yaml
(generate the node coordinates, the element connectivity, the topology sections and the conditions programmatically).
Output ONLY the Python script as one ```python fenced block."""
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for 4C.\n\n" + served


def extract(text):
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"


def grade(d: Path) -> tuple[bool, str]:
    vtus = sorted(d.glob("*vtk-files/*.vtu"))
    if not vtus:
        return False, "no VTU written"
    m = meshio.read(vtus[-1])
    key = next((k for k in m.point_data if "phi" in k.lower() and "flux" not in k.lower()), None)
    if key is None:
        return False, f"no scalar field in the VTU (point_data {list(m.point_data)})"
    T = np.asarray(m.point_data[key], float).ravel(); P = m.points
    ex = P[:, 0] * np.sin(np.pi * P[:, 1]); err = float(np.abs(T - ex).max() / 0.8)
    return err < 0.05, f"{key}: max rel err {err:.3e} over {len(T)} points"


def run_round(code: str, d: Path) -> dict:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "gen_deck.py").write_text(code)
    r = subprocess.run([PY, "gen_deck.py"], cwd=d, capture_output=True, text=True, timeout=300)
    deck = d / "slab_T.4C.yaml"
    if r.returncode != 0 or not deck.is_file():
        return {"stage": "gen", "ok": False, "feedback": "gen_deck.py did not produce slab_T.4C.yaml:\n" + (r.stderr or r.stdout)[-1200:]}
    env = dict(os.environ, LD_LIBRARY_PATH=LD)
    try:
        subprocess.run(CMD, shell=True, cwd=d, timeout=600, env=env)
    except subprocess.TimeoutExpired:
        pass
    console = (d / "run_4C.log").read_text(errors="ignore") if (d / "run_4C.log").is_file() else ""
    finished = "finished normally" in console
    hook = _fourc_run_check(CMD, console, d)               # what the parent's shell reply carries since 71e09623
    n_defects = len(re.findall(r"^\s+- ", hook, re.M))
    ok, why = grade(d) if finished else (False, "4C did not finish")
    if ARM == "hook":
        fb = hook.strip() or ("4C's console ended with:\n" + "\n".join(console.splitlines()[-12:]))
    else:
        stop = fourc_error_lines(console)
        fb = ("4C stopped: " + stop) if stop else ("4C's console ended with:\n" + "\n".join(console.splitlines()[-12:]))
    if finished and not ok:
        fb += f"\n\nThe run finished but the result is not accepted: {why}"
    return {"stage": "run", "ok": ok, "finished": finished, "lint_defects": n_defects, "why": why, "feedback": fb,
            "deck_lines": len(deck.read_text().splitlines())}


results = []
for i in range(NSEEDS):
    seed = 7100 + i
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": TASK}]
    log = []; ok = False
    for rnd in range(1, ROUNDS + 1):
        t0 = time.time()
        resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=seed * 10 + rnd, max_tokens=20000,
                                              extra_body={"reasoning": {"max_tokens": 4000}}, messages=msgs)
        code = extract(resp.choices[0].message.content or "")
        d = HERE / f"lintloop_work_{ARM}_{seed}" / f"round{rnd}"
        res = run_round(code, d)
        res.update({"round": rnd, "secs": round(time.time() - t0), "provider": getattr(resp, "provider", None)})
        log.append({k: (v if k != "feedback" else v[:400]) for k, v in res.items()})
        print(f"[{ARM} {seed}] round {rnd}: {'PASS' if res['ok'] else 'FAIL'} stage={res['stage']} finished={res.get('finished')} "
              f"lint_defects={res.get('lint_defects')} {res.get('why','')} | {res['secs']}s", flush=True)
        ok = res["ok"]
        if ok:
            break
        msgs += [{"role": "assistant", "content": "```python\n" + code + "```"},
                 {"role": "user", "content": "gen_deck.py was run in an empty directory, then `" + CMD + "` was run there. It did NOT pass:\n\n"
                  + res["feedback"] + "\n\nHand in the corrected COMPLETE gen_deck.py as one ```python fenced block."}]
    results.append({"seed": seed, "ok": ok, "rounds": log})
    print(f"[{ARM} {seed}] {'PASS' if ok else 'FAIL'} after {len(log)} round(s)", flush=True)
    (HERE / f"results_lintloop_{ARM}.json").write_text(json.dumps(results, indent=1))
print(f"[{ARM}] {sum(r['ok'] for r in results)}/{len(results)} PASS; rounds to pass: {[len(r['rounds']) if r['ok'] else None for r in results]}")
