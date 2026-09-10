"""Micro-test (the real step): can the 27B, as a focused worker, write the 4C Neumann-side PARTICIPANT
(deck generated per level, 4C run, flux recovered, exports.json written) from exactly what OASiS serves?
Usage: python micro_4c_participant.py <n> [tag]. Validated by RUNNING the script (4C binary on this machine)."""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "part4c"
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
served = (HERE / "served_fourc_facts.txt").read_text() + "\n\n" + (HERE / "served_fourc_contract.txt").read_text()
if "THE 4C DECK GRAMMAR" not in served:
    served += "\n\n" + (HERE / "served_fourc_grammar.txt").read_text()

TASK = """You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant script for 4C and make it run standalone.
SUBDOMAIN B: the rectangle (0.6, 1.4) x (0, 1), scalar diffusion with k = 5:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 3*x**2*y + 2*y**2 - 4*x*y**3   (as written; convert to 4C's function syntax yourself)
OUTER BOUNDARY (x = 1.4, y = 0, y = 1): u = 0 (Dirichlet)
INTERFACE (x = 0.6): this side is the NEUMANN side; it reads the partner's outward normal flux from ./imports.json (partner name "A"), applies it as its own inward flux on the interface, and exports its own interface trace and its own consistent outward flux to ./exports.json.
MESH: level 1 = 8 x 10 QUAD4 elements, level k doubles both; read the level from ./config.json (key "level").
The 4C binary is /home/alexander/4C/build/4C and needs LD_LIBRARY_PATH=/opt/4C-dependencies/lib; run it line-buffered (stdbuf -oL -eL) and keep its console output in a log file next to the deck.
Write the COMPLETE, RUNNABLE script participant_B.py (run as `python participant_B.py` in its own directory with config.json and imports.json present; numpy and meshio are available). Keep the served handshake, sign convention, flux recovery and export self-check as given; generate the deck in a loop, run 4C, then recover and export. Output ONLY the Python inside one ```python fenced block."""

def run_participant(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / "participant_B.py").write_text(code)
        (d / "config.json").write_text(json.dumps({"level": 1}))
        ys = [i / 10 for i in range(11)]
        (d / "imports.json").write_text(json.dumps({"A": {"field_name": "u", "n_points": len(ys),
            "coordinates": [[0.6, y] for y in ys], "values": [0.0 for _ in ys], "normal_fluxes": [1.5 for _ in ys]}}))
        env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg")
        try:
            r = subprocess.run([PY, "participant_B.py"], cwd=td, capture_output=True, text=True, timeout=600, env=env)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 600s"
        ex = d / "exports.json"
        logs = list(d.glob("*.log")) + list(d.glob("**/*.log"))
        fourc_ran = any("finished normally" in (l.read_text(errors="ignore")) for l in logs)
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-500:]
            return False, f"rc={r.returncode} exports={ex.is_file()} 4C_finished={fourc_ran} :: {tail}"
        try:
            e = json.loads(ex.read_text()); q = [float(v) for v in e.get("normal_fluxes") or []]
            ok = len(q) > 0 and all(abs(v) < 1e6 for v in q) and fourc_ran
            return ok, f"exports n={len(q)} maxflux={max(abs(v) for v in q):.3e} 4C_finished={fourc_ran}"
        except Exception as exc:
            return False, f"bad exports.json: {exc}"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=4000 + i, max_tokens=16000,
        messages=[{"role": "system", "content": "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for 4C.\n\n" + served},
                  {"role": "user", "content": TASK}])
    msg = resp.choices[0].message; text = msg.content or ""
    fin = resp.choices[0].finish_reason
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"
    (HERE / f"participant_{TAG}_{i}.py").write_text(code)
    kept = "EXPORT SELF-CHECK" in code
    ok, info = run_participant(code) if code.strip() else (False, f"EMPTY (finish_reason={fin})")
    results.append({"i": i, "ok": ok, "info": info, "kept_selfcheck": kept, "finish": fin, "secs": round(time.time() - t0, 1), "lines": code.count("\n")})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | kept self-check={kept} | finish={fin} | {info[:280]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
