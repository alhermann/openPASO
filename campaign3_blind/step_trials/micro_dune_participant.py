"""Micro-test: can the 27B, as a focused step worker, write a working DUNE-fem Dirichlet-side participant
from exactly what OASiS serves?  Usage: python micro_dune_participant.py <n> [tag]
Validated by RUNNING the script with the DUNE interpreter on a synthetic imports.json (timeout 900 s: first-run JIT)."""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TAG = sys.argv[2] if len(sys.argv) > 2 else "dune"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL = "qwen/qwen3.5-27b"
DUNE_PY = "/home/alexander/miniconda3/envs/dune-py313/bin/python"
served = (HERE / "served_dune_facts.txt").read_text() + "\n\n" + (HERE / "served_dune_contract.txt").read_text()
if (HERE / "served_dune_template.txt").exists():   # what prepare_simulation(dune, ...) also hands a real run
    served += "\n\n# From prepare_simulation(solver='dune', physics=...): the served minimal working example (a DIFFERENT, generic problem; its API is measured working on this install)\n" + (HERE / "served_dune_template.txt").read_text()

TASK = """You are the worker for ONE step of a partitioned coupled simulation. Your step: write side A's participant script for DUNE-fem and make it run standalone.
SUBDOMAIN A: the rectangle (0, 0.6) x (0, 1), scalar diffusion with k = 1:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 2*x*y**2 - x**2   (as written)
OUTER BOUNDARY (x = 0, y = 0, y = 1): u = 0 (Dirichlet)
INTERFACE (x = 0.6): this side is the DIRICHLET side; it reads the partner's interface values from ./imports.json (partner name "B") and imposes them; it exports its own interface trace and its own consistent outward flux to ./exports.json.
MESH LEVEL 1: 6 x 10 uniform quadrilaterals. ./config.json follows the served contract's schema: {"level": 1, "nx": 6, "ny": 10, "x0": 0.0, "x1": 0.6, "y0": 0.0, "y1": 1.0, "k": 1.0, "reaction": 0.0}; the polynomial source is NOT in the config -- it is yours to write into the form.
Write the COMPLETE, RUNNABLE script participant_A.py (it is run as `python participant_A.py` in its own directory with config.json and imports.json present). Keep every served line as given; write only the two marked holes (the simplex grid and P1 space; the form, source, boundary conditions and solve) yourself. Output ONLY the Python inside one ```python fenced block."""

def run_participant(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / "participant_A.py").write_text(code)
        (d / "config.json").write_text(json.dumps({"level": 1, "nx": 6, "ny": 10, "x0": 0.0, "x1": 0.6, "y0": 0.0, "y1": 1.0, "k": 1.0, "reaction": 0.0}))
        ys = [i / 10 for i in range(11)]
        (d / "imports.json").write_text(json.dumps({"B": {"field_name": "u", "n_points": len(ys),
            "coordinates": [[0.6, y] for y in ys], "values": [0.3 * y * (1 - y) for y in ys],
            "normal_fluxes": [0.0 for _ in ys]}}))
        try:
            r = subprocess.run([DUNE_PY, "participant_A.py"], cwd=td, capture_output=True, text=True, timeout=900,
                               env=dict(os.environ, MPLBACKEND="Agg"))
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 900s"
        ex = d / "exports.json"
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-500:]
            return False, f"rc={r.returncode} no_exports={not ex.is_file()} :: {tail}"
        try:
            e = json.loads(ex.read_text())
            n = len(e.get("normal_fluxes") or []); vals = e.get("values") or []
            # PASS = the exported flux matches the reference computed by a validated fill of the
            # SAME served scaffold on the SAME inputs (dune_reference_exports.json; the fill is never
            # served). A finite flux alone is not a pass: one sample exported garbage that way.
            ref = json.loads((HERE / "dune_reference_exports.json").read_text())["normal_fluxes"]
            q = [float(v) for v in e["normal_fluxes"]]
            err = max(abs(a - b) for a, b in zip(q, ref)) if len(q) == len(ref) else float("inf")
            ok = n > 0 and err < 1e-3 * max(abs(r) for r in ref)
            return ok, f"exports n={n} values={len(vals)} maxflux={max(abs(x) for x in q):.3e} err_vs_ref={err:.2e}"
        except Exception as exc:
            return False, f"bad exports.json: {exc}"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=2000 + i, max_tokens=24000,
        extra_body={"reasoning": {"max_tokens": 4000}},   # bound the model's thinking so the answer is not starved of output tokens
        messages=[{"role": "system", "content": "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for DUNE-fem.\n\n" + served},
                  {"role": "user", "content": TASK}])
    msg = resp.choices[0].message; text = msg.content or ""
    if not text.strip():
        print(f"[{TAG}] sample {i}: EMPTY content (finish_reason={resp.choices[0].finish_reason})", flush=True)
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"
    (HERE / f"participant_{TAG}_{i}.py").write_text(code)
    kept = "EXPORT SELF-CHECK" in code
    ok, info = run_participant(code) if code.strip() else (False, f"EMPTY (finish_reason={resp.choices[0].finish_reason})")
    attempts = 1
    if not ok and REPAIRS > 0:      # one repair round, as the ladder's worker loop gives: the run's own output goes back
        for _r in range(REPAIRS):
            attempts += 1
            fix = client.chat.completions.create(model=MODEL, temperature=0.7, seed=3000 + i, max_tokens=24000,
                extra_body={"reasoning": {"max_tokens": 4000}},
                messages=[{"role": "system", "content": "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for DUNE-fem.\n\n" + served},
                          {"role": "user", "content": TASK},
                          {"role": "assistant", "content": "```python\n" + code + "```"},
                          {"role": "user", "content": "That script was run exactly as specified and did NOT produce a correct exports.json. Its output ended with:\n\n" + info[-1500:] + "\n\nHand in the corrected COMPLETE script, again as one ```python fenced block, keeping every served line as given."}])
            t2 = fix.choices[0].message.content or ""
            m2 = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", t2, re.S)
            code = m2.group(1) if m2 else t2
            code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"
            (HERE / f"participant_{TAG}_{i}_fix{_r+1}.py").write_text(code)
            kept = "EXPORT SELF-CHECK" in code
            ok, info = run_participant(code) if code.strip() else (False, "EMPTY repair reply")
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "kept_selfcheck": kept, "attempts": attempts, "secs": round(time.time() - t0, 1), "lines": code.count("\n")})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | kept self-check={kept} | {info[:260]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
