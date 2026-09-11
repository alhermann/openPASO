"""Micro-test: can the 27B, as a focused step worker, write a working Kratos NEUMANN-side participant
from exactly what OASiS serves?  Usage: python micro_kratos_neumann.py <n> [tag] [repairs]
Graded by RUNNING the script with the Kratos interpreter on a manufactured problem
(u = (1.4-x) sin(pi y) on B = (0.6,1.4)x(0,1), k = 5; exported trace 0.8 sin(pi y), outward flux -5 sin(pi y))."""
import os, re, sys, json, subprocess, tempfile, time, math
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
TAG = sys.argv[2] if len(sys.argv) > 2 else "kratosN"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
served = (HERE / "served_kratos_facts.txt").read_text() + "\n\n" + (HERE / "served_kratos_neumann.txt").read_text()

TASK = """You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant script for Kratos Multiphysics (ConvectionDiffusionApplication) and make it run standalone.
SUBDOMAIN B: the rectangle (0.6, 1.4) x (0, 1), scalar diffusion with k = 5:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 5*pi**2*(1.4 - x)*sin(pi*y)   (as written)
OUTER BOUNDARY (x = 1.4, y = 0, y = 1): u = 0 (Dirichlet)
INTERFACE (x = 0.6): this side is the NEUMANN side; it reads the partner's outward normal flux from ./imports.json (partner name "left"), applies it as its own load on the interface, and exports its own interface trace and its own consistent outward flux to ./exports.json.
MESH LEVEL 1: 8 x 10 elements.
The Kratos interpreter is /home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python.
Write the COMPLETE, RUNNABLE script participant_B.py (run as `python participant_B.py` in its own directory with imports.json present). Copy the served NEUMANN-side contract, edit only its placeholder block (geometry, K, F_SRC, T_OUTER, FULL_OUTER_DIRICHLET, NX, NY) for this subdomain, keep every served line as given, and write only the marked SOLVE holes yourself. Output ONLY the Python inside one ```python fenced block."""

def run_participant(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / "participant_B.py").write_text(code)
        ys = [i / 10 for i in range(11)]
        (d / "imports.json").write_text(json.dumps({"left": {"field_name": "u", "n_points": len(ys),
            "coordinates": [[0.6, y] for y in ys], "values": [0.8 * math.sin(math.pi * y) for y in ys],
            "normal_fluxes": [5.0 * math.sin(math.pi * y) for y in ys]}}))
        try:
            r = subprocess.run([PY, "participant_B.py"], cwd=td, capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, MPLBACKEND="Agg"))
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 600s"
        ex = d / "exports.json"
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-600:]
            return False, f"rc={r.returncode} exports={ex.is_file()} :: {tail}"
        try:
            e = json.loads(ex.read_text())
            co = e.get("coordinates") or []; q = [float(v) for v in (e.get("normal_fluxes") or [])]; vals = [float(v) for v in (e.get("values") or [])]
            eq = max(abs(qq + 5.0 * math.sin(math.pi * c[1])) for c, qq in zip(co, q)) if q else float("inf")
            ev = max(abs(vv - 0.8 * math.sin(math.pi * c[1])) for c, vv in zip(co, vals)) if vals else float("inf")
            ok = len(q) > 0 and eq < 0.25 and ev < 0.06   # the unelided served program reaches 8.2e-2 / 1.8e-2 at 8x10
            return ok, f"exports n={len(q)} values={len(vals)} flux_err={eq:.2e} trace_err={ev:.2e}"
        except Exception as exc:
            return False, f"bad exports.json: {exc}"

def _extract(text: str) -> str:
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for Kratos Multiphysics.\n\n" + served
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=2000 + i, max_tokens=24000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK}])
    fin = resp.choices[0].finish_reason; code = _extract(resp.choices[0].message.content or "")
    (HERE / f"participant_{TAG}_{i}.py").write_text(code)
    kept = "CONSERVATION SELF-CHECK" in code
    ok, info = run_participant(code) if code.strip() else (False, f"EMPTY (finish_reason={fin})")
    attempts = 1
    if not ok and REPAIRS > 0:
        for _r in range(REPAIRS):
            attempts += 1
            fix = client.chat.completions.create(model=MODEL, temperature=0.7, seed=3000 + i, max_tokens=24000,
                extra_body={"reasoning": {"max_tokens": 4000}},
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK},
                          {"role": "assistant", "content": "```python\n" + code + "```"},
                          {"role": "user", "content": "That script was run exactly as specified and did NOT produce a correct exports.json. Its output ended with:\n\n" + info[-1500:] + "\n\nHand in the corrected COMPLETE script, again as one ```python fenced block, keeping every served line as given."}])
            code = _extract(fix.choices[0].message.content or "")
            (HERE / f"participant_{TAG}_{i}_fix{_r+1}.py").write_text(code)
            kept = "CONSERVATION SELF-CHECK" in code
            ok, info = run_participant(code) if code.strip() else (False, "EMPTY repair reply")
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "kept_selfcheck": kept, "attempts": attempts, "finish": fin, "secs": round(time.time() - t0, 1), "lines": code.count("\n")})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | kept self-check={kept} | attempts={attempts} | {info[:300]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
