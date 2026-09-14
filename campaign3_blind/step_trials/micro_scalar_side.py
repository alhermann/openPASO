"""Micro-test: can the 27B, as a focused step worker, write a working SCALAR-diffusion participant for
one side of a straight-interface coupling from exactly what OASiS serves (the full coupling reply)?
Usage: python micro_scalar_side.py <code> <dirichlet|neumann> <n> [tag] [repairs]
Graded by RUNNING the script with the code's interpreter on a manufactured problem:
  dirichlet: A = (0, 0.6) x (0, 1), k = 1, u = x sin(pi y)         -> exports trace 0.6 sin(pi y), outward flux -sin(pi y)
  neumann:   B = (0.6, 1.4) x (0, 1), k = 5, u = (1.4 - x) sin(pi y) -> exports trace 0.8 sin(pi y), outward flux -5 sin(pi y)
PASS = exported flux and trace within the tolerances a P1 8x10 solve of the same problem reaches (flux 0.25, trace 0.06)."""
import os, re, sys, json, subprocess, tempfile, time, math
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
CODE = sys.argv[1]; ROLE = sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
TAG = sys.argv[4] if len(sys.argv) > 4 else f"{CODE}_{ROLE[0].upper()}"
REPAIRS = int(sys.argv[5]) if len(sys.argv) > 5 else 0
MODEL = "qwen/qwen3.5-27b"
INTERP = {"ngsolve": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
          "skfem": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
          "kratos": "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python",
          "fenics": "/home/alexander/miniconda3/envs/fenics/bin/python"}[CODE]
LABEL = {"ngsolve": "NGSolve", "skfem": "scikit-fem", "kratos": "Kratos Multiphysics", "fenics": "FEniCSx (dolfinx)"}[CODE]
served = (HERE / f"served_{CODE}_coupling.txt").read_text()

if ROLE == "dirichlet":
    TASK = f"""You are the worker for ONE step of a partitioned coupled simulation. Your step: write side A's participant script for {LABEL} and make it run standalone.
SUBDOMAIN A: the rectangle (0, 0.6) x (0, 1), scalar diffusion with k = 1:  -div(k grad u) = f
SOURCE TERM: f(x, y) = pi**2*x*sin(pi*y)   (as written)
OUTER BOUNDARY (x = 0, y = 0, y = 1): u = 0 (Dirichlet)
INTERFACE (x = 0.6): this side is the DIRICHLET side; it reads the partner's interface values from ./imports.json (partner name "right"), imposes them as an essential condition on the interface, and exports its own interface trace and its own consistent outward normal flux to ./exports.json.
MESH LEVEL 1: 6 x 10 elements.
The interpreter is {INTERP}.
Write the COMPLETE, RUNNABLE script participant_A.py (run as `python participant_A.py` in its own directory with imports.json present). Copy the served DIRICHLET-side contract, edit only its placeholder block (geometry, K, F_SRC, outer value, NX, NY, PARTNER) for this subdomain, keep every served line as given, and write only the marked SOLVE holes yourself. Output ONLY the Python inside one ```python fenced block."""
    PARTNER = "right"; X_IF = 0.6
    def imports(ys): return {"field_name": "u", "n_points": len(ys), "coordinates": [[X_IF, y] for y in ys],
                             "values": [0.6 * math.sin(math.pi * y) for y in ys], "normal_fluxes": [0.0 for _ in ys]}
    def q_ref(y): return -math.sin(math.pi * y)
    def u_ref(y): return 0.6 * math.sin(math.pi * y)
    QTOL, UTOL = 0.25, 0.06
else:
    TASK = f"""You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant script for {LABEL} and make it run standalone.
SUBDOMAIN B: the rectangle (0.6, 1.4) x (0, 1), scalar diffusion with k = 5:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 5*pi**2*(1.4 - x)*sin(pi*y)   (as written)
OUTER BOUNDARY (x = 1.4, y = 0, y = 1): u = 0 (Dirichlet)
INTERFACE (x = 0.6): this side is the NEUMANN side; it reads the partner's outward normal flux from ./imports.json (partner name "left"), applies it as its own natural (load) condition on the interface, and exports its own interface trace and its own consistent outward normal flux to ./exports.json.
MESH LEVEL 1: 8 x 10 elements.
The interpreter is {INTERP}.
Write the COMPLETE, RUNNABLE script participant_B.py (run as `python participant_B.py` in its own directory with imports.json present). Copy the served NEUMANN-side contract, edit only its placeholder block (geometry, K, F_SRC, outer value, NX, NY, PARTNER) for this subdomain, keep every served line as given, and write only the marked SOLVE holes yourself. Output ONLY the Python inside one ```python fenced block."""
    PARTNER = "left"; X_IF = 0.6
    def imports(ys): return {"field_name": "u", "n_points": len(ys), "coordinates": [[X_IF, y] for y in ys],
                             "values": [0.0 for _ in ys], "normal_fluxes": [5.0 * math.sin(math.pi * y) for y in ys]}
    def q_ref(y): return -5.0 * math.sin(math.pi * y)
    def u_ref(y): return 0.8 * math.sin(math.pi * y)
    QTOL, UTOL = 0.25, 0.06
FNAME = "participant_A.py" if ROLE == "dirichlet" else "participant_B.py"

def run_participant(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / FNAME).write_text(code)
        ys = [i / 10 for i in range(11)]
        (d / "imports.json").write_text(json.dumps({PARTNER: imports(ys)}))
        try:
            r = subprocess.run([INTERP, FNAME], cwd=td, capture_output=True, text=True, timeout=600,
                               env=dict(os.environ, MPLBACKEND="Agg", OMP_NUM_THREADS="2"))
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 600s"
        ex = d / "exports.json"
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-700:]
            return False, f"rc={r.returncode} exports={ex.is_file()} :: {tail}"
        try:
            e = json.loads(ex.read_text())
            co = e.get("coordinates") or []; q = [float(v) for v in (e.get("normal_fluxes") or [])]; vals = [float(v) for v in (e.get("values") or [])]
            eq = max(abs(qq - q_ref(c[1])) for c, qq in zip(co, q)) if q else float("inf")
            ev = max(abs(vv - u_ref(c[1])) for c, vv in zip(co, vals)) if vals else float("inf")
            xs_ok = all(abs(float(c[0]) - X_IF) < 1e-9 for c in co)
            ok = len(q) > 0 and xs_ok and eq < QTOL and ev < UTOL
            return ok, f"exports n={len(q)} values={len(vals)} x_ok={xs_ok} flux_err={eq:.2e} trace_err={ev:.2e}"
        except Exception as exc:
            return False, f"bad exports.json: {exc}"

def _extract(text: str) -> str:
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=900)
SYS = f"You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for {LABEL}.\n\n" + served
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=2000 + i, max_tokens=24000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK}])
    fin = resp.choices[0].finish_reason; code = _extract(resp.choices[0].message.content or "")
    (HERE / f"participant_{TAG}_{i}.py").write_text(code)
    kept = "SELF-CHECK" in code
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
            kept = "SELF-CHECK" in code
            ok, info = run_participant(code) if code.strip() else (False, "EMPTY repair reply")
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "kept_selfcheck": kept, "attempts": attempts, "finish": fin, "secs": round(time.time() - t0, 1), "lines": code.count("\n")})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | kept self-check={kept} | attempts={attempts} | {round(time.time()-t0)} s | {info[:400]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
