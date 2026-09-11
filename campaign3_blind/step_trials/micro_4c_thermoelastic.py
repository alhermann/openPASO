"""Micro-test (the real step): can the 27B, as a focused worker, write the 4C DIRICHLET-side THERMO-ELASTIC
participant (two decks per level, 4C run twice, flux + traction recovered, exports.json written) from exactly
what OASiS serves for physics='thermoelastic'?  Usage: python micro_4c_thermoelastic.py <n> [tag] [repairs] [extra_served_file,...].
Graded by RUNNING the script against a manufactured thermo-elastic solution (never the campaign's)."""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, sympy as sp
from openai import OpenAI
HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "te4c"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
served = (HERE / "served_fourc_thermoelastic_pointer.txt").read_text()
if "THE 4C DECK GRAMMAR" not in served:
    served += "\n\n" + (HERE / "served_fourc_grammar.txt").read_text()
# argv[4]: extra served replies (comma-separated files), e.g. the prepare_simulation(solver='fourc',
# physics='tsi') reply a real agent can fetch (its 3x3 illustrative template and the corpus TSI tests).
for _extra in (sys.argv[4].split(",") if len(sys.argv) > 4 and sys.argv[4] else []):
    served += "\n\n# ---- reply of another OASiS door the agent can call ----\n" + Path(_extra).read_text()

LX, LY = 0.8, 1.0; KV, LAM, MU, BETA = 2.0, 500.0, 300.0, 1.0
x, y = sp.symbols("x y")
T = x * sp.sin(sp.pi * y); ux = sp.Rational(2, 100) * x * sp.sin(sp.pi * y); uy = sp.Rational(1, 100) * x**2 * sp.sin(sp.pi * y)
exx, eyy, exy = sp.diff(ux, x), sp.diff(uy, y), (sp.diff(ux, y) + sp.diff(uy, x)) / 2
sxx = 2 * MU * exx + LAM * (exx + eyy) - BETA * T; syy = 2 * MU * eyy + LAM * (exx + eyy) - BETA * T; sxy = 2 * MU * exy
fT = sp.simplify(-KV * (sp.diff(T, x, 2) + sp.diff(T, y, 2)))
fx = sp.simplify(-(sp.diff(sxx, x) + sp.diff(sxy, y))); fy = sp.simplify(-(sp.diff(sxy, x) + sp.diff(syy, y)))
f4 = lambda e: str(e).replace("**", "^")
L = lambda e: sp.lambdify((x, y), e, "numpy")
Tf, uxf, uyf = L(T), L(ux), L(uy)
qn_f, qx_f, qy_f = L(-KV * sp.diff(T, x)), L(-sxx), L(-sxy)

TASK = f"""You are the worker for ONE step of a partitioned coupled simulation. Your step: write side A's participant script for 4C and make it run standalone.
SUBDOMAIN A: the rectangle (0, 0.8) x (0, 1). Steady thermoelasticity, plane strain, small strain: BOTH of
    -div(k grad T) = f_T
    -div(sigma_tot(u, T)) = (f_x, f_y),  sigma_tot = 2*mu*sym(grad(u)) + lambda*div(u)*I - beta*T*I
with k = 2, lambda = 500, mu = 300, beta = 1 (so E = {float(MU*(3*LAM+2*MU)/(LAM+MU)):.6g}, nu = {float(LAM/(2*(LAM+MU))):.6g}, alpha = beta/(3*lambda+2*mu) = {float(BETA/(3*LAM+2*MU)):.8g}).
SOURCE TERMS (as written; convert to 4C's function syntax yourself):
    f_T = {sp.sstr(fT)}
    f_x = {sp.sstr(fx)}
    f_y = {sp.sstr(fy)}
OUTER BOUNDARY (x = 0, y = 0, y = 1): T = 0 and u = 0 (both components).
INTERFACE (x = 0.8): this side is the DIRICHLET side. It reads the partner's interface temperature and displacement from ./imports.json (partner name "B"; values = [T, ux, uy] per interface point), imposes them at its interior interface nodes, and exports its own consistent outward heat flux and traction to ./exports.json exactly as the served contract specifies (normal_fluxes = [qn, qx, qy] per point).
MESH LEVEL 1: 8 x 10 elements (h = 0.1). ./config.json follows the served contract's schema: {{"level": 1, "nx": 8, "ny": 10, "x0": 0.0, "x1": 0.8, "y0": 0.0, "y1": 1.0, "k": 2.0, "lam": 500.0, "mu": 300.0, "beta": 1.0, "iface": "right", "side": "dirichlet", "source_T": "<f_T in 4C syntax>", "source_ux": "<f_x in 4C syntax>", "source_uy": "<f_y in 4C syntax>", "fourc_bin": "/home/alexander/4C/build/4C", "fourc_ld": "/opt/4C-dependencies/lib"}}.
The 4C binary is /home/alexander/4C/build/4C and needs LD_LIBRARY_PATH=/opt/4C-dependencies/lib; run it line-buffered (stdbuf -oL -eL) and keep each run's console output in a log file next to its deck.
Write the COMPLETE, RUNNABLE script participant_A.py (run as `python participant_A.py` in its own directory with config.json and imports.json present; numpy and meshio are available). Keep the served handshake, sign convention, finish check, recovery and export self-check as given; fill the hole with the two decks and the two runs, leaving behind the names the contract lists. Output ONLY the Python inside one ```python fenced block."""

CONFIG = {"level": 1, "nx": 8, "ny": 10, "x0": 0.0, "x1": LX, "y0": 0.0, "y1": LY, "k": KV, "lam": LAM, "mu": MU, "beta": BETA,
          "iface": "right", "side": "dirichlet", "source_T": f4(fT), "source_ux": f4(fx), "source_uy": f4(fy),
          "fourc_bin": "/home/alexander/4C/build/4C", "fourc_ld": "/opt/4C-dependencies/lib"}
YS = [i / 20 for i in range(21)]
IMPORTS = {"B": {"field_name": "thermoelastic", "n_points": len(YS), "coordinates": [[LX, yy] for yy in YS],
                 "values": [[float(Tf(LX, yy)), float(uxf(LX, yy)), float(uyf(LX, yy))] for yy in YS],
                 "normal_fluxes": [[0.0, 0.0, 0.0] for _ in YS]}}

def run_participant(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / "participant_A.py").write_text(code)
        (d / "config.json").write_text(json.dumps(CONFIG)); (d / "imports.json").write_text(json.dumps(IMPORTS))
        env = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib", MPLBACKEND="Agg")
        try:
            r = subprocess.run([PY, "participant_A.py"], cwd=td, capture_output=True, text=True, timeout=900, env=env)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 900s"
        ex = d / "exports.json"
        logs = list(d.glob("*.log")) + list(d.glob("**/*.log"))
        n_fin = sum(l.read_text(errors="ignore").count("finished normally") for l in logs)
        ndof = bool(re.search(r"^NDOF = \d+", r.stdout, re.M))
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-600:]
            return False, f"rc={r.returncode} exports={ex.is_file()} 4C_runs_finished={n_fin} :: {tail}"
        try:
            e = json.loads(ex.read_text())
            C = np.asarray(e.get("coordinates") or [], float); Q = np.asarray(e.get("normal_fluxes") or [], float)
            if C.ndim != 2 or Q.ndim != 2 or Q.shape[1] != 3 or len(C) != len(Q) or len(C) == 0:
                return False, f"exports shape coords={C.shape} fluxes={Q.shape}"
            ex_ = np.column_stack([qn_f(C[:, 0], C[:, 1]), qx_f(C[:, 0], C[:, 1]), qy_f(C[:, 0], C[:, 1])])
            errs = [float(np.abs(Q[:, c] - ex_[:, c]).max() / np.abs(ex_[:, c]).max()) for c in range(3)]
            ok = all(er < 0.15 for er in errs) and n_fin >= 2 and ndof   # the validated fill reaches 3.7e-3 / 1.45e-2 / 1.43e-2 at 8x10
            return ok, f"exports n={len(C)} qn_err={errs[0]:.2e} qx_err={errs[1]:.2e} qy_err={errs[2]:.2e} 4C_runs={n_fin} NDOF_line={ndof} values_n={len(e.get('values') or [])}"
        except Exception as exc:
            return False, f"bad exports.json: {exc}"

def extract(text):
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for 4C.\n\n" + served
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=5000 + i, max_tokens=32000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK}])
    text = resp.choices[0].message.content or ""; fin = resp.choices[0].finish_reason
    code = extract(text); (HERE / f"participant_{TAG}_{i}.py").write_text(code)
    kept = "EXPORT SELF-CHECK" in code
    ok, info = run_participant(code) if code.strip() else (False, f"EMPTY (finish_reason={fin})")
    attempts = 1
    if not ok and REPAIRS > 0:
        for _r in range(REPAIRS):
            attempts += 1
            fix = client.chat.completions.create(model=MODEL, temperature=0.7, seed=6000 + i, max_tokens=32000,
                extra_body={"reasoning": {"max_tokens": 4000}},
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK},
                          {"role": "assistant", "content": "```python\n" + code + "```"},
                          {"role": "user", "content": "That script was run exactly as specified and did NOT pass. Its run ended with:\n\n" + info[-1500:] + "\n\nHand in the corrected COMPLETE script, again as one ```python fenced block, keeping every served line as given."}])
            code = extract(fix.choices[0].message.content or ""); (HERE / f"participant_{TAG}_{i}_fix{_r+1}.py").write_text(code)
            kept = "EXPORT SELF-CHECK" in code
            ok, info = run_participant(code)
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "kept_selfcheck": kept, "attempts": attempts, "finish": fin, "secs": round(time.time() - t0, 1), "lines": code.count("\n")})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | kept self-check={kept} | attempts={attempts} | finish={fin} | {info[:300]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}", flush=True)
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
