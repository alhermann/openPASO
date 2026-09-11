"""Micro-test (the real step): can the 27B, as a focused worker, write the FEniCSx NEUMANN-side THERMO-ELASTIC
participant from exactly what OASiS serves for physics='thermoelastic'?  Usage: python micro_fenics_thermoelastic.py <n> [tag] [repairs].
Graded by RUNNING the script against a manufactured thermo-elastic solution (never the campaign's)."""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np, sympy as sp
from openai import OpenAI
HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "tefx"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 0
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/miniconda3/envs/fenics/bin/python"
served = (HERE / "served_fenics_thermoelastic_pointer.txt").read_text()

XA, XB, LY = 0.8, 1.6, 1.0; KV, LAM, MU, BETA = 3.0, 400.0, 500.0, 1.0
x, y = sp.symbols("x y")
# T's x-derivative must not vanish at the interface x = 0.8 (x*(1.6-x) has its maximum there and gave a
# zero heat datum, a degenerate trial): the extra (x + 0.4) factor moves the extremum
T = x * (XB - x) * (x + sp.Rational(2, 5)) * sp.sin(sp.pi * y); ux = sp.Rational(2, 100) * x * (XB - x) * sp.sin(sp.pi * y); uy = sp.Rational(1, 100) * x**2 * (XB - x) * sp.sin(sp.pi * y)
exx, eyy, exy = sp.diff(ux, x), sp.diff(uy, y), (sp.diff(ux, y) + sp.diff(uy, x)) / 2
sxx = 2 * MU * exx + LAM * (exx + eyy) - BETA * T; syy = 2 * MU * eyy + LAM * (exx + eyy) - BETA * T; sxy = 2 * MU * exy
fT = sp.simplify(-KV * (sp.diff(T, x, 2) + sp.diff(T, y, 2)))
fx = sp.simplify(-(sp.diff(sxx, x) + sp.diff(sxy, y))); fy = sp.simplify(-(sp.diff(sxy, x) + sp.diff(syy, y)))
L = lambda e: sp.lambdify((x, y), e, "numpy")
Tf, uxf, uyf = L(T), L(ux), L(uy)
# the partner's export = this side's natural datum (n_own = -e_x): [k grad T . n_own, sigma . n_own]
gT, gx, gy = L(-KV * sp.diff(T, x)), L(-sxx), L(-sxy)

TASK = f"""You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant script for FEniCSx (dolfinx) and make it run standalone.
SUBDOMAIN B: the rectangle (0.8, 1.6) x (0, 1). Steady thermoelasticity, plane strain, small strain: BOTH of
    -div(k grad T) = f_T
    -div(sigma_tot(u, T)) = (f_x, f_y),  sigma_tot = 2*mu*sym(grad(u)) + lambda*div(u)*I - beta*T*I
with k = 3, lambda = 400, mu = 500, beta = 1.
SOURCE TERMS (as written):
    f_T = {sp.sstr(fT)}
    f_x = {sp.sstr(fx)}
    f_y = {sp.sstr(fy)}
OUTER BOUNDARY (x = 1.6, y = 0, y = 1): T = 0 and u = 0 (both components).
INTERFACE (x = 0.8): this side is the NEUMANN side. It reads the partner's outward heat flux and traction from ./imports.json (partner name "A"; normal_fluxes = [qn, qx, qy] per interface point, in the served sign convention), applies them unchanged as its natural boundary data on the interface, and exports its own interface temperature and displacement (values = [T, ux, uy]) plus its own consistent flux to ./exports.json exactly as the served contract specifies.
MESH LEVEL 1: 8 x 10 cells (h = 0.1); set every placeholder in the served EDIT block to this problem.
Write the COMPLETE, RUNNABLE script participant_B.py (run as `python participant_B.py` in its own directory with imports.json present; the interpreter is the dolfinx one). Keep the served handshake, sign convention, recovery and export self-check as given; fill only the marked holes. Output ONLY the Python inside one ```python fenced block."""

YS = [i / 20 for i in range(21)]
IMPORTS = {"A": {"field_name": "thermoelastic", "n_points": len(YS), "coordinates": [[XA, yy] for yy in YS],
                 "values": [[float(Tf(XA, yy)), float(uxf(XA, yy)), float(uyf(XA, yy))] for yy in YS],
                 "normal_fluxes": [[float(gT(XA, yy)), float(gx(XA, yy)), float(gy(XA, yy))] for yy in YS]}}

def run_participant(code: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d / "participant_B.py").write_text(code)
        (d / "imports.json").write_text(json.dumps(IMPORTS))
        env = dict(os.environ, MPLBACKEND="Agg")
        try:
            r = subprocess.run([PY, "participant_B.py"], cwd=td, capture_output=True, text=True, timeout=900, env=env)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 900s"
        ex = d / "exports.json"
        ndof = bool(re.search(r"^NDOF = \d+", r.stdout, re.M))
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-600:]
            return False, f"rc={r.returncode} exports={ex.is_file()} :: {tail}"
        try:
            e = json.loads(ex.read_text())
            C = np.asarray(e.get("coordinates") or [], float); V = np.asarray(e.get("values") or [], float); Q = np.asarray(e.get("normal_fluxes") or [], float)
            if C.ndim != 2 or V.ndim != 2 or V.shape[1] != 3 or len(C) != len(V) or len(C) == 0:
                return False, f"exports shape coords={C.shape} values={V.shape} fluxes={Q.shape}"
            inner = (C[:, 1] > 1e-9) & (C[:, 1] < LY - 1e-9)
            exV = np.column_stack([Tf(C[:, 0], C[:, 1]), uxf(C[:, 0], C[:, 1]), uyf(C[:, 0], C[:, 1])])
            ev = [float(np.abs(V[inner, c] - exV[inner, c]).max() / np.abs(exV[:, c]).max()) for c in range(3)]
            exQ = -np.column_stack([gT(C[:, 0], C[:, 1]), gx(C[:, 0], C[:, 1]), gy(C[:, 0], C[:, 1])])
            eq = [float(np.abs(Q[inner, c] - exQ[inner, c]).max() / max(np.abs(exQ[:, c]).max(), 1e-12)) for c in range(3)] if Q.ndim == 2 and Q.shape == V.shape else [9, 9, 9]
            ok = all(er < 0.06 for er in ev) and all(er < 0.1 for er in eq) and ndof   # the validated file reaches 1.8e-2 / 1.2e-2 / 8e-3 and 1.6e-2 at 8x10
            return ok, f"exports n={len(C)} T_err={ev[0]:.2e} ux_err={ev[1]:.2e} uy_err={ev[2]:.2e} load_consistency={max(eq):.2e} NDOF_line={ndof}"
        except Exception as exc:
            return False, f"bad exports.json: {exc}"

def extract(text):
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for FEniCSx.\n\n" + served
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=7000 + i, max_tokens=20000,
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
            fix = client.chat.completions.create(model=MODEL, temperature=0.7, seed=8000 + i, max_tokens=20000,
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
