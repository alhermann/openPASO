"""Micro-test of a capability OASiS gained today: an interface that BENDS.

One coupled development problem has a two-leg interface -- a square corner cut out of the other
subdomain -- and until today no contract could express it. This asks the 27B, with exactly what OASiS
serves, to write the side whose interface is its own left and top edges, then RUNS it. PASS = its
exports match, on the interior of both legs, a reference produced by filling the SAME served scaffold
(never served) and running it.

Usage: python micro_bent_side.py <n> [tag] [repairs]"""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from openai import OpenAI

HERE = Path(__file__).resolve().parent
REF = Path("/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/fx_bent")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "bentB"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MODEL = "qwen/qwen3.5-27b"
FENICS = "/home/alexander/miniconda3/envs/fenics/bin/python"
served = (HERE / "served_fenics_coupling.txt").read_text()

TASK = """You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant script for FEniCSx (dolfinx) and make it run standalone.
SUBDOMAIN B: the square (0.5, 1) x (0, 0.5), scalar diffusion with k = 5/2:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 0   (there is no volumetric source on this subdomain)
INTERFACE: the BENT polyline made of two legs -- the segment x = 1/2 for 0 < y < 1/2, then the segment y = 1/2 for 1/2 < x < 1. Both legs are interface; this subdomain's own left and top edges are exactly those legs.
OUTER BOUNDARY (y = 0 and x = 1, the bottom and right edges): u = 0 (Dirichlet)
INTERFACE CORNERS: where the interface meets the outer boundary, the outer boundary condition above applies on BOTH subdomains. Those points belong to the outer boundary, not to the interface, on either side.
ROLE: this side is the NEUMANN side; it reads the partner's outward normal flux from ./imports.json (partner name "left"), applies it as its own natural condition on BOTH legs, and exports its own interface trace and its own consistent outward normal flux to ./exports.json.
MESH LEVEL 1: 16 x 16 elements.
The FEniCSx interpreter is /home/alexander/miniconda3/envs/fenics/bin/python.
Write the COMPLETE, RUNNABLE script participant_B.py (run as `python participant_B.py` in its own directory with imports.json present). Copy the served contract, edit only its placeholder block for this subdomain, keep every served line as given, and write only the marked SOLVE holes yourself. Output ONLY the Python inside one ```python fenced block."""


def _ref():
    e = json.loads((REF / "exports.json").read_text())
    return (np.asarray(e["coordinates"], float), np.asarray(e["values"], float).ravel(),
            np.asarray(e["normal_fluxes"], float).ravel())


def run_participant(code: str):
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "participant_B.py").write_text(code)
        (d / "imports.json").write_text((REF / "imports.json").read_text())
        try:
            r = subprocess.run([FENICS, "participant_B.py"], cwd=td, capture_output=True, text=True,
                               timeout=900, env=dict(os.environ, MPLBACKEND="Agg"))
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 900s"
        ex = d / "exports.json"
        if r.returncode != 0 or not ex.is_file():
            return False, f"rc={r.returncode} exports={ex.is_file()} :: " + (r.stderr or r.stdout)[-600:]
        e = json.loads(ex.read_text())
        co = np.atleast_2d(np.asarray(e["coordinates"], float))
        v = np.asarray(e["values"], float).ravel()
        q = np.asarray(e["normal_fluxes"], float).ravel()
        rco, rv, rq = _ref()
        on1 = np.isclose(co[:, 0], 0.5)
        on2 = np.isclose(co[:, 1], 0.5)
        if on1.sum() < 5 or on2.sum() < 5:
            return False, f"exported {on1.sum()} points on leg 1 and {on2.sum()} on leg 2: not both legs"
        if len(co) != len(rco):
            return False, f"exported {len(co)} interface points, the reference has {len(rco)}"
        # order-independent comparison: match each exported point to the reference point at the same place
        idx = [int(np.argmin(np.abs(rco[:, 0] - p[0]) + np.abs(rco[:, 1] - p[1]))) for p in co]
        # THE ENDS ARE NOT GRADED (Dirichlet-Neumann corners), here as in the tasks
        inner = ~(np.isclose(co[:, 1], 0.0) | np.isclose(co[:, 0], 1.0))
        ev = float(np.max(np.abs(v[inner] - rv[idx][inner])) / max(np.max(np.abs(rv)), 1e-30))
        eq = float(np.max(np.abs(q[inner] - rq[idx][inner])) / max(np.max(np.abs(rq)), 1e-30))
        ok = ev < 0.08 and eq < 0.08
        return ok, f"n={len(co)} legs=({int(on1.sum())},{int(on2.sum())}) trace_rel={ev:.2e} flux_rel={eq:.2e}"


def _extract(text: str) -> str:
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"


client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=1200)
SYS = ("You are a finite-element simulation assistant. What follows is the documentation the OASiS "
       "server gave you for FEniCSx.\n\n" + served)
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=2000 + i, max_tokens=24000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK}])
    code = _extract(resp.choices[0].message.content or "")
    (HERE / f"participant_{TAG}_{i}.py").write_text(code)
    ok, info = run_participant(code) if code.strip() else (False, "EMPTY reply")
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
            (HERE / f"participant_{TAG}_{i}_fix{_r + 1}.py").write_text(code)
            ok, info = run_participant(code) if code.strip() else (False, "EMPTY repair")
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "attempts": attempts, "secs": round(time.time() - t0, 1)})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | attempts={attempts} | {round(time.time()-t0)} s | {info[:300]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
