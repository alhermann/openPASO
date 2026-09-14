"""Micro-test aimed at where round 49's C10 cells actually stopped: side B, the DUNE-fem 3-D participant.

Measured in that round: side A (Kratos 3-D) exported interface data in all three cells and side B never
ran once. This asks the 27B, as a focused step worker with exactly what OASiS serves, to write that side
and then RUNS it. PASS = its exports match, on the interface interior, a reference produced by filling
the SAME served scaffold (dune3d_ref/, never served) and running it.

Usage: python micro_dune3d_side.py <n> [tag] [repairs]"""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
import numpy as np
from openai import OpenAI

HERE = Path(__file__).resolve().parent
REF = Path("/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/dune3d_ref")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "dune3dB"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MODEL = "qwen/qwen3.5-27b"
DUNE_PY = "/home/alexander/miniconda3/envs/dune-py313/bin/python"
served = (HERE / "served_dune_3d_coupling.txt").read_text()

TASK = """You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant script for DUNE-fem and make it run standalone.
SUBDOMAIN B: the box (0.625, 1.5) x (0, 1) x (0, 1), scalar conduction with k = 4:  -div(k grad u) = f
SOURCE TERM: f(x, y, z) = 8*pi**2*sin(pi*y)*sin(pi*z)   (as written; it does not depend on x)
OUTER BOUNDARY (x = 1.5, y = 0, y = 1, z = 0, z = 1): u = 0 (Dirichlet)
INTERFACE (the plane x = 0.625): this side is the NEUMANN side; it reads the partner's outward normal flux from ./imports.json (partner name "left"), applies it as its own natural condition on that plane, and exports its own interface trace and its own consistent outward normal flux to ./exports.json.
INTERFACE CORNERS: where the interface plane meets the outer boundary, the outer boundary condition above applies on BOTH subdomains. Those points belong to the outer boundary, not to the interface, on either side.
MESH LEVEL 1: 6 x 6 x 6 elements.
The DUNE interpreter is /home/alexander/miniconda3/envs/dune-py313/bin/python.
Write the COMPLETE, RUNNABLE script participant_B.py (run as `python participant_B.py` in its own directory with imports.json present). Copy the served 3-D contract, edit only its placeholder block for this subdomain, keep every served line as given, and write only the marked SOLVE holes yourself. Output ONLY the Python inside one ```python fenced block."""


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
            r = subprocess.run([DUNE_PY, "participant_B.py"], cwd=td, capture_output=True, text=True,
                               timeout=2400, env=dict(os.environ, MPLBACKEND="Agg"))
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT 2400s"
        ex = d / "exports.json"
        if r.returncode != 0 or not ex.is_file():
            tail = (r.stderr or r.stdout or "")[-700:]
            return False, f"rc={r.returncode} exports={ex.is_file()} :: {tail}"
        try:
            e = json.loads(ex.read_text())
            co = np.asarray(e["coordinates"], float)
            v = np.asarray(e["values"], float).ravel()
            q = np.asarray(e["normal_fluxes"], float).ravel()
            rco, rv, rq = _ref()
            if co.shape != rco.shape:
                return False, f"exported {co.shape} interface points, the reference has {rco.shape}"
            # THE RIM IS NOT GRADED, here as in the campaign's tasks: those nodes carry the outer
            # boundary's reaction too and the recovered flux there does not converge.
            inner = ((co[:, 1] > 1e-9) & (co[:, 1] < 1 - 1e-9) & (co[:, 2] > 1e-9) & (co[:, 2] < 1 - 1e-9))
            if inner.sum() < 4:
                return False, "fewer than four interior interface points"
            ev = float(np.max(np.abs(v[inner] - rv[inner])) / max(np.max(np.abs(rv[inner])), 1e-30))
            eq = float(np.max(np.abs(q[inner] - rq[inner])) / max(np.max(np.abs(rq[inner])), 1e-30))
            ok = ev < 0.08 and eq < 0.08
            return ok, f"n={len(v)} interior={int(inner.sum())} trace_rel={ev:.2e} flux_rel={eq:.2e}"
        except Exception as exc:                       # noqa: BLE001
            return False, f"bad exports.json: {exc}"


def _extract(text: str) -> str:
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    code = m.group(1) if m else text
    return "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"


client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=1200)
SYS = ("You are a finite-element simulation assistant. What follows is the documentation the OASiS "
       "server gave you for DUNE-fem.\n\n" + served)
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=2000 + i, max_tokens=24000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK}])
    fin = resp.choices[0].finish_reason
    code = _extract(resp.choices[0].message.content or "")
    (HERE / f"participant_{TAG}_{i}.py").write_text(code)
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
            (HERE / f"participant_{TAG}_{i}_fix{_r + 1}.py").write_text(code)
            ok, info = run_participant(code) if code.strip() else (False, "EMPTY repair reply")
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "attempts": attempts, "secs": round(time.time() - t0, 1)})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | attempts={attempts} | {round(time.time()-t0)} s | {info[:320]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
