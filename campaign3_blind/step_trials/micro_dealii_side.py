"""Micro-test aimed at the other side round 49 never got running: the deal.II participant, which is a
C++ program the agent must write and build plus the thin Python wrapper that drives it.

C4's side B never ran in any cell, and no cell had even touched cmake by its halfway point. This asks
the 27B, as a focused step worker with exactly what OASiS serves, for the three files and then BUILDS
and RUNS them. PASS = the exports match a manufactured solution the shipped reference reproduces to
4e-3 (u) and 5.8e-3 (q) on the same mesh.

Usage: python micro_dealii_side.py <n> [tag] [repairs]"""
import os, re, sys, json, math, subprocess, tempfile, time
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "dealiiB"
REPAIRS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"
DEAL_II_DIR = "/home/alexander/dealii/build"
served = (HERE / "served_dealii_coupling.txt").read_text()

TASK = """You are the worker for ONE step of a partitioned coupled simulation. Your step: write side B's participant for deal.II and make it run standalone. deal.II has no Python API, so this side is TWO files you write plus the build.
SUBDOMAIN B: the rectangle (0.6, 1.4) x (0, 1), steady conduction with k = 5:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 5*pi**2*(1.4 - x)*cos(pi*y)   (as written)
OUTER BOUNDARY: u = 0 on x = 1.4. The two y-faces (y = 0 and y = 1) carry NO condition: they are natural, i.e. zero flux.
INTERFACE (x = 0.6): this side is the NEUMANN side; it reads the partner's outward normal flux from ./imports.json (partner name "left"), applies it as its own natural condition on that face, and exports its own interface trace and its own consistent outward normal flux to ./exports.json.
MESH LEVEL 1: 8 x 10 elements, Q1 (FE_Q degree 1).
BUILD: deal.II is at /home/alexander/dealii/build. The Python interpreter that runs the wrapper is /home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python.
Hand in THREE files, each in its own fenced block and in this order:
  1. ```cpp      — solver.cc, the deal.II program. It is run by the wrapper; the two of them exchange plain text files of YOUR design.
  2. ```cmake    — CMakeLists.txt for it. The build is `cmake -S . -B build -DDEAL_II_DIR=/home/alexander/dealii/build -DCMAKE_BUILD_TYPE=Release && make -C build -j8`, and your executable must end up at ./build/solver.
  3. ```python   — participant_B.py, the wrapper. Copy the served contract, edit its placeholder block for this subdomain, keep every served line as given, and write the marked SOLVE holes yourself. It is run as `python participant_B.py` in the directory holding imports.json and build/.
Output ONLY those three fenced blocks."""


def _blocks(text: str) -> dict:
    out = {}
    for lang in ("cpp", "cmake", "python"):
        m = re.search(rf"```{lang}[^\n]*\n(.*?)```", text, re.S)
        if m:
            out[lang] = m.group(1)
    if "cpp" not in out:                      # some replies label it c++
        m = re.search(r"```(?:c\+\+|C\+\+)[^\n]*\n(.*?)```", text, re.S)
        if m:
            out["cpp"] = m.group(1)
    return out


def build_and_run(files: dict):
    if not {"cpp", "cmake", "python"} <= set(files):
        return False, f"missing file(s): handed in {sorted(files)}"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "solver.cc").write_text(files["cpp"])
        (d / "CMakeLists.txt").write_text(files["cmake"])
        (d / "participant_B.py").write_text(files["python"])
        ys = [i / 10 for i in range(11)]
        (d / "imports.json").write_text(json.dumps({"left": {
            "field_name": "temperature", "n_points": len(ys),
            "coordinates": [[0.6, y] for y in ys], "values": [0.0] * len(ys),
            "normal_fluxes": [5.0 * math.cos(math.pi * y) for y in ys]}}))
        c = subprocess.run(["cmake", "-S", ".", "-B", "build", f"-DDEAL_II_DIR={DEAL_II_DIR}",
                            "-DCMAKE_BUILD_TYPE=Release"], cwd=td, capture_output=True, text=True, timeout=900)
        if c.returncode != 0:
            return False, "cmake failed :: " + (c.stderr or c.stdout)[-500:]
        m = subprocess.run(["make", "-C", "build", "-j8"], cwd=td, capture_output=True, text=True, timeout=1800)
        if m.returncode != 0:
            return False, "make failed :: " + (m.stderr or m.stdout)[-700:]
        r = subprocess.run([PY, "participant_B.py"], cwd=td, capture_output=True, text=True, timeout=900)
        ex = d / "exports.json"
        if not ex.is_file():
            return False, f"built, but no exports.json (rc={r.returncode}) :: " + (r.stderr or r.stdout)[-500:]
        e = json.loads(ex.read_text())
        co = e["coordinates"]; v = [float(x) for x in e["values"]]; q = [float(x) for x in e["normal_fluxes"]]
        eu = max(abs(vv - 0.8 * math.cos(math.pi * c2[1])) for c2, vv in zip(co, v))
        eq = max(abs(qq + 5.0 * math.cos(math.pi * c2[1])) for c2, qq in zip(co, q))
        ok = len(v) > 3 and eu < 0.05 and eq < 0.25      # the shipped reference reaches 4.0e-3 / 5.8e-3
        return ok, f"built and ran | n={len(v)} u_err={eu:.2e} q_err={eq:.2e}"


client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=1200)
SYS = ("You are a finite-element simulation assistant. What follows is the documentation the OASiS "
       "server gave you for deal.II.\n\n" + served)
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=2000 + i, max_tokens=32000,
        extra_body={"reasoning": {"max_tokens": 6000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK}])
    text = resp.choices[0].message.content or ""
    files = _blocks(text)
    for lang, ext in (("cpp", "cc"), ("cmake", "cmake"), ("python", "py")):
        if lang in files:
            (HERE / f"dealii_{TAG}_{i}.{ext}").write_text(files[lang])
    ok, info = build_and_run(files) if files else (False, f"no fenced files (finish={resp.choices[0].finish_reason})")
    attempts = 1
    if not ok and REPAIRS > 0:
        for _r in range(REPAIRS):
            attempts += 1
            fix = client.chat.completions.create(model=MODEL, temperature=0.7, seed=3000 + i, max_tokens=32000,
                extra_body={"reasoning": {"max_tokens": 6000}},
                messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK},
                          {"role": "assistant", "content": text[:20000]},
                          {"role": "user", "content": "Those files were built and run exactly as specified and did NOT produce a correct exports.json. The output ended with:\n\n" + info[-1500:] + "\n\nHand in the corrected THREE files again, each in its own fenced block, keeping every served line as given."}])
            text = fix.choices[0].message.content or ""
            files = _blocks(text)
            for lang, ext in (("cpp", "cc"), ("cmake", "cmake"), ("python", "py")):
                if lang in files:
                    (HERE / f"dealii_{TAG}_{i}_fix{_r + 1}.{ext}").write_text(files[lang])
            ok, info = build_and_run(files) if files else (False, "no fenced files in the repair")
            if ok:
                break
    results.append({"i": i, "ok": ok, "info": info, "attempts": attempts, "secs": round(time.time() - t0, 1)})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | attempts={attempts} | {round(time.time()-t0)} s | {info[:300]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
