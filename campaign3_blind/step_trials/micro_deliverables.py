"""Micro-test (ladder step 4): given a converged side's own per-level dump and exports.json, can the 27B
write the per-level field file at prescribed probe points (interpolated) and the interface file?
Validated against the analytic field the dump was made from.  Usage: python micro_deliverables.py <n> [tag]"""
import os, re, sys, json, subprocess, tempfile, time
import numpy as np
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "deliv"
MODEL = "qwen/qwen3.5-27b"
PY = "/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python"

def u_exact(x, y): return x * y * (1.0 - y) + 0.1 * x * x
rng = np.random.default_rng(7)
PROBES = [[round(float(a), 4), round(float(b), 4)] for a, b in zip(rng.uniform(0.02, 0.58, 20), rng.uniform(0.02, 0.98, 20))]
IFACE = [[0.6, round(float(y), 4)] for y in np.linspace(0.1, 0.9, 5)]

def make_dump(d: Path):
    nx, ny = 12, 20
    rows = ["x,y,u"]
    for j in range(ny + 1):
        for i in range(nx + 1):
            x, y = 0.6 * i / nx, 1.0 * j / ny
            rows.append(f"{x:.10g},{y:.10g},{u_exact(x, y):.12g}")
    (d / "field_level1.csv").write_text("\n".join(rows) + "\n")
    ys = [j / ny for j in range(ny + 1)]
    (d / "exports.json").write_text(json.dumps({"field_name": "u", "n_points": len(ys),
        "coordinates": [[0.6, y] for y in ys], "values": [u_exact(0.6, y) for y in ys],
        "normal_fluxes": [-(y * (1 - y) + 0.12) for y in ys]}))   # some outward flux profile

TASK = f"""You are the worker for ONE step (ladder step 4) of a partitioned coupled simulation: write level 1's deliverables for side A.
In the working directory you have side A's converged level-1 field dump ./field_level1.csv (columns x,y,u: nodal values of a P1 field on a uniform 12 x 20 quad mesh of the rectangle (0,0.6) x (0,1)) and its ./exports.json, which is exactly {"field_name", "n_points", "coordinates": [[x, y], ...], "values": [...], "normal_fluxes": [...]} in interface-node order (the interface trace u and this side's own outward normal flux at the interface nodes on x = 0.6).
Write ./solution_level1_A.csv with columns x,y,u: the field EVALUATED AT THESE PRESCRIBED PROBE POINTS by interpolation inside the element that contains each point (bilinear on the quad, or P1 on its triangles); never nearest node:
{json.dumps(PROBES)}
Write ./interface_level1_A.csv with columns x,y,u,qn: the interface trace u and this side's OWN outward normal flux qn at these prescribed interface points, interpolated from exports.json along y:
{json.dumps(IFACE)}
Write ONE Python script make_level1.py that does both (numpy available), output ONLY the Python inside one ```python fenced block."""

def validate(d: Path) -> tuple[bool, str]:
    sol, ifc = d / "solution_level1_A.csv", d / "interface_level1_A.csv"
    if not sol.is_file() or not ifc.is_file():
        return False, f"missing files: solution={sol.is_file()} interface={ifc.is_file()}"
    try:
        S = np.genfromtxt(sol, delimiter=",", names=True); I = np.genfromtxt(ifc, delimiter=",", names=True)
    except Exception as e:
        return False, f"unreadable csv: {e}"
    if len(S) != len(PROBES): return False, f"solution rows {len(S)} != {len(PROBES)} probes"
    err = max(abs(float(u) - u_exact(float(x), float(y))) for x, y, u in zip(S["x"], S["y"], S["u"]))
    exp = json.loads((d / "exports.json").read_text()); ys = np.array([c[1] for c in exp["coordinates"]]); q = np.array(exp["normal_fluxes"])
    qerr = max(abs(float(qn) - float(np.interp(float(y), ys, q))) for y, qn in zip(I["y"], I["qn"])) if len(I) == len(IFACE) else 9.9
    ok = err < 5e-3 and qerr < 1e-6 and len(I) == len(IFACE)
    return ok, f"max|u-exact| at probes={err:.2e} (bilinear P1 on h=0.05 should be <5e-3), interface rows={len(I)}, max|qn-interp|={qerr:.1e}"

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.7, seed=5000 + i, max_tokens=12000,
        extra_body={"reasoning": {"max_tokens": 4000}},
        messages=[{"role": "system", "content": "You are a finite-element simulation assistant working one bounded step of a coupled run."},
                  {"role": "user", "content": TASK}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"```python[^\n]*\n(.*?)(?:```|\Z)", text, re.S); code = m.group(1) if m else text
    code = "\n".join(l for l in code.splitlines() if not l.strip().startswith("```")) + "\n"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); make_dump(d); (d / "make_level1.py").write_text(code)
        r = subprocess.run([PY, "make_level1.py"], cwd=td, capture_output=True, text=True, timeout=300)
        ok, info = (validate(d) if r.returncode == 0 else (False, "rc=%d :: %s" % (r.returncode, (r.stderr or "")[-300:])))
    (HERE / f"deliv_{TAG}_{i}.py").write_text(code)
    results.append({"i": i, "ok": ok, "info": info, "secs": round(time.time() - t0, 1)})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | {info[:260]}", flush=True)
print(f"[{TAG}] pass rate {sum(r['ok'] for r in results)}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
