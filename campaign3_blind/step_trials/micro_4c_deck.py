"""Micro-test: can the 27B write a valid 4C deck for one subdomain from exactly what OASiS serves?

Usage: python micro_4c_deck.py <n_samples> [tag]   (OPENROUTER_API_KEY in the environment)
Validates each deck by RUNNING 4C on it (timeout 180 s). Prints per-sample verdicts and error classes.
The mini task is invented (not a development problem): subdomain (0.6,1.4)x(0,1), k=5, Neumann side at x=0.6.
"""
import os, re, sys, json, subprocess, tempfile, time
from pathlib import Path
from openai import OpenAI

HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TAG = sys.argv[2] if len(sys.argv) > 2 else "base"
MODEL = "qwen/qwen3.5-27b"
FOURC = "/home/alexander/4C/build/4C"
ENV = dict(os.environ, LD_LIBRARY_PATH="/opt/4C-dependencies/lib")

served = "\n\n".join(((HERE / n).read_text()) for n in
                     ("served_fourc_facts.txt", "served_fourc_contract.txt"))
if "THE 4C DECK GRAMMAR" not in served:
    served += "\n\n" + (HERE / "served_fourc_grammar.txt").read_text()

TASK = """You are solving ONE subdomain of a partitioned coupled problem with 4C (write a YAML input deck and run the 4C binary).
SUBDOMAIN B: the rectangle (0.6, 1.4) x (0, 1), scalar diffusion with k = 5:  -div(k grad u) = f
SOURCE TERM: f(x, y) = 3*x**2*y + 2*y**2 - 4*x*y**3   (as written; convert to 4C's function syntax yourself)
OUTER BOUNDARY (x = 1.4, y = 0, y = 1): u = 0 (Dirichlet)
INTERFACE (x = 0.6): this side is the NEUMANN side; it receives the partner's outward normal flux and applies it as its own inward flux. For THIS test use a constant imported flux q = 1.5 along the whole interface.
MESH LEVEL 1: 8 x 10 QUAD4 elements (uniform).
Write the COMPLETE 4C YAML deck `deck_level1.4C.yaml` for this subdomain and this level: every section 4C needs, all node coordinates and elements written out, the source as a function, the interface flux as a condition, and the boundary-flux output the coupling needs. Output ONLY the YAML inside one ```yaml fenced block, nothing else."""

def run_4c(yaml_text: str) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "deck_level1.4C.yaml"; p.write_text(yaml_text)
        try:
            # stdbuf: without line buffering MPI_Abort eats 4C's own error message
            r = subprocess.run(["stdbuf", "-oL", "-eL", FOURC, str(p), "out"], cwd=td, env=ENV, capture_output=True, text=True, timeout=180)
        except subprocess.TimeoutExpired:
            return False, "TIMEOUT"
        out = (r.stdout or "") + (r.stderr or "")
        ok = r.returncode == 0 and "finished normally" in out
        err = ""
        lines = out.splitlines()
        for k, line in enumerate(lines):
            if "PROC 0 ERROR" in line or "Failed to match" in line or "yaml" in line.lower() and "error" in line.lower():
                err = " / ".join(x.strip() for x in lines[k:k + 3])[:300]; break
        if not err:
            for line in lines:
                if "error" in line.lower() and "MPI_ABORT" not in line and "errorcode" not in line:
                    err = line.strip()[:200]; break
        return ok, (err or f"rc={r.returncode}")

def classify(yaml_text: str) -> dict:
    return {"problemtype": (re.search(r"PROBLEMTYPE:\s*\"?([A-Za-z_]+)", yaml_text) or [None, "?"])[1],
            "material": (re.search(r"MAT_[A-Za-z_]+", yaml_text) or [None])[0] if re.search(r"MAT_[A-Za-z_]+", yaml_text) else "?",
            "uses_**": "**" in yaml_text, "uses_^": "^" in yaml_text,
            "has_neumann": "NEUMANN" in yaml_text, "has_fluxcalc": "FLUX" in yaml_text.upper()}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
results = []
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.7, seed=1000 + i, max_tokens=16000,
        messages=[{"role": "system", "content": "You are a finite-element simulation assistant. What follows is the documentation the OASiS server gave you for 4C.\n\n" + served},
                  {"role": "user", "content": TASK}])
    msg = resp.choices[0].message
    text = msg.content or ""
    fin = resp.choices[0].finish_reason
    reasoning = getattr(msg, "reasoning", None) or (getattr(msg, "model_extra", None) or {}).get("reasoning")
    if not text.strip():
        print(f"[{TAG}] sample {i}: EMPTY content (finish_reason={fin}, reasoning_chars={len(reasoning or '')}, usage={getattr(resp, 'usage', None)})", flush=True)
    # take the yaml between the first ```yaml line and the closing fence, or to the end if the fence is missing
    m = re.search(r"```ya?ml[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    yaml_text = m.group(1) if m else text
    yaml_text = "\n".join(l for l in yaml_text.splitlines() if not l.strip().startswith("```")) + "\n"
    ok, err = run_4c(yaml_text)
    c = classify(yaml_text)
    results.append({"i": i, "ok": ok, "err": err, **c, "secs": round(time.time() - t0, 1), "chars": len(yaml_text)})
    print(f"[{TAG}] sample {i}: {'PASS' if ok else 'FAIL'} | {err[:110]} | {c}", flush=True)
    (HERE / f"deck_{TAG}_{i}.yaml").write_text(yaml_text)
passed = sum(r["ok"] for r in results)
print(f"[{TAG}] pass rate {passed}/{N}")
(HERE / f"results_{TAG}.json").write_text(json.dumps(results, indent=1))
