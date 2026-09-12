"""Dedicated wording test: when the coupled first reply lands, does the 27B PARENT's first worker brief carry the
task's data for subdomain A (source terms, coefficients, boundary values), or only the recipe?  Measured on cell
7022: the worker wrote placeholder source terms because the brief said 'from the task' and a worker sees only the
parent's task= string.  Arm A = the current first reply WITHOUT the 'THE WORKER SEES ONLY THIS BRIEF' sentence
(the wording before 0b58cec2); arm B = the current reply.  N samples each, one completion each.
Usage: python micro_parent_brief_data.py <n> [tag].  Neutral manufactured task, not a campaign task."""
import os, re, sys, json, time
from pathlib import Path
from openai import OpenAI
sys.path.insert(0, "/home/alexander/Schreibtisch/ofa-v2/src")
from core.instructions import INSTRUCTIONS
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
TAG = sys.argv[2] if len(sys.argv) > 2 else "pbrief"
MODEL = "qwen/qwen3.5-27b"
HERE = Path(__file__).resolve().parent
TASK = """Solve the following boundary value problem as a COUPLED simulation using TWO codes: 4C (write a YAML input file and run the 4C binary) on subdomain A and FEniCSx (dolfinx) on subdomain B.
DOMAIN: the rectangle (0, 1.5) x (0, 1), split at x = 0.8: subdomain A = (0, 0.8) x (0, 1), subdomain B = (0.8, 1.5) x (0, 1).
PHYSICS: steady thermoelasticity, plane strain, small strain, on both subdomains:
    -div(k grad T) = f_T
    -div(sigma_tot(u, T)) = (f_x, f_y),  sigma_tot = 2*mu*sym(grad(u)) + lambda*div(u)*I - beta*T*I
COEFFICIENTS: subdomain A: k = 2, lambda = 500, mu = 300, beta = 1.   subdomain B: k = 3, lambda = 600, mu = 1600, beta = 1.
SOURCE TERMS on subdomain A (as written):
    f_T = 2*pi**2*x*sin(pi*y)
    f_x = 0.02*pi**2*x*sin(pi*y)*(2*mu + lambda) - 0.01*pi*cos(pi*y)*(lambda + mu) + beta*sin(pi*y)
    f_y = 0.01*pi**2*x**2*sin(pi*y)*(2*mu + lambda) - 0.02*pi*cos(pi*y)*(lambda + mu) + beta*pi*x*cos(pi*y)
SOURCE TERMS on subdomain B: f_T = 0, f_x = 0, f_y = 0.
OUTER BOUNDARY: T = 0 and u = 0 on x = 0 and on x = 1.5; T = 0 and u = 0 on y = 0 and y = 1.
INTERFACE (x = 0.8): temperature, displacement, normal heat flux and traction continuous. Subdomain A is the Dirichlet side, B the Neumann side.
MESH LEVELS: three, 8x10 / 16x20 / 32x40 elements per subdomain-width unit as h = 0.1, 0.05, 0.025.
DELIVERABLES per level k: field_level<k>_<side>.csv at the probe points listed in probes.csv, interface_level<k>_<side>.csv, history_level<k>.csv, run_level<k>_<side>.log with that side's solver console and an NDOF line. Then summary.txt. Budget: 45 minutes."""
# arms: name=file pairs on argv[3:], e.g. B=served_fourc_thermoelastic_first_B.txt C=served_fourc_thermoelastic_first.txt
ARMS = []
for spec in sys.argv[3:]:
    name, fn = spec.split("=", 1)
    ARMS.append((name, (HERE / fn).read_text()))
if not ARMS:
    first = (HERE / "served_fourc_thermoelastic_first.txt").read_text()
    SENT = re.search(r"THE WORKER SEES ONLY THIS BRIEF, NOT YOUR\s+TASK:.*?placeholder source terms\)\. ", first, re.S)
    assert SENT, "the data-paste sentence is not in the served first reply"
    ARMS = [("A_old", first.replace(SENT.group(0), "")), ("B_new", first)]
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
SYS = INSTRUCTIONS + "\n\nHost-side tools (also available): run_bash, read_file, write_file, web_search, spawn_subagent(role, task, context)."


def ask(reply, seed):
    r = client.chat.completions.create(model=MODEL, temperature=0.7, seed=seed, max_tokens=3000,
        extra_body={"reasoning": {"max_tokens": 2000}},
        messages=[{"role": "system", "content": SYS}, {"role": "user", "content": TASK +
                   "\n\nknowledge(topic='coupling', solver='fourc', physics='thermoelastic') just returned:\n\n" + reply +
                   "\n\nState your NEXT tool call exactly as you would issue it (one call, with its complete arguments), and nothing else."}])
    return (r.choices[0].message.content or "").strip(), getattr(r, "provider", None)


def judge(a):
    first_call = re.search(r"(couple_levels|couple|spawn_subagent|audit_results|write_file|run_bash|knowledge|check_input)\s*\(", a)
    call = first_call.group(1) if first_call else "?"
    m = re.search(r"spawn_subagent\s*\((.*)", a, re.S)
    brief = m.group(1) if m else ""
    src = bool(re.search(r"sin\(\s*pi\s*\*\s*y\s*\)|pi\*\*2|pi\^2|pi²", brief))
    coef = bool(re.search(r"\b500\b", brief)) and bool(re.search(r"\b300\b", brief))
    bc = bool(re.search(r"T\s*=\s*0", brief))
    unfilled = "<SUBDOMAIN A, COPIED FROM YOUR TASK" in brief or "<THAT SUBDOMAIN, COPIED" in brief
    return {"first_call": call, "brief_chars": len(brief), "source_terms": src, "coefficients": coef, "boundary": bc,
            "placeholder_left": unfilled, "data_pasted": src and coef and not unfilled}


out = {}
for name, reply in ARMS:
    hits = 0; answers = []
    for i in range(N):
        t0 = time.time()
        try:
            a, prov = ask(reply, 9100 + i)
        except Exception as e:                       # noqa: BLE001
            print(f"[{TAG}] {name} sample {i}: API error {e!r}", flush=True); continue
        j = judge(a); j.update({"provider": prov, "secs": round(time.time() - t0), "text": a[:1500]})
        hits += int(j["data_pasted"]); answers.append(j)
        print(f"[{TAG}] {name} sample {i}: first call = {j['first_call']}; brief {j['brief_chars']} chars; source terms {j['source_terms']}; "
              f"coefficients {j['coefficients']}; boundary {j['boundary']}; DATA PASTED = {j['data_pasted']} | {j['secs']}s {prov}", flush=True)
    out[name] = {"data_pasted": hits, "n": len(answers), "answers": answers}
    print(f"[{TAG}] {name}: data pasted into the worker brief in {hits}/{len(answers)}", flush=True)
    (HERE / f"results_{TAG}.json").write_text(json.dumps(out, indent=1))
