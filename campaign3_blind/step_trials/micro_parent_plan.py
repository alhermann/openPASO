"""Micro-test: given a candidate OASiS step ladder, does the 27B as PARENT produce bounded, verifiable
sub-agent briefs instead of attempting (or refusing) the whole task?  Usage: python micro_parent_plan.py <n>"""
import os, re, sys, json, time
from pathlib import Path
from openai import OpenAI
HERE = Path(__file__).resolve().parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MODEL = "qwen/qwen3.5-27b"

LADDER = """COUPLED WORKFLOW LADDER (from OASiS). A partitioned coupled run is these steps, in this order, and each step has a
check you can see on disk before the next. Give EACH step to ONE sub-agent with only that step's brief and the OASiS
material for that code (knowledge(topic='coupling', solver=<code>) and prepare_simulation(<code>, <physics>)); the
sub-agent works until the check passes or reports the exact error, and you never judge the whole job at once.
  1. side A participant runs STANDALONE in ./side_A with a synthetic ./imports.json and ./config.json ->
     check: exit code 0, ./side_A/exports.json exists with finite values and fluxes, its export self-check passed.
  2. side B participant runs STANDALONE in ./side_B the same way -> same check.
  3. couple(level 1): call couple() with both participants and history_path -> check: converged true, the residual
     history file for level 1 exists with >= 3 rows and a falling residual.
  4. level-1 deliverables: each side's field file at the prescribed probe points and its interface file at the
     prescribed interface points, from the converged fields; each side's run log = the captured participant log ->
     check: the files exist and the audit_results tool reports no missing-fields finding for level 1.
  5. repeat 3-4 for levels 2 and 3 (config.json level=2,3) -> same checks.
  6. summary file naming ONLY files that exist, then audit_results on the working directory -> check: no
     invented names, no missing levels.
A sub-agent's brief must state: the code, the role (Dirichlet or Neumann), the subdomain and its data, which OASiS
calls to make, exactly what files to produce, and the check that ends the step."""

TASK = """Solve the following boundary value problem as a COUPLED simulation using TWO codes: DUNE-fem on subdomain A and 4C on subdomain B.
GLOBAL DOMAIN: the rectangle (0, 1.4) x (0, 1). SUBDOMAIN A: (0, 0.6) x (0, 1), k = 1. SUBDOMAIN B: (0.6, 1.4) x (0, 1), k = 5.
INTERFACE: x = 0.6. EQUATION: -div(k grad u) = f in each subdomain, u = 0 on the outer boundary, u and k*du/dn continuous across the interface.
SOURCE: f(x, y) = 2*x*y**2 - x**2 (A), f(x, y) = 3*x**2*y + 2*y**2 - 4*x*y**3 (B).
ROLES: A is the Dirichlet side, B the Neumann side. MESH LEVELS: h = 1/8, 1/16, 1/32 (three levels).
DELIVERABLES per level and side: a field file at the 100 listed probe points, an interface file at the 20 listed interface points, a residual history per level, a run log per side and level, and a summary file. Time budget: 45 minutes."""

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
for i in range(N):
    t0 = time.time()
    resp = client.chat.completions.create(model=MODEL, temperature=0.5, seed=3000 + i, max_tokens=6000,
        messages=[{"role": "system", "content": "You are the PARENT agent of a coupled simulation run. You have tools spawn_subagent(role, task, context), run_bash, write_file and the OASiS MCP tools. You do not solve steps yourself; you delegate bounded steps and check their results.\n\n" + LADDER},
                  {"role": "user", "content": TASK + "\n\nProduce your plan as a JSON list of sub-agent briefs, one per ladder step, each with keys: step, code, role, brief, oasis_calls, files_to_produce, check. Output ONLY the JSON in one ```json fenced block."}])
    text = resp.choices[0].message.content or ""
    m = re.search(r"```json[^\n]*\n(.*?)(?:```|\Z)", text, re.S)
    raw = m.group(1) if m else text
    try:
        plan = json.loads(raw)
    except Exception as e:
        print(f"[plan] sample {i}: UNPARSABLE ({e}); first 200 chars: {text[:200]!r}"); continue
    steps = plan if isinstance(plan, list) else plan.get("steps") or plan.get("plan") or []
    def has(s, *words): return any(w in json.dumps(s).lower() for w in words)
    verdict = {
        "n_steps": len(steps),
        "standalone_A_first": bool(steps) and has(steps[0], "standalone", "exports.json"),
        "couple_step": any(has(s, "couple(") or has(s, "couple") for s in steps),
        "field_files_step": any(has(s, "field file", "probe points") for s in steps),
        "every_step_has_check": all(bool(s.get("check")) for s in steps if isinstance(s, dict)),
        "levels_2_3": any(has(s, "level 2", "level=2", "levels 2") for s in steps),
        "gave_up": has(plan, "could_not_complete", "cannot be completed", "not feasible"),
    }
    print(f"[plan] sample {i} ({time.time()-t0:.0f}s): {verdict}")
    (HERE / f"plan_{i}.json").write_text(json.dumps(plan, indent=1))
