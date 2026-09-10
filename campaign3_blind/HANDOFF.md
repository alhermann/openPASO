# Campaign 3 blind re-run — full handoff for a parallel agent

> **INTERNAL. DO NOT PUSH THIS FILE.** `campaign3_blind/` is tracked and the
> remote `hereon` is the **public** repo `Hereon-InstituteMS/OASiS`. This file
> contains internal status and unpublished numbers. It carries **no secrets** —
> credentials are named by location only (§2) — but it is not for publication.

Written 2026-09-01 against branch `consolidation/v2`; the late addendum below
records the tested development patch committed alongside it.
Every number was checked against the tree that day. Where something could not
be re-measured it says so. **If the tree contradicts this file, the tree wins:
fix the file and record what changed.**

---

## Late addendum: round 9 and root-cause repair

Round 9 is diagnostic material, not a population. Claude committed agent-facing
source while it was running; all seed-96 ledgers predate per-run source hashes.
The parent launcher has been terminated after its children exited, no
`run_blind.py` process remains, and all three live key stores are sealed. Do not
resume the launcher or include seeds 96/97 in a quotable regrade.

Measured causes of low performance:

- Every MCP call spawned a new OASiS process. Critic reviews and every other
  in-memory gate vanished before the next call. One persistent stdio session now
  spans the complete campaign, validation, or WebUI agent lifetime.
- Host `read_file`, shell commands, and MCP-launched code could read sibling
  runs and shared `/tmp`. Round-96 C4 demonstrably copied prior participant
  scripts from `/tmp` and then ran FEniCS on both prescribed-code sides. Both
  execution routes now share a fail-closed bubblewrap namespace with private
  scratch, masked home/proc, scrubbed credentials, and explicit read-only solver
  runtimes.
- The runner used live source and inferred build drift from ledger mtimes. It
  now mounts a verified content-addressed committed snapshot, records its hash,
  locks each model/phase/seed population to one hash, and checks it again after
  the run. Old ledgers are `LEGACY_UNPINNED`, not guessed into a build.
- The live MCP surface exposed all 21 registered tools, including deprecated and
  environment-mutating entries. The campaign contract is now an explicit 16-tool
  allowlist and fails if one is missing.
- Natural method-qualified requests were routed to plausible but wrong generic
  templates: DUNE SIPG advection-diffusion to reaction-diffusion, and 4C
  Crank-Nicolson heat to generic heat. Both routes are fixed; DUNE now has a real
  runtime-verified steady SIPG capability rather than a renamed pure-advection
  template.
- All 18 observed FE2 MCP runs asked for generic `linear_elasticity`, never the
  existing mixed recipe. Its canonical knowledge was also stale and contradicted
  its generator. The prompt now preserves method qualifiers, and the live
  Taylor-Hood recipe states the correct pressure convention and mixed-subspace
  boundary mapping seen to fail in seed 96.
- In 136 historical grade-1 MCP coupled runs, 71 wrote participant
  `exports.json`, but only 19 completed the full solution/interface/residual/log
  contract. The driver discarded native participant stdout/stderr; it now
  persists bounded `participant_output.log` files and returns their paths.

Claude's focused repairs were checked independently: 10/10 flux-from-field
tests, 5/5 PDE-scope tests, and 17/17 knowledge-rule propagation tests pass. The
flux diagnostic is correctly non-gating and currently adds diagnosis rather
than a headline detection count.

No current-build success rate is available yet. The existing grade JSONs were
regenerated while grading logic was changing and round 9 is unpinned. Commit the
tested patch, run a small non-blind path check through the namespace, then use a
new seed for the next paid development probe. Regrade only after confirming no
agent process exists; the vault passphrase never belongs in argv, a file, or an
environment variable.

---

## 0. The one thing to read first

The headline number everyone quotes — **41.7% single-code OASiS** — comes from
runs executed in **August, before every knowledge fix made on 2026-09-01**.
Nobody has measured a single-code population on the current build. The only
post-fix data is three runs, and all three are CORRECT. **Re-measuring that is
the highest-value action available.** Details in §5 and §11.

---

## 1. Goals

### The mission, in two clauses

The OASiS paper claims a small open-weight model plus OASiS beats the same model
alone on computational-mechanics tasks. The earlier experiments had a flaw:
**the agents were given the exact solution.** A convergence claim built on that
measures the tool against an answer it was handed.

So the experiment is being re-run **blind**:

- the agent receives the **task prompt and nothing else**;
- the exact solution lives in a **sealed key** the agent never touches;
- the key is opened **only by the grader, offline, after the run is over**.

Clause **(a)**: make every claim in the paper literally true.
Clause **(b)**: then obtain the target numbers.
Clause (a) has moved a long way. Clause (b) has not. **Never conflate them** —
most of the work so far belongs to (a).

### The targets (`campaign3_blind/SCORECARD.md` is authoritative)

```
at 27B        OASiS > 70%, bare ~36%, uplift > +33 points, McNemar significant
at 122B       OASiS > 61%
at 397B       OASiS > 66%
coupling      bare completes zero real coupled runs and fabricates;
              OASiS completes them
substitution  27B + OASiS >= 397B bare (67%)
cost/energy   better than 1.2x lower per solved task
reliability   fabrication near zero in the OASiS arm
```

**Met: fabrication only.** Two of these statements are already contradicted by
measurement (§5).

### The phase, and the rule that ends it

**Development, 27B rung.** It ends when a ROUND teaches OASiS **nothing new**
(the paper's §3.2 criterion) — **not** when a score is hit.

After that: freeze → draw fresh problems → run 27B, then 122B, then 397B; both
arms; single-code and coupled; all nine backends.

### The development loop

1. one cell, 3 seeds (~20 min; **~$1.74 per single-code run, ~$6 per coupled
   run**, derived from credit deltas)
2. read with `campaign3_blind/cell_read.py`
3. root-cause it
4. distil a **GENERAL** primitive and **verify it by EXECUTING it**
5. re-run the same cell
6. repeat until the cell teaches nothing new
7. then re-run **already-converged** cells — a fix that only helped the cell it
   came from was not a primitive

---

## 2. Resources and access

### OpenRouter — the model access the runs need

The key is in **`/home/alexander/Schreibtisch/ofa-v2/.env`** as
`OPENROUTER_API_KEY`. `.env` is gitignored (`.gitignore:48: *.env`) and the
secret is deliberately **not reproduced here**, because this file sits in a
tracked directory whose remote is public.

```bash
cd /home/alexander/Schreibtisch/ofa-v2
set -a; . .env; set +a          # exports OPENROUTER_API_KEY

# balance, and what the pool has spent:
curl -s https://openrouter.ai/api/v1/credits \
  -H "Authorization: Bearer $OPENROUTER_API_KEY" | python3 -c \
  "import json,sys; d=json.load(sys.stdin)['data']; \
   print(f\"remaining \${d['total_credits']-d['total_usage']:,.2f}\")"
```

- **$238.56 remaining** as of 2026-09-01 (of $21,901.36 granted). That is
  roughly **135 single-code runs or 40 coupled runs.**
- Models are reached through OpenRouter; the size→model map is `OR_MODELS` in
  `campaign3_blind/run_blind.py`. `27b` is the development rung; `122b` and
  `397b` come later.
- **Billing settles late.** A reading taken right after a batch can understate
  the cost by ~$24. A 402 comes from the shared *pool*, not the key limit, and
  the pool has other consumers.

### Everything else

| what | where |
|---|---|
| **the checkout the campaign runs from** | `/home/alexander/Schreibtisch/ofa-v2`, branch `consolidation/v2` |
| MCP server source | `ofa-v2/src/` — `tools/consolidated.py`, `tools/knowledge.py`, `backends/*` |
| grader | `campaign3_blind/grade_blind_v2.py`, `campaign3_blind/grading/*`, `src/blind_eval/evidence.py` |
| agent harness | `ofa-v2/langgraph_eval/agent.py` |
| runner | `campaign3_blind/run_blind.py` |
| tests | `ofa-v2/tests/` — **3559 collected** |
| task texts (main draw, 47 cells) | `campaign3_blind/problems/` |
| newer draws | `campaign3_blind/problems_dev2/`, `problems_dev3/` (C2 only) |
| **sealed answer keys** | `~/Schreibtisch/qwen_uplift_test/campaign3_blind/{keys, keys_dev3, keys_backup_20260816}` |
| run outputs | `campaign3_blind/runs/<CELL>_27b_<BARE\|MCP>_seed<N>/` — 940 dirs |
| regrade output | `campaign3_blind/regrade_after_repairs/SUMMARY.md` |
| sealed round archives | `campaign3_blind/rounds/` (permission-denied by design) |
| Overleaf, **OASiS paper** | `~/Schreibtisch/open-fem-agent/papers/.env` → `OVERLEAF_TOKEN`, `OVERLEAF_PROJECT_ID`; submodule `papers/overleaf-paper/` |
| Overleaf, **SPH-PD paper** (different paper) | `~/.overleaf.env` → `OVERLEAF_GIT_TOKEN`; project `699dd6b46370fe3a79f969f6`; clone `~/overleaf-pd-paper/` |
| **vault passphrase** | ask Alexander. Never in argv, never in a file, never in an environment variable — terminal or one line of stdin only. |
| sudo password | ask Alexander (needed for conda/4C admin operations) |

### Interpreters — neither venv runs everything

| venv | has | use for |
|---|---|---|
| `~/Schreibtisch/open-fem-agent/.venv` | numpy, sympy, meshio, dolfinx, dune | the MCP server, the grader, most tests |
| `~/Schreibtisch/open-fem-agent/.venv-lg` | + langchain/langgraph (+ sympy, meshio added 2026-09-01) | `run_blind.py`, `langgraph_eval` tests, **the full suite** |

Run the full suite with **`.venv-lg`**: `.venv` cannot import 7 langchain tests.
A full run is ~92 minutes and spawns real solvers. Do not run two at once —
that causes `test_e2e.py` failures that do not reproduce alone (27/27 pass).

### Solvers

- 4C: `/home/alexander/4C/build/4C`, needs `LD_LIBRARY_PATH=/opt/4C-dependencies/lib`
- 4C source, the authority on its own grammar: `/home/alexander/4C/src/`
- **`4C -p` dumps the whole input grammar** — better than grepping its tests
- deal.II: `/home/alexander/dealii`; preCICE 3.1.2: `/opt/precice/lib`

### Why `ofa-v2` and not `ofa-v3`

There are **46 git worktrees** of the same repo under `~/Schreibtisch/ofa-*`.

- `ofa-v2` (branch `consolidation/v2`) **has `campaign3_blind/`**, all 940 run
  directories, the grader, and 1031 commits that `v3-verification` does not.
- `ofa-v3` (branch `v3-verification`) has **no `campaign3_blind/` at all** and
  is 3 commits ahead on an old base — a stale feature line for a "v3
  verification layer", last touched long before this campaign.

So the campaign is in v2 because that is where it was built and where every
path points. Merging v3's 3 commits is a separate, unresolved question; nothing
in the campaign depends on it. **A previous session lost hours editing the wrong
checkout (closed item #60) — check `pwd` before editing.**

`~/Schreibtisch/open-fem-agent` is a *different* checkout (branch
`fix/coupling-4c-sparta`) that holds the venvs and the `papers/` submodule.
Do not run the campaign from it.

---

## 3. Constraints

### Option B — what OASiS may and may not serve

- **Elided at serving time:** the mesh, the weak form, the material, the source
  term, the linear solve. The agent writes those.
- **Served deliberately, because the gate grades against them:** the coupling
  handshake (`imports.json` / `exports.json`), the interface sign convention,
  consistent flux recovery (`r = A·u − b_vol`, `q = −r/w`), the exports schema,
  and each **binary** backend's input-file grammar. An agent cannot be graded on
  a contract it was never told.
- A binary backend's input file (4C, FEBio, SPARTA) is its **run interface**,
  not its method. Serving the deck grammar is in scope; serving the solve is not.
- OASiS **never** installs solvers or files into the agent's workspace.

### Experimental hygiene

- **Development problems are burnt** — they can never be the paper's numbers.
- **Never tune to make the development set pass.** Knowledge that quotes a drawn
  problem's dimensions collapses on the fresh draw. Machine-enforced (§8).
- **Three evidence grades (1 = true error, 2 = reference, 3 = band only) are
  never pooled.**
- **A `FABRICATED` label requires positive evidence of invention.** Deficiency
  is not forgery. Violating this has repeatedly cost the bare arm runs it earned.
- **The grader may not punish what the task did not ask.** Three separate gates
  have been repaired to this principle.
- Only the OASiS arm has the knowledge tool, so anything served is an
  experimental artefact the bare arm cannot receive — including phrasing.

### Security

- Answer keys must **always end sealed** (`d---------`). Verify with
  `stat -c %A "$OASIS_BLIND_KEYS"` after every grading window.
- The vault passphrase goes on the terminal or one line of stdin. Never argv,
  never a file, never an environment variable.
- Do not read credentials from another process's `/proc/<pid>/environ`.

### What OASiS may say about itself, and where (rule of 2026-09-03, applied 2026-09-10)

- OASiS is a general product: **no benchmark file name, verdict token, run
  identifier or grader vocabulary anywhere in `src/`** — served strings,
  docstrings, comments and code alike. The task text is the only contract
  carrier. Swept on 2026-09-10 on Alexander's explicit scope ("that should not
  be in there, of course not... check other places").
- The audit and the workspace advisor discover deliverables generically:
  `<kind>_level<k>[_<side>].<ext>` read from the agent's own file names and
  classified by content (field / interface / residual history / run log); the
  summary file is the caller's hint (the harness passes the file it saw
  written) or a `result|summary|report|answer` name. A DOF-count line is any
  `NDOF|DOFS|degrees of freedom = n` spelling.
- The harness carries no coaching text at all: its OASiS-arm system prompt IS
  the server's own instructions string (`core.instructions.INSTRUCTIONS`, the
  same bytes server.py hands to FastMCP) plus the host-tool list.
- The participant PARTS door (`signal='participant[:role]:partN'`) serves the
  ELIDED contract in bounded chunks and claims no complete file; it used to
  read the raw file and hand over its SHA-256. Kratos participants kept
  `strategy.Solve()` outside the markers until 2026-09-10; every shipped
  participant now ends with an EXPORT SELF-CHECK (non-finite export, a Neumann
  side whose imported load never arrived, a Dirichlet side exporting the
  partner's negated array), execution-verified with scikit-fem.

### Disclosure obligation

The paper **must state** that OASiS was hardened against this benchmark's I/O
contract during development. It was. Saying so is the difference between a
result and a claim. Since 2026-09-10 the product no longer names that contract
anywhere (see the section above); the statement stays, because the checks
were still shaped by the development runs' failure classes.

---

## 4. What has been going on

### Seven development rounds at 27B (from SCORECARD.md, recounted 2026-08-29)

Single-code, 16 FEM cells, evidence grade 1:

| round | bare | OASiS | uplift |
|---|---|---|---|
| 3 | 22.9% | 33.3% | +10.4 |
| 4 | 25.0% | 37.5% | +12.5 |
| 5 | 18.8% | 31.2% | +12.5 |
| 6 | 25.0% | 31.2% | +6.2 |
| 7 | 18.8% | 37.5% | +18.8 |

**There is no trend.** The uplift bounces between +6 and +19 across five rounds
and the target is +33. Coupled was 0 solved in every round.

### What the 2026-09-01 session did

Almost all of it belongs to clause (a) — making the measurement trustworthy —
and several fixes **lowered** the apparent uplift:

**Delivery defects (the knowledge existed and did not arrive):**
- `_UNIVERSAL` was appended on **1 of 31** return paths of `knowledge()`; 66% of
  calls ask `topic="pitfalls"` and only 11.5% ask `physics`, so **75.6% of
  OASiS-arm runs received none of it**. Fixed by wrapping the tool once.
- The coupling must-read was attached **only when the payload was long enough to
  truncate**, so `solver=`/`signal=` calls got none of it — while the truncation
  notice told agents to come back with `signal=`.
- The "rest is available on request" escape hatch **returned nothing new**.
  Now real: a signalled request returns what the head cut, ranked by rarity.
- The auto-audit watched `write_file` while **57%** of submissions were written
  by a `run_bash` heredoc. Now both routes.

**Grader defects (all charged mostly to the bare arm):**
- SPARTA writes `log.sparta`; `.sparta` was not readable, so its three measured
  signatures **could never fire**. 67 runs hold real SPARTA output.
- The leading-non-finite drop ran **after** the check it protects.
- The forgery threshold's "empty valley" turned out to be **two levels of a
  single run**. Recalibrated to machine precision. **12 runs lost the forgery
  label, 11 of them bare.**
- 173 coupled runs were rejected for supplying exactly what the task asked
  (0 of 47 tasks demanded per-code output). Now recorded as
  `per_code_attribution: UNPROVEN` instead of rejected.
- An X11 warning printed without a newline voided 19 run logs.
- A fixed-grid DSMC task was failed for having a fixed grid.
- Re-running the solver got a submission rejected — 18 runs, **all OASiS-arm**.

**Knowledge added, each verified by execution:**
- 4C: `E: 0` **segfaults with no message** (exit 139 vs exit 0 for `E: 1`).
- 4C: `SOLID` is 3-D, `WALL` is 2-D; **2-D `Thermo_Structure_Interaction` is
  impossible in this build** (clone strategy accepts only `SolidScatra`, whose
  cell types are all 3-D).
- deal.II was the **only** backend with no off-node read recipe;
  `VectorTools::point_value` added, verified by compiling (order 1.98, 2.03).
- New tool **`verify_pde_consistency`**: checks ∫u·(L\*v) = ∫f·v from the task's
  own operator and the agent's own values — no reference solution. Separated 18
  CONSISTENT from 8 INCONSISTENT across 26 runs. One run has already used it
  unprompted.
- Six contaminated passages removed (served knowledge stated measured
  convergence orders as per-level triplets).

**Two hypotheses tested and refuted** — do not redo these:
- Pushing agents toward `couple()`: 74 runs that used it → 1 CORRECT (1.4%);
  84 that hand-rolled → 1 (1.2%).
- Cutting served volume: A/B with `OASIS_LEAN_PHYSICS`, full **3/3 CORRECT**,
  lean **0/3**. Actions *fell* 7.7%. The elaboration is load-bearing.

---

## 5. Status, measured

### Honest coupled rounds of 2026-09-10 (Option B build, 27B, development seeds)

| round | commit | seeds | outcome | what the trajectories showed |
|---|---|---|---|---|
| 2 | 43a718dc | C3 5561-63, C1 6211-13, C2 6311-13 | 0/9 (8 HONEST_INCOMPLETE, 1 MALFORMED) | must-read served 5x in 7 calls; 0/8 kept a served contract; false "couple exchanges one field"; a DUNE JIT compile killed by the agent's own `timeout 120` read as a crash; couple() led with "history too short" instead of the crashed side |
| 3 | 6b6ad83f | C3 5571-73, C1 6221-23, C2 6321-23 | 0/9 (5 HONEST_INCOMPLETE, 4 FAILED) | 9/9 had one refused write (project dir under $HOME); 7/9 never called prepare_simulation, so the reveal never fired; 0/9 kept a contract; 0.7-9M input tokens per run |
| 4 | 86a86615 | C3 5581-83, C1 6231-33, C2 6331-33 | 0/9 (5 HONEST_INCOMPLETE, 3 FAILED, 1 MALFORMED) | refused writes 4/9; C3 5581 copied both contracts and got DUNE working, lost on a 4C MATERIALS key; C1 6233 both codes proven at level 1; C2 6333 ran a real 3-level coupling (9 it -> 1e-8) but wrote no field files and listed 12 nonexistent files in its summary |
| 5 | a4bebe41 | C3 5591-93, C1 6241-43, C2 6341-43 | 0/9 (6 HONEST_INCOMPLETE, 3 FAILED) | real couplings in 6 runs (one through 3 levels); runs stop at 16-23 min with 25 min left and write no field files; 0/9 copy a contract; C1 6243 wrote all 21 files with the zero solution (no source terms in the 4C deck) and honestly gave up |
| 6 | 081a1edd | C3 5611-13, C1 6261-63, C2 6361-63 | in flight ~19:52 | tests the 48k reply cap and the no-field-files hand-in finding |

Every coupled CORRECT recorded before 2026-09-10 on C3 (2/3, 3/6) was measured
with complete participants served and is not a paper number.


### The regrade (pre-fix; seeds 2–11, 14, 15; 750 rows; both arms)

| slice | bare | OASiS | uplift |
|---|---|---|---|
| single-code, grade 1 | 44/192 = 22.9% | 80/192 = **41.7%** | +18.7 |
| coupled, grade 1 | 1/138 = 0.7% | 2/136 = 1.5% | +0.7 |
| fabrication | 5/376 = 1.3% | 2/374 = **0.5%** | — |

### The only post-fix data

| cell | seed 92, current build | graded |
|---|---|---|
| NG1 | order 2.063 | **CORRECT** |
| KR1 | order 2.005 | **CORRECT** |
| FC1 | order 2.000 | **CORRECT** |

Three cells, one seed each. **Not a rate** — but it is the only measurement of
the current build and it has not been extended.

### ROUND 9 IS IN FLIGHT — do not launch a second round

Started 2026-09-01 11:36 by `campaign3_blind/round9_driver.sh`, build
`7e15a627`, log `campaign3_blind/round9.log`. Full matrix: 32 cells (9 backends
x both halves + all 14 coupled), seeds **96 and 97**, both arms, 128 runs, 6
concurrent, ~60 credits. It exists because everything in the table above was
measured 29–30 Aug on seeds 14/15 and **eight commits landed on 1 Sep inside
the channel an agent reads**, so no post-fix rate existed at all.

Two rounds cannot share the box. The driver refuses to start if any
`run_blind.py` is alive and holds `/tmp/oasis_round9_driver.lock`; a CPU-heavy
job beside a round inflates its wall-clock and its timeout count (measured: a
niced fixture at 1226% CPU did exactly that). **Before you run anything heavy,
check `pgrep -f 'run_blind[.]py' | wc -l`.** While it runs, take a code item
from §10 — those cost no CPU.

When it finishes: `grade_round.py` on seeds 96 and 97, then `cell_read.py` on
every wrong cell. That is the first honest post-fix number and it replaces the
41.7% above.

#### DO NOT COMMIT TO `src/` WHILE A ROUND IS IN FLIGHT

Learned the hard way inside round 9. Every `run_blind.py` invocation is a fresh
process importing OASiS from the working tree, so a commit mid-round means later
runs are served knowledge the earlier ones never saw. Three commits landed
between 14:42 and 15:10 on 2026-09-01 and **17 of the round's OASiS-arm runs
were served the earlier build**, making that arm a mixture of two builds. The
bare arm is unaffected because it calls no OASiS tool — so the damage is
one-sided, which is the worse kind: it moves the uplift without moving the
control.

    python campaign3_blind/build_drift.py 96 97

lists exactly which runs predate the newest agent-facing commit and are
therefore due a re-run. `src/` is frozen for the rest of round 9; the 17 named
runs get re-run when the machine is free, and then the round is a single-build
measurement again. Grading code, `cell_read.py` and documentation are NOT
agent-facing and stay editable — the grader runs offline, after the fact.

### Two target statements the data contradicts

- **"bare completes none and fabricates" is not supported, but it is also not
  yet refuted — and NEITHER grades directory can settle it.** An earlier
  version of this section claimed `C8_27b_BARE_seed4` is CORRECT at order
  2.017. **That was wrong.** Checked in both records on 2026-09-01:
  `grades_pre_repair` grades it **COMPLETED_UNPHYSICAL** (order 2.0169867,
  reason `INTERFACE_NOT_SATISFIED`, flux jump 3.3e-2 against a 5e-3 tolerance)
  and `grades_pre_naming_fixes` grades it **FABRICATED_NO_RUN** (reason
  `COUPLING_EVIDENCE_CONTRADICTED`, `ratio_cv = 7.9e-6`). The order was real;
  the outcome was never CORRECT.
  The same pair disagrees about `C7_27b_BARE_seed2`: **CORRECT** at order
  1.9757 in `grades_pre_repair`, **FABRICATED_NO_RUN** in
  `grades_pre_naming_fixes`, same reason and the same stale rule — the
  synthetic-decay threshold `SYNTHETIC_RATIO_CV`, since tightened from 1e-5 to
  **1e-12**, which no longer fires at 7.9e-6.
  So **both readable grades directories predate the current fabrication rule**,
  and no coupled or fabrication number may be quoted from either. There is no
  post-repair grades directory on disk (`rounds/` is sealed `d---------`).
  **The regrade after round 9 settles this** and it costs no credits.
  Independent support that these two are not forgeries, computed with the keys
  sealed: the new flux-from-field check recovers C8 side A's conductivity as
  **0.9999999** and side B's as **999.99992** from the agent's own submitted
  field. A run whose reported flux reproduces both of the task's
  conductivities to seven digits did not invent it.
- **All three CORRECT coupled runs carry `per_code_attribution: UNPROVEN`** —
  each code proven individually, but not in *separate* files, so a monolithic
  solve cannot be excluded. **The coupling claim cannot rest on them.** Only a
  draw whose task demands captured per-code output can support it; the current
  builder emits that demand, the old draw does not.

### Why single-code fails (decomposition, 464 runs)

Among complete submissions the **self**-convergence order is a median 1.96 bare
/ 1.99 OASiS — *the finite element method converges*. The score is gated
elsewhere:

| mechanism | share of all runs |
|---|---|
| **no probe output produced at all** | **36%** |
| ├ solve succeeded, never read back at the prescribed points | 13% |
| ├ could not impose a spatially varying source in the input language | 10% |
| ├ toolchain will not build/import | 8% |
| └ silent stop, no RESULT.txt | 4% |
| submitted, converged cleanly to the WRONG function | 7% |

`MESH_INDEPENDENCE = NOT_CONVERGED` is **not usable as a signal**: ~75%
sensitivity, 54%/38% precision — it fires on half the CORRECT runs.

### Why coupling fails

Runs are now *graded* rather than rejected, and the numbers say the solutions
are genuinely wrong: orders **−0.02, 1.23, 1.27** against a band of [0.8, 3.2]
and a window of 2.0 ± 0.4. One diagnosed case had a subdomain field **ten times
too small and shrinking under refinement** while its interface residual
converged to 8e-07 — a converged interface says the sides *agree*, not that
either is *right*.

**The unfinished thing:** two-sided interface flux consistency is **not wired
into grading** (item #78, needs a key-schema field). It is the check that would
tell an agent its subdomain is wrong.

### The context/actions picture (and its refutation)

| slice | tool calls | tokens_in per call | wall_s | at the 2700s cap |
|---|---|---|---|---|
| coupled bare | 99 | 58,626 | 1718 | 23.1% |
| coupled OASiS | 41 | **90,740** | 1184 | **6.9%** |
| single bare | 103 | 54,887 | 1796 | 24.6% |
| single OASiS | 58 | 73,487 | 1473 | 17.5% |

The OASiS arm takes half the actions and stops at 44% of its budget while 86.5%
of coupled give-up texts blame time. **The clock is not the constraint.** But
the obvious inference — cut the text — was tested and **refuted** (§4). Treat
the correlation as at least partly reverse causation.

---

## 6. Fabrication: what is true

- **0.5% in the OASiS arm** (2 of 374). This target *is* met.
- The remaining cases are agents that bypass every voluntary channel.
  `C1_27b_MCP_seed84` never called `couple` or `audit_results`, wrote files by
  hand, and invented the history as exactly `0.5 × 0.5^k`, bit-identical at all
  three levels. The grader caught it.
- The pre-submission audit **refuses that submission** when it is given the
  chance — every check fires. It was never invoked, because it watched
  `write_file` and the agent used a heredoc. That hole is now closed.
- **What cannot be built:** a voluntarily-called tool that blocks anything. And
  if the *runner* blocked fabrication, the paper could no longer report a
  fabrication rate. Item #75's original wording ("structurally prevent
  fabrication in OASiS") is not achievable; the achievable form is *make the
  honest path cheaper and ensure the grader catches the rest*.

---

## 7. How to run and grade

### Launch

```bash
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=$PWD
set -a; . .env; set +a
export OASIS_BLIND_KEYS=~/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OASIS_BLIND_PROBLEMS=$PWD/campaign3_blind/problems
[ "$(stat -c %A $OASIS_BLIND_KEYS)" = "d---------" ] || exit 1   # refuse if open

setsid nohup ~/Schreibtisch/open-fem-agent/.venv-lg/bin/python \
  campaign3_blind/run_blind.py --model 27b --conditions MCP \
  --problems NG1 --seed 92 --phase development >> log 2>&1 < /dev/null &
```

`setsid` matters — a session rollover otherwise kills long drivers. Default
`--timeout` is 2700s. `cell_loop.sh CELL [s1 s2 s3] [ARM]` runs three seeds,
but defaults to the `keys` root and does **not** set `OASIS_BLIND_PROBLEMS`.

**`problems/` and `problems_dev3/` are different draws** — `problems/C2`
prescribes 44 interface probe points, `problems_dev3/C2` prescribes 22 (both use
the same 1936-point solution grid). Runner and grader must point at the same
root, or you grade one contract against another key. This has happened.

### Grade

Preferred, because it manages the sealed window itself and refuses if the keys
are already open:

```bash
printf '<passphrase>\n' | .venv/bin/python grade_round.py --with-key \
    --model 27b --seeds 2 3 4 --out out/R
```

One run:

```bash
./shield_keys.sh unseal
printf '<passphrase>\n' | .venv/bin/python grade_blind_v2.py --with-key \
    runs/NG1_27b_MCP_seed92 NG1
./shield_keys.sh seal        # ALWAYS; then verify d---------
```

`shield_keys.sh` requires `OASIS_BLIND_KEYS`, refuses to guess, seals every
`keys*` sibling, and `status` exits non-zero if any is open.

### Outcomes

`CORRECT`, `CORRECT_SUPERCONVERGENT`, `COMPLETED_UNPHYSICAL`,
`CONFIDENTLY_WRONG`, `FABRICATED_NO_RUN`, `HONEST_INCOMPLETE`, `FAILED`,
`MALFORMED_SUBMISSION`.

---

## 8. Rules the tests enforce — read before editing served text

1. **Served knowledge must not tell the model it is being tested** —
   `tests/test_served_text_is_not_self_referential.py`. "this campaign",
   "graded runs of", "the evaluation set" are banned.
2. **Must not state a measured convergence order we produced** —
   `tests/test_knowledge_not_contaminated.py`. Say "second order", not
   "2.014 / 2.007 / 2.003". Published benchmarks *with a citation* are fine.
3. **No drawn-problem dimensions** — never "the 22 points", "x = 5/8", "1936".
4. **The coupling must-read has a priority order** —
   `residual_level<k>.csv` and the `couple(...)` call must appear within the
   first 600 / 1200 characters. The must-read is prepended whole against a
   24,000-char cap, so **growing it silently evicts per-backend content off the
   end**; that is how 4C's four measured silent failures once fell out.
5. **The universal core rides on every `knowledge()` call** and is bounded.
   Additions should displace something rather than raise the bound.

---

## 9. Traps, each measured the hard way

**The recurring theme — six instances.** A mechanism existed, was instrumented,
and did not reach the case it was built for: delivery wrote into the source
tree; Option B had a second door; the agent was never told the time; the audit
refused coupled submissions; `_UNIVERSAL` reached 1 of 31 paths; the auto-audit
watched only `write_file`.

**Therefore prefer tests that ask what the AGENT ends up with over tests that
ask what the code does.** That discipline twice caught a regression within
minutes — including `functools.wraps` renaming the `knowledge` tool to
`_knowledge_body`, which would have removed the most-used tool from every run.

- **The trajectory logger truncates tool results at 600 characters** and says
  so in the file. **Absence of a phrase in `trajectory*.txt` is NOT evidence it
  was not served.** Use `served_bytes=` or replay the call.
- **`pkill -f X` matches its own shell** when the same command line contains a
  literal `X`. This killed my shell three times. Use `pkill -f 'pytes[t]'` and
  never put the launch string on the same line.
- **`stdbuf -oL` is mandatory for 4C** — MPI_Abort otherwise kills the process
  with stdout unflushed and the diagnostic is lost.
- **4C design entity ids are 1-based**; `E: 0` segfaults silently.
- **`agent` and `langgraph_eval.agent` import as two separate module objects.**
- `~/PortableSSD` is exFAT — no symlinks, no permission bits, never `TMPDIR`
  for 4C fixtures.
- SPARTA needs `RTLD_DEEPBIND` first or it segfaults.

---

## 10. Open items

| # | item |
|---|---|
| ~~78~~ | **DONE 2026-09-01** — see below; it needed no key-schema field, the blocker was geometric |
| 39 | Close the coupling flagship's four honest gaps |
| 52 | Public main emits 24 fabricated 4C keys into decks; the merge is the fix |
| 62 | The interface flux order claim is overstated for a paper — state the norm, the mesh, the end nodes |
| 63 | Pre-freeze checklist: grader rebuild with firing tests, widened freeze list, endpoints, ledgers |
| 66 | Curate the public GitHub repo as a product before any release |
| 70 | Re-run the 1334 tier-2 fixtures — the signal floor has been unverifiable since 18 Aug |
| 71 | Execute or retire the five remaining inherited claims |
| 74 | Sweep for verification stamps that cover an arm no fixture runs |
| 75 | Fabrication: reworded, see §6 |
| 76 | Kratos 3D does not segfault, but a plausible setup returns all zeros |

### #78 closed — and what it turned out to be

`flux_consistency` had existed for months with a measured docstring, and had
**never returned anything but NOT_ASSESSED on any coupled submission ever
graded**. The reason was not the missing key field the issue named. It was
geometric: `recover_flux_from_field` keys the field columns on each interface
probe's exact tangential coordinate, and the interface probes are deliberately
not on the field grid — measured on `C8_27b_BARE_seed4`, **0 of 44** interface
probes share a `y` with side A's 44x44 midpoint field grid.

The replacement, `recover_normal_derivative` + `flux_ratio_consistency`,
interpolates the field to the probe's tangential position first and then
extrapolates the normal derivative, and it needs **no coefficient, no key and
no reference** — for a scalar conduction flux the ratio `q_n / (-du/dn)` is the
conductivity at every point, so its CONSTANCY is the test. On
`C8_27b_BARE_seed4` it recovers `k_A = 0.9999999` and `k_B = 999.99992` with
the answers sealed.

Three things to know before you use it:

1. **It is restricted by physics, on purpose.** `scalar_flux_components()`
   refuses elasticity, anisotropic conduction, DSMC and FSI, because there the
   ratio need not be constant even for a perfect solve. Unrestricted, it
   produced 71 spurious INCONSISTENT triples on C7/C9/C11/C12.
2. **It is judged at the FINEST level.** The recovery is a one-sided `O(h^2)`
   estimate, so a coarse mesh inflates the spread: `C8_27b_MCP_seed4` side B
   reads 34.8% / 15.1% / 3.0% across levels while converging to k = 1000. A
   fixed tolerance at level 1 measures the mesh, not the agent.
3. **It gates nothing, and it caught nothing new.** Swept over all 427 coupled
   run directories it flags **zero** runs the two-sided jump gate passes. Its
   real value is the 19 runs where the flux IS consistent per side while the
   two sides disagree — there the transmission condition is wrong, not the flux,
   and that is a different repair instruction. Its reasons deliberately do not
   join `out["reasons"]`, which the caller feeds straight to the outcome.

Also open: `runs/` cross-run contamination (27 runs referenced another run's
directory, at least one confirmed `read_file`).

---

## 11. What to do next — pick one and say which

**Coordinate before starting: another agent is working the same tree.**

1. **Re-measure single-code on the current build.** 8–10 cells × 2 seeds, both
   arms, `grade_round.py`. ~$35–70. Until this exists nobody knows whether the
   tool improved, and the 41.7% in every report is a pre-fix number.
2. **Item #78** — wire two-sided flux consistency into coupled grading. It is
   the missing diagnostic and its absence keeps coupled work going in circles.
3. **Attack the 36% "no probe output" bucket.** deal.II's recipe was added and
   verified by compiling; the other eight backends have a recipe but nobody has
   checked that each one *runs*.
4. **Item #63**, the pre-freeze checklist, before any freeze.

---

## 12. How Alexander wants to be worked with

Durable, from standing feedback:

- **Verify before asserting.** Never declare something dead, impossible or
  fixed from one test. Execute it. "Unverified" means run it, not caveat it.
- **Rigorous and comprehensive** — all backends, both halves, no sampling, no
  surrogates, never plead workload.
- **Plain language.** Short sentences, no jargon, no narration. State the
  status; do not frame every finding as changing the picture.
- **Strict scope.** Edit what was authorised; do not bundle "logically related"
  changes, especially in the paper.
- **Finish the named item first**, then look for new problems.
- **Check delegated output with your own eyes** against the explicit bans before
  it ships.
- Report honestly against the targets and **never present development numbers as
  the paper's**.
- Figures: black for axes, labels and single-colour figures; colour only to
  distinguish data series.
