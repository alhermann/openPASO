<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="logo/openPASO_koralle_dunkel.gif">
    <source media="(prefers-color-scheme: light)" srcset="logo/openPASO_graphit_transparent.gif">
    <img src="logo/openPASO_koralle_dunkel.gif" alt="openPASO" width="220">
  </picture>
</p>

<h1 align="center">openPASO</h1>

<p align="center"><b>open Platform for Agentic Simulation and Optimization</b></p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-FF6B4A?style=flat-square" alt="MIT licence"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-64748B?style=flat-square" alt="Python 3.10+"></a>
  <a href="https://doi.org/10.5281/zenodo.20543501"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20543501-64748B?style=flat-square" alt="DOI"></a>
  <a href="#which-solvers-can-it-use"><img src="https://img.shields.io/badge/solvers-9-FF6B4A?style=flat-square" alt="9 solvers"></a>
</p>

<p align="center">
  <i>Describe a physics problem in ordinary words.<br>
  An AI model picks a solver, writes the input, runs it, and checks the answer.</i>
</p>

---

## What openPASO does

Finite element software is powerful and hard to use. Every code has its own input
format, its own words for the same thing, and its own traps. Learning one takes weeks.
Learning nine takes years.

openPASO puts **nine simulation codes behind one door** and lets an AI model open it.
You write what you want in a normal sentence. The model chooses a suitable code, writes
the input file for it, starts the solver, reads the real output, and checks whether the
answer is right.

You do not need to know the input format. openPASO gives the model what it needs: what
each keyword means **in the version you have installed**, which settings break, what a
specific error message really means, and how to verify a result.

> [!NOTE]
> openPASO checks whether the **numbers are computed correctly** — the convergence order,
> the consistency, the evidence that the solver truly ran. It does not check whether your
> model describes reality. Choosing the right physics is still your work.

---

## See it work

```console
$ python run_agent.py "Use the discover tool to list available solvers, then answer in one sentence."

  model    deepseek/deepseek-v4.1-flash
  starting the openPASO server ...

  24 solver tools ready
────────────────────────────────────────────────────────────────────────
  → discover(query='list')
    ← discover: 1 line(s)

All nine backends are available on this install: 4C Multiphysics,
FEniCSx (dolfinx 0.10.0), deal.II 9.8.0-pre, FEBio, NGSolve 6.2.2604,
scikit-fem 12.0.1, Kratos Multiphysics 10.3, DUNE-fem, and SPARTA (DSMC).

────────────────────────────────────────────────────────────────────────
  done
```

---

## Start here

There are two ways to use openPASO. **Pick one.** Both need the steps in
[Install](#install) first.

### Option A — you already pay for an AI app

If you use **Claude Code**, **Claude Desktop**, **Cursor** or another app that supports
MCP, this costs you nothing extra. The app brings the AI model. openPASO brings the
solvers.

> MCP (Model Context Protocol) is the standard plug that connects tools to AI apps.
> openPASO is such a tool.

**Claude Code** — run this once, in the openPASO folder:

```bash
claude mcp add openpaso .venv/bin/python -- -m server \
  -e PYTHONPATH=src -e PYVISTA_OFF_SCREEN=true
```

**Claude Desktop** — open Settings → Developer → Edit Config, and add this.
Replace `/path/to/openPASO` with the real folder on your computer:

```json
{
  "mcpServers": {
    "openpaso": {
      "command": "/path/to/openPASO/.venv/bin/python",
      "args": ["-m", "server"],
      "cwd": "/path/to/openPASO/src",
      "env": {
        "PYTHONPATH": "/path/to/openPASO/src",
        "PYVISTA_OFF_SCREEN": "true"
      }
    }
  }
}
```

**Cursor, Windsurf, or any other MCP app** — use the same four values:
command `/path/to/openPASO/.venv/bin/python`, arguments `-m server`, working folder
`/path/to/openPASO/src`, and the two environment variables above.

Then simply ask, in the app:

> Solve the Poisson equation on a unit square with a known exact solution,
> and verify the convergence rate.

### Option B — you have an OpenRouter key

Use this if you do not have such an app, or if you want to choose the model yourself.
**OpenRouter** gives you one key for many AI models, and you pay for what you use.

**Three steps:**

```bash
# 1. Copy the settings template
cp .env.example .env

# 2. Open .env and paste your key after OPENROUTER_API_KEY=
#    Get a key at https://openrouter.ai/keys

# 3. Ask for a simulation
python run_agent.py "Solve -Δu = 1 on the unit square with u = 0 on the boundary
                     using scikit-fem, and report the maximum value."
```

Your key lives in the file **`.env`** in the openPASO folder, and nowhere else.
That file stays on your computer; git never uploads it.

> [!TIP]
> Start with a cheap model. `.env.example` lists a few good ones. You can change the
> model at any time by editing `OPENPASO_MODEL` in `.env`, or by adding
> `--model some/model-id` to the command.

### Which option should I choose?

| | Option A — an AI app | Option B — an OpenRouter key |
|---|---|---|
| **You need** | Claude Code, Claude Desktop, Cursor, … | an account at openrouter.ai |
| **Extra cost** | none beyond your subscription | you pay per use |
| **Choice of model** | whatever the app offers | any model on OpenRouter |
| **Good for** | trying it out, daily work | scripting, experiments, cheap models |

---

## Install

You need Python 3.10 or newer, and **at least one** simulation code. You do not need all
nine. openPASO tells you what is missing and how to get it.

```bash
git clone https://github.com/alhermann/openPASO.git
cd openPASO
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install scikit-fem              # the easiest solver to start with
```

Only for **Option B**, also install the agent packages:

```bash
pip install -r langgraph_eval/requirements-langgraph.txt
```

Check that it starts:

```bash
cd src && python -m server          # stops with Ctrl+C
```

### Which solvers can it use?

| Solver | What it is good at | How hard to install |
|---|---|---|
| **scikit-fem** | pure Python, assembly-level control | `pip` — seconds |
| **NGSolve** | Maxwell, Helmholtz, high-order, eigenvalues | `pip` — seconds |
| **Kratos Multiphysics** | structures, fluids, FSI, DEM, MPM | `pip` — seconds |
| **DUNE-fem** | DG methods, VEM, h/p-adaptivity | `pip` — minutes |
| **FEniCSx (dolfinx)** | fast prototyping, weak forms, Navier–Stokes | `conda` — minutes |
| **deal.II** | adaptive refinement, matrix-free, MPI + GPU | system package |
| **FEBio** | biomechanics, tissue, active contraction | binary download |
| **4C Multiphysics** | FSI, thermo-mechanics, contact, beams, particles | build from source |
| **SPARTA** | rarefied gas (DSMC particles), experimental | build from source |

<details>
<summary><b>Install commands for each solver</b></summary>

```bash
# pip-installable — into the SAME interpreter that runs openPASO
pip install ngsolve scikit-fem meshio

# Kratos: use the metapackage. Installing KratosMultiphysics alone lets pip take
# the newest wheel, and the 10.4.x wheels are tagged manylinux_2_28 while actually
# requiring GLIBC_2.32 — they install fine and then fail to import.
# Check your system with: ldd --version | head -1
pip install KratosMultiphysics-all

# DUNE-fem: from PyPI. mpi4py is an undeclared dependency; without it the first
# import stops with "Please run pip install mpi4py before rerunning your Dune script."
pip install dune-fem mpi4py

# FEniCSx: conda-forge is the supported route. Real and complex numbers are
# SEPARATE environments, not a runtime switch.
conda create -n fenics -c conda-forge fenics-dolfinx

# deal.II on Ubuntu/Debian. Note that 20.04 ships 9.1.1, which is old enough to
# miss many current functions. Check yours with:
#   grep DEAL_II_PACKAGE_VERSION /usr/include/deal.II/base/config.h
# Build from source with -DCMAKE_BUILD_TYPE=DebugRelease if you want assertion
# messages — a Release build removes every Assert.
sudo apt install libdeal.ii-dev

# FEBio: download the binary from https://febio.org/downloads/ and then
export FEBIO_BINARY=/path/to/febio4
```

**macOS and deal.II.** Install the official `deal.II.app` bundle and point
`DEAL_II_DIR` at its `Contents/Resources/Libraries`. If a build then fails inside
`<complex>` or `<cmath>`, that is an Xcode SDK header clash inside the bundle, not an
openPASO problem. Fix it with:

```bash
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk
```

**Conda environments** are found automatically when their name contains `fenics`,
`dolfinx` or `dune`. For any other layout, name the interpreter yourself — openPASO
uses it both to find the code and to run it:

```bash
export FENICS_PYTHON=/path/to/env/bin/python
export DUNE_PYTHON=/path/to/env/bin/python
```

**If a solver will not install**, ask openPASO itself. It answers with the route that
works on your system, the exact first-run error messages, and which environment
variables it checks rather than trusts:

```
knowledge(topic='install', solver='fourc')
```

</details>

---

## What you can simulate

- **Classic PDEs** — Poisson, heat conduction, diffusion
- **Solid mechanics** — linear elasticity, large deformation, plasticity, contact, eigenfrequencies
- **Flow** — Stokes and Navier–Stokes: lid-driven cavity, vortex shedding, channel flow
- **Transport** — transient heat, reaction–diffusion, convection
- **Electromagnetics** — Maxwell, cavity resonances, magnetostatics
- **Particle methods** — SPH, DEM, peridynamics
- **Two codes on one problem** — split a domain between two different solvers and iterate
  until they agree: thermo-mechanics, fluid–structure interaction, domain decomposition,
  even a continuum code coupled to a rarefied-gas particle code
- **Meshes** — Gmsh geometries: L-shape, plate with a hole, channel, or your own
- **Convergence studies** — refine the mesh step by step and measure the error

---

## How openPASO checks the answer

A wrong result that looks right is worse than no result. Four things guard against it.

1. **An exact solution to compare against.** For many problems openPASO can construct a
   source term whose exact answer is known, so the error is a number, not an opinion.
2. **The convergence order.** Refine the mesh and the error must fall at the rate the
   theory predicts. If it does not, the run failed, whatever the pictures look like.
3. **Evidence that the solver really ran.** openPASO reads the solver's own console
   output and its own data files. A number with no run behind it is reported as such.
4. **A second AI instance reviews the setup** before the run starts — units, boundary
   conditions, mesh resolution, material values. The server keeps the record of that
   review and looks it up instead of believing a claim.

When two codes are coupled, a coupling that does not converge is reported as a failure.
It is never presented as a result.

---

## The tools the model gets

| Tool | What it does |
|---|---|
| `discover` | lists the solvers, what is installed, what each can do |
| `prepare_simulation` | knowledge, real examples and a template in one call — always the first step |
| `run_simulation` | runs the Python solvers (FEniCSx, NGSolve, scikit-fem, DUNE-fem) |
| `run_with_generator` | writes the input file and runs the compiled solvers (4C, deal.II, Kratos) |
| `verify_mesh_independence` | refines the mesh and reports whether the answer stopped changing |
| `knowledge` | physics, pitfalls, materials, coupling, and comparisons between codes |
| `examples` | real test files from the solvers' own test suites |
| `couple` | runs one problem across two codes and iterates until they agree |
| `couple_precice` | the same through the preCICE coupling library |
| `transfer_field` | moves a field from one solver's output to another |
| `generate_mesh` | builds a mesh with Gmsh |
| `visualize` | field statistics, plots, automatic checks |
| `developer` | reads and changes a solver's own source code, then rebuilds it |
| `setup_backend` | install help for a solver that is missing |

---

## How it is built

```
You ──▶ AI model (an app, or your API key) ──▶ MCP ──▶ openPASO (src/server.py)
                                                            │
   ┌──────────┬──────────┬────────┬─────────┬────────┬──────┴───┬────────┬────────┐
FEniCSx    deal.II       4C    NGSolve   skfem    Kratos     DUNE    FEBio   SPARTA
(Python)    (C++)     (YAML)  (Python) (Python)  (JSON)   (Python)  (XML)  (native)
```

- **Server** — `src/server.py`, an MCP server over stdio. Backends load as plugins
  through `src/core/registry.py`.
- **Backends** — one package per code under `src/backends/`, each with a catalog of the
  physics it supports and input generators for the compiled codes.
- **Knowledge** — per-code pitfalls, element catalogs, API references read from the
  version you actually installed, and a layer that compares the codes with each other.
- **Coupling** — `src/core/coupling_driver.py`. Each side is a black box that reads
  `imports.json` and writes `exports.json` under a fixed contract. openPASO checks the
  contract, runs the fixed-point iteration with Aitken relaxation, and reports
  convergence or failure.
- **preCICE bridge** — `src/core/precice_config.py` and the `couple_precice` tool.

---

## Contributing

Contributions are welcome. One rule stands above the rest: **every improvement must help
all simulations**, not one example.

Good contributions are solver pitfalls, element catalogs, new backends and new coupling
generators. Not welcome are databases of parameters for one benchmark, or templates built
around one particular problem. Templates use placeholders, never the dimensions of a
specific case.

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Licence and citation

openPASO is released under the [MIT licence](LICENSE).

This repository is the **experimental fork**, where new capabilities are developed and
tested. The stable, citable release lives at
**<https://github.com/Hereon-InstituteMS/OASiS>** and is archived on Zenodo. If you use
this software in research, please cite the archived release:

```bibtex
@software{openpaso2026,
  title  = {openPASO: an open-source multi-physics and multi-code framework
            for verified computer simulations},
  author = {Hermann, Alexander and Shojaei, Arman and Scheider, Ingo and Cyron, Christian},
  year   = {2026},
  doi    = {10.5281/zenodo.20543501},
  url    = {https://github.com/Hereon-InstituteMS/OASiS}
}
```

<p align="center">
  <sub>Helmholtz-Zentrum Hereon · Institute of Materials Mechanics &nbsp;·&nbsp;
  Hamburg University of Technology</sub>
</p>
