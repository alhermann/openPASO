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
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10--3.13-64748B?style=flat-square" alt="Python 3.10 to 3.13"></a>
  <a href="https://doi.org/10.5281/zenodo.20543501"><img src="https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20543501-64748B?style=flat-square" alt="DOI"></a>
  <a href="#which-solvers-can-it-use"><img src="https://img.shields.io/badge/solvers-9-FF6B4A?style=flat-square" alt="9 solvers"></a>
</p>

<p align="center">
  <i>Describe a physics problem in ordinary words.<br>
  An AI model picks a solver, writes the input, runs it, and checks the answer.</i>
</p>

---

## What openPASO does

> [!TIP]
> Not a simulation person? Every word you need for the install steps is explained in
> one line under [Words used here](#words-used-here) at the bottom.

A **solver** is a program that computes how something physically behaves: how a metal
part bends, how heat spreads through a wall, how air flows around a wing. The method
most of these programs use is called the **finite element method** — it cuts the object
into many small pieces, called a **mesh**, and computes the answer piece by piece.

These programs are powerful and hard to use. Each one has its own input file format, its
own words for the same thing, and its own traps. Learning one takes weeks. Learning nine
takes years.

openPASO puts **nine such programs behind one door** and lets an AI model open it. You
write what you want in a normal sentence. The model chooses a suitable program, writes
the input file for it, starts it, reads the real output, and checks whether the answer is
right.

You do not need to know any input format. openPASO gives the model what it needs: what
each setting means **in the version you have installed**, which settings break, what a
specific error message really means, and how to check a result.

> [!NOTE]
> openPASO checks whether the **numbers are computed correctly** — whether they stop
> changing as the mesh gets finer, and whether the solver really ran. It does not check
> whether your model describes reality. Choosing the right physics is still your work.

---

## Install

> [!IMPORTANT]
> **openPASO is not useful on its own.** It is the part that drives the solvers; the
> thinking is done by an AI model, which you bring. So before you install, know which of
> these two you will have — the install itself is the same either way, but it is a
> ten-minute install and you should not reach the end and find out you have neither.

There are two ways to use openPASO. **Pick one.**

|  | Option A — an AI app | Option B — your own API key |
|---|---|---|
| **You need** | Claude Code, Claude Desktop or Cursor | an account at openrouter.ai |
| **Extra cost** | none beyond your subscription | you pay for what you use |
| **Choice of model** | whatever the app offers | any model on OpenRouter |
| **Good for** | trying it out, everyday work | scripting, experiments, cheap models |

Not sure whether your app works? If you have **Claude Code**, type `claude mcp list` in a
terminal. If the command exists, use Option A.

Now install. Do this whichever way you picked.

You need **Python 3.10 to 3.13** and **at least one** solver. You do not need all nine —
openPASO tells you what is missing and how to get it.

Check before you start:

```bash
python3 --version    # 3.10, 3.11 or 3.12 → go straight on. 3.13 → read the box below.
cc --version         # Linux/macOS only, and only for 3.13. "not found" → see the box.
```

> [!IMPORTANT]
> **On Python 3.13 you need a C compiler, and the install takes about two minutes
> longer.** openPASO keeps numpy below version 2, because the preCICE coupling library
> requires that, and no ready-made numpy package exists for 3.13 below version 2 — so
> `pip` has to compile it from source.
>
> - Debian or Ubuntu: `sudo apt install build-essential`
> - macOS: `xcode-select --install`
> - Fedora or RHEL: `sudo dnf install gcc gcc-c++ make`
> - Windows: there is no `cc`. Either install
>   [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/),
>   choosing the "Desktop development with C++" workload, and open a new terminal —
>   **or, much simpler, install Python 3.12** from
>   <https://www.python.org/downloads/> and ignore this box.
>
> Without a compiler the install stops on numpy with `Unknown compiler(s)` and
> `metadata-generation-failed`.
>
> Python 3.14 and newer are untested. Use 3.12 if you can choose.
>
> **If your Python is older than 3.10 or newer than 3.13**, install a supported one
> beside it — they live side by side without conflict. Get it from
> <https://www.python.org/downloads/>, or with conda:
> `conda create -n paso python=3.12 && conda activate paso`. Then use that `python3`
> for the `venv` step below.

```bash
git clone https://github.com/alhermann/openPASO.git
cd openPASO
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install scikit-fem            # the easiest solver to start with
```

Then install the agent packages. (They are only used by one of the two ways described
below, but installing them now costs nothing and saves you a decision you cannot make
yet.)

```bash
pip install -r langgraph_eval/requirements-langgraph.txt
```

### Check that it works

One command. It needs no API key, no AI app and no network:

```bash
python check_install.py
```

It prints which of the nine solvers openPASO can actually see on your machine, and for
each one it cannot see, the command that installs it. If you installed `scikit-fem`
above, it must appear with a tick:

```
✔ Python 3.12 — supported.
✔ openPASO imports, and its tools are registered.

Solvers openPASO can see on this machine — 1 of 9:

  ✔ scikit-fem
      scikit-fem 12.0.2 at /home/you/openPASO/.venv/bin/python
  ✘ NGSolve
      not installed — to get it:  python -m pip install ngsolve
  ...

✔ Install is working: 1 solver(s) ready.
```

**One tick in that list is enough to go on.** You do not need all nine.

If it says `openPASO itself does not import`, your virtual environment is not active:
run `source .venv/bin/activate` and try again. If it says `No solver is usable`, run
`pip install scikit-fem` and check again.

> [!NOTE]
> **Windows:** the commands above are for Linux and macOS. On Windows use
> `python -m venv .venv` and then `.venv\Scripts\activate`. A few other commands differ
> throughout this page: `copy` instead of `cp`, `dir` instead of `ls`, `set` instead of
> `export`, and `cd` (Command Prompt) or `pwd` (PowerShell) to print the current folder.
> openPASO is developed and tested on Linux. It should work on Windows, but that is not
> something we measure, and the solvers that build from source (4C, SPARTA, deal.II) are
> the least likely to.

### Which solvers can it use?

| Solver | What it is good at | How to install |
|---|---|---|
| **scikit-fem** | pure Python, full control over the assembly | `pip` — seconds |
| **NGSolve** | electromagnetics, acoustics, high-order elements | `pip` — seconds |
| **Kratos Multiphysics** | structures, fluids, coupled problems, particles | `pip install KratosMultiphysics-all` |
| **DUNE-fem** | discontinuous Galerkin, adaptive meshes | `pip` — minutes |
| **FEniCSx (dolfinx)** | fast prototyping, fluid flow | `conda` — minutes |
| **deal.II** | adaptive refinement, very large parallel runs | system package |
| **FEBio** | biomechanics, soft tissue, muscle | download a binary |
| **4C Multiphysics** | fluid–structure interaction, contact, beams, particles | build from source (hours) |
| **SPARTA** | rarefied gas, particle method, experimental | build from source (hours) |

<details>
<summary><b>Install commands, solver by solver</b></summary>

Install the `pip` ones into the **same** virtual environment you made above
(`.venv`, the one you activated).

```bash
pip install ngsolve scikit-fem meshio
```

`meshio` is optional. It reads and writes many mesh file formats, which several
examples use.

```bash
# Kratos. Use this package. It works on every system.
pip install KratosMultiphysics-all

# Why not plain "KratosMultiphysics": its 10.4.x wheels are labelled for an
# older system C library than they really need, so on an older system they
# install without any error and then fail to import. "-all" picks a build that
# runs. On a new system (ldd --version | head -1 shows 2.32 or higher at the
# end of the line) plain KratosMultiphysics also works, but there is no reason
# to prefer it.

# DUNE-fem: from PyPI. mpi4py is a hidden requirement; without it the first
# import stops with "Please run pip install mpi4py before rerunning your Dune script."
pip install dune-fem mpi4py
```

FEniCSx does **not** go into `.venv`. It needs its own conda environment, and openPASO
finds it there by itself. (deal.II, FEBio, 4C and SPARTA live outside `.venv` too — they
are not Python packages at all.)

```bash
# needs conda: https://docs.conda.io/projects/miniconda/
conda create -n fenics -c conda-forge fenics-dolfinx
```

Real and complex numbers are **separate** FEniCSx environments, not a switch you flip
at run time. Most people only need the real one.

```bash
# deal.II on Ubuntu/Debian
sudo apt install libdeal.ii-dev
```

Ubuntu 20.04 ships deal.II 9.1.1, which is old enough to miss many current functions.
Check yours with `grep DEAL_II_PACKAGE_VERSION /usr/include/deal.II/base/config.h`.
9.3 or newer is fine. Below 9.3, openPASO will still use it, but some
deal.II examples will not compile — build a newer one from source, or use one of
the other solvers instead.
If you build it yourself, use `-DCMAKE_BUILD_TYPE=DebugRelease` — a `Release` build
removes every internal check, so mistakes fail silently instead of telling you what
went wrong.

```bash
# FEBio: download the binary from https://febio.org/downloads/ then
export FEBIO_BINARY=/path/to/febio4      # the program file itself, not its folder
```

Any `export` line only lasts until you close the terminal. To keep it, put the same
line at the end of `~/.bashrc` (or `~/.zshrc`) and open a new terminal.

**Conda environments** are found automatically when their name contains `fenics`,
`dolfinx` or `dune`. For any other name, point openPASO at the interpreter yourself:

```bash
export FENICS_PYTHON=/path/to/env/bin/python
export DUNE_PYTHON=/path/to/env/bin/python
```

**4C and SPARTA** are built from source and take hours. They have their own
instructions, and you do not need them to start:

- 4C: <https://github.com/4C-multiphysics/4C> (build guide in its documentation)
- SPARTA: <https://sparta.github.io/>

Afterwards tell openPASO where they are: `export FOURC_BINARY=/path/to/4C` and
`export SPARTA_BINARY=/path/to/spa_serial`.

**If a solver will not install**, ask openPASO. Once it is connected (next section),
write to the AI in plain words:

> How do I install 4C on Ubuntu? Use the knowledge tool.

It answers with the route that works, the exact first-run error messages, and which
settings it checks rather than trusts.

</details>

---

## Start here

You picked A or B at the top of [Install](#install). Here is each one in full.

### Option A — you already pay for an AI app

This costs nothing extra. The app brings the AI model. openPASO brings the solvers.

> **MCP** (Model Context Protocol) is the standard plug that connects tools to AI apps.
> openPASO is such a tool. Your app only needs to support MCP.

**Claude Code** — run this once, from the openPASO folder:

```bash
claude mcp add openpaso -s user \
  -e PYTHONPATH="$PWD/src" \
  -e VIRTUAL_ENV="$PWD/.venv" \
  -e PYVISTA_OFF_SCREEN=true \
  -- "$PWD/.venv/bin/python" -m server
```

`$PWD` means "the folder I am in right now", so it fills in the full path for you.
Three details matter, and each one fails silently if you get it wrong:

- **`-e` must come before the `--`.** Everything after `--` is the command itself. An
  `-e` placed after it is handed to Python as an argument and ignored.
- **The paths must be full paths.** `PYTHONPATH=src` looks right and does not work: your
  AI app starts openPASO from its own folder, not from this one, so a relative path
  points at nothing.
- **`-s user`** makes openPASO available in every folder. Without it the default is
  "this folder only", and openPASO is simply absent everywhere else, with no message.

Check that it worked:

```bash
claude mcp list          # openpaso should be listed, and ✔ Connected
```

If Claude Code was already running, restart it so it picks up the new server.

If instead of `✔ Connected` you see **`⏸ Pending approval`**, you used `-s project` or
`-s local` rather than `-s user`. Those scopes write the server into the current folder
and ask you to approve it: start `claude` in that folder and accept, or remove it with
`claude mcp remove openpaso` and run the command above again exactly as written.

**Claude Desktop** — first get your full path. In the openPASO folder run:

```bash
pwd        # prints something like /home/you/openPASO — copy it
```

Then open Settings → Developer → Edit Config and add the block below.
**`/path/to/openPASO` appears three times. Replace all three.**

```json
{
  "mcpServers": {
    "openpaso": {
      "command": "/path/to/openPASO/.venv/bin/python",
      "args": ["-m", "server"],
      "env": {
        "PYTHONPATH": "/path/to/openPASO/src",
        "VIRTUAL_ENV": "/path/to/openPASO/.venv",
        "PYVISTA_OFF_SCREEN": "true"
      }
    }
  }
}
```

> [!WARNING]
> **On Windows two things change, and both fail silently if you skip them.**
> The Python program is at `.venv\Scripts\python.exe`, not `.venv/bin/python`.
> And **every backslash must be written twice** inside a JSON file: `C:\Users\...`
> is not valid JSON and the app will ignore the whole file without a word.
>
> ```json
> {
>   "mcpServers": {
>     "openpaso": {
>       "command": "C:\\Users\\you\\openPASO\\.venv\\Scripts\\python.exe",
>       "args": ["-m", "server"],
>       "env": {
>         "PYTHONPATH": "C:\\Users\\you\\openPASO\\src",
>         "VIRTUAL_ENV": "C:\\Users\\you\\openPASO\\.venv",
>         "PYVISTA_OFF_SCREEN": "true"
>       }
>     }
>   }
> }
> ```
>
> To get your own path, run `cd` in Command Prompt (or `pwd` in PowerShell) inside
> the openPASO folder, then double every backslash when you paste it.

**If the file is empty**, paste the whole block above.
**If the file already has something in it**, do not paste a second `{ ... }` — that makes
the file invalid and the app ignores it without saying so. Add only the
`"openpaso": { ... }` part, inside the `"mcpServers": { ... }` that is already there, and
put a comma after the entry before it.
**If the file has something in it but no `"mcpServers"` at all**, add the whole
`"mcpServers": { ... }` part as a new entry beside what is already there, and put a
comma after the entry before it.

Before you save, search **this block** for `/path/to/` — if it still finds anything, you
missed one. That is the most common way this goes wrong, and it fails silently.

**Then quit Claude Desktop completely and start it again.** Closing the window is not
enough. Until you do this, openPASO is simply not there, with no message.

**Check it worked:** in Claude Desktop, open the tools menu in the message box.
`openpaso` should be in the list. If it is not:

1. You did not fully quit and restart the app.
2. The file is not valid JSON. Paste it into any online JSON checker.
3. The `command` path does not exist. In a terminal run
   `ls /path/to/openPASO/.venv/bin/python` — if that errors, the path is wrong.
   On Windows: `dir C:\Users\you\openPASO\.venv\Scripts\python.exe`.

On Windows, search the block for `\you\` instead of `/path/to/` — same check, same
reason.

These are full paths for the same reason as above: the app starts openPASO from its own
folder. `VIRTUAL_ENV` is needed because some solvers build helper code at run time and
pick their Python from the environment; without it they can pick the wrong one.

> [!IMPORTANT]
> **If you set any solver variable with `export`, it must also go inside the `env`
> block.** Your AI app does not start openPASO from your terminal, so it never sees your
> `export` line or your `~/.bashrc`. Add each one as another `"NAME": "value"` line:
>
> ```json
>       "env": {
>         "PYTHONPATH": "/path/to/openPASO/src",
>         "VIRTUAL_ENV": "/path/to/openPASO/.venv",
>         "PYVISTA_OFF_SCREEN": "true",
>         "FEBIO_BINARY": "/path/to/febio4",
>         "FENICS_PYTHON": "/path/to/fenics-env/bin/python"
>       }
> ```
>
> Without this the solver installs correctly, works in your terminal, and openPASO still
> reports it as missing. The same applies to `FOURC_BINARY`, `SPARTA_BINARY` and
> `DUNE_PYTHON`. To see what openPASO itself finds, run `python check_install.py` — but
> run it with the same variables set, or it will not see them either.

**Cursor, Windsurf, or any other MCP app** — use the command
`/path/to/openPASO/.venv/bin/python`, the arguments `-m server`, and **all three**
settings from the `env` block above. Then quit the app completely and start it again.

Then simply ask, inside the app:

> Solve the Poisson equation on a unit square with a known exact solution,
> and check that the error falls at the expected rate.

### Option B — you have an OpenRouter key

**OpenRouter** gives you one key for many AI models, and you pay for what you use.
Three steps, all from the openPASO folder:

```bash
# 1. Copy the settings file
cp .env.example .env

# 2. Open .env in any editor. Paste your key after OPENROUTER_API_KEY=
#    Get a key at https://openrouter.ai/keys
#    A model is already chosen for you, so you can leave the rest alone.

# 3. Ask for a simulation. The virtual environment must be active:
#    if your prompt does not show (.venv), run  source .venv/bin/activate
python run_agent.py "Solve the Poisson equation on the unit square with scikit-fem, and report the maximum value."
```

Your key lives in the file **`.env`** in the openPASO folder and nowhere else. That file
stays on your computer; git never uploads it.

> [!TIP]
> Start with a cheap model. `.env.example` lists several and already selects one. To use
> a different one, either change `OPENPASO_MODEL` in `.env`, or add
> `--model some/model-id` to the command.

### Did it work?

`run_agent.py` prints everything as it happens:

- `→ toolname(...)` — the model is calling a solver tool
- `← toolname: N line(s)` — the tool answered
- plain text — the model talking to you
- `done` on the last line — the run finished

Files the solvers produce are written to the **`simulation_outputs/`** folder inside
openPASO. (You can change that with `OPENPASO_OUTPUT_DIR` in `.env`.)

**If something went wrong**, `run_agent.py` says so in one sentence and tells you what to
do. The most common cases:

| Message | What to do |
|---|---|
| `there is no .env file yet` | Run `cp .env.example .env` from the openPASO folder, then paste your key into it |
| `OPENROUTER_API_KEY is empty in ...` | The `.env` file exists but the key line is blank. Paste your key after `OPENROUTER_API_KEY=` |
| `OpenRouter rejected the key` | The key is wrong — check it at <https://openrouter.ai/keys> |
| `your OpenRouter account is out of credit` | Add credit at <https://openrouter.ai/credits> |
| `the chosen model cannot use tools` | Pick a model marked with tool support on <https://openrouter.ai/models> |
| `a Python package is missing` | Run `pip install -r langgraph_eval/requirements-langgraph.txt` |
| `the openPASO server has no Python to run in` | Your `.venv` is missing. Redo the [Install](#install) steps |

**If you used Option A and the app does not see openPASO:**

| What you see | What to do |
|---|---|
| `claude mcp list` shows openpaso as failed or not connected | Run `claude mcp get openpaso`. `PYTHONPATH` must be a full path starting with `/`. If it shows only `src`, run `claude mcp remove openpaso -s user` and add it again from inside the openPASO folder |
| openpaso is missing when you work in another folder | It was added for one folder only. Remove it and add it again with `-s user` |
| the app lists it but every tool call fails | The `command` path must point at `.venv/bin/python` inside openPASO, not at your system Python |


---

## What a run looks like

This is a real transcript of Option B, so you can see the shape of a session before you
start one. If you chose Option A you never type this command — you ask the same thing
inside your app and see the same tool calls there.

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

---

## What you can simulate

- **Classic equations** — Poisson, heat conduction, diffusion
- **Solids** — bending and stretching, large deformation, plasticity, contact, vibration frequencies
- **Fluids** — slow flow and fast flow: cavity flow, flow past an obstacle, flow in a channel
- **Transport over time** — heat that changes with time, chemical reactions that spread
- **Electromagnetics** — wave problems, cavity resonances, magnets
- **Particle methods** — materials modelled as many interacting particles instead of a mesh
- **Two solvers on one problem** — split a body in two, give each half to a different
  program, and repeat until the two halves agree at the boundary between them
- **Meshes** — generated with Gmsh: an L-shape, a plate with a hole, a channel, or your own
- **Accuracy studies** — refine the mesh step by step and measure how the error shrinks

---

## How openPASO checks the answer

A wrong result that looks right is worse than no result. Four things guard against it.

1. **A known exact answer to compare against.** For many problems openPASO can build the
   problem backwards from an answer it already knows, so the error is a number and not an
   opinion.
2. **The error must shrink at the right speed.** Make the mesh twice as fine and the error
   must fall by the amount the mathematics predicts. If it does not, the run failed, no
   matter how good the picture looks.
3. **Proof that the solver really ran.** openPASO reads the solver's own console output
   and its own data files. A number with no run behind it is reported as exactly that.
4. **A second AI instance reviews the setup** before the run starts: units, boundary
   conditions, mesh fineness, material values. The server keeps the record of that review
   and looks it up instead of believing a claim.

When two solvers work on one problem together and they never come to agree, that is
reported as a failure. It is never presented as a result.

---

## The tools the model gets

The server offers about two dozen tools. These 14 are the ones you will see it use:

| Tool | What it does |
|---|---|
| `discover` | lists the solvers, what is installed, what each one can do |
| `prepare_simulation` | knowledge, real examples and a starting file in one call — always the first step |
| `run_simulation` | runs the Python solvers (FEniCSx, NGSolve, scikit-fem, DUNE-fem) |
| `run_with_generator` | writes the input file and runs the compiled solvers (4C, deal.II, Kratos) |
| `verify_mesh_independence` | refines the mesh and reports whether the answer stopped changing |
| `knowledge` | physics, known traps, materials, coupling, and comparisons between solvers |
| `examples` | real test files from the solvers' own test suites |
| `couple` | runs one problem across two solvers until they agree |
| `couple_precice` | the same, through the preCICE coupling library |
| `transfer_field` | moves a result from one solver's output into another solver |
| `generate_mesh` | builds a mesh with Gmsh |
| `visualize` | statistics, plots, automatic checks |
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

- **Server** — `src/server.py`. Solvers load as plugins through `src/core/registry.py`.
- **Backends** — one package per solver under `src/backends/`, each with a list of the
  physics it supports and writers for the input files of the compiled solvers.
- **Knowledge** — known traps per solver, element lists, and settings read from the
  version you actually installed, not from a manual.
- **Coupling** — `src/core/coupling_driver.py`. Each side is a black box that reads
  `imports.json` and writes `exports.json` under a fixed contract. openPASO checks the
  contract, repeats the exchange until both sides agree, and reports agreement or failure.
- **preCICE bridge** — `src/core/precice_config.py` and the `couple_precice` tool.

---

## Words used here

| Word | Meaning in one line |
|---|---|
| **solver** | a program that computes the physics |
| **backend** | one such program, as openPASO sees it |
| **finite element method** | cutting an object into small pieces and solving piece by piece |
| **mesh** | the set of small pieces |
| **element** | one piece of the mesh |
| **boundary condition** | what you fix at the edges: a temperature, a force, a fixed wall |
| **refine the mesh** | use more and smaller pieces, for a more accurate answer |
| **mesh independence** | the answer stops changing when you refine further — a good sign |
| **convergence rate / order** | how fast the error shrinks as you refine |
| **coupling** | two solvers working on one problem, exchanging values at their shared boundary |
| **verification** | checking that the numbers are computed correctly |
| **validation** | checking that the model matches the real world — **not** done here |
| **MCP** | the standard plug that connects tools to AI apps |
| **git** | the tool that downloads this project's files (`git clone`) |
| **pip** | the tool that installs Python packages |
| **compiler** | a program that turns source code into something your machine can run. Some packages need one |
| **venv** | a private Python folder for one project's packages |
| **`pip install -e .`** | install the project in this folder (`.`) so your edits take effect straight away (`-e`) |
| **PYTHONPATH** | tells Python which folder to find openPASO's code in |
| **VIRTUAL_ENV** | tells programs which private Python folder to use |
| **PYVISTA_OFF_SCREEN** | draws pictures without opening a window, so it works on a machine with no screen |
| **langgraph** | the library that lets an AI model use tools in a loop; only Option B needs it |
| **Gmsh** | the program openPASO uses to build meshes |
| **preCICE** | a separate library for coupling two solvers; an alternative to openPASO's own `couple` |
| **wheel** | a ready-built Python package that `pip` downloads instead of compiling |
| **YAML**, **XML** | two other text formats for settings; some solvers use these instead of JSON |
| **MPI** / **mpi4py** | the standard way programs split work across many processors |
| **ldd** | a command that prints which system libraries your machine has |
| **DSMC** | a particle method for gas so thin that the usual flow equations stop working |
| **JSON** | a text format for settings. Every bracket, quote and comma must match, or the whole file is ignored |
| **binary** | a program you can run, already compiled — you do not build it yourself |
| **interpreter** | the `python` program itself; several can be installed side by side |
| **prompt** (terminal) | the text your terminal shows before you type, such as `$`. Not the same as the question you ask an AI |
| **fork** | a copy of a project developed separately. This repository is one; install from here, cite the original |
| **conda** | another tool for private Python folders, like venv. FEniCSx needs it |
| **agent** | an AI model that can use tools by itself, not only write text |
| **server** | the background program the AI app talks to; openPASO is one |
| **environment variable** | a setting your terminal passes to a program. `export NAME=value` sets one, and it is forgotten when you close the terminal |
| **API key** | a password that lets a program use a paid AI service |
| **OpenRouter** | a service giving one key access to many AI models |
| **tool support** | whether a model is able to call tools. Not every model is |
| **Poisson equation** | a standard textbook problem used to check that a solver works |
| **unit square** | the square from 0 to 1 in both directions, the usual test shape |

---

## Contributing

Contributions are welcome. One rule stands above the rest: **every improvement must help
all simulations**, not one example.

Good contributions are known solver traps, element lists, new solvers and new coupling
templates. Not welcome are collections of numbers for one benchmark, or templates built
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
