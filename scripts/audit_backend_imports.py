"""Cross-backend import-availability audit.

For each registered backend, identify the underlying library each
physics module depends on, then verify that the library is
importable in the environment that backend uses. Produces
``scripts/scan_results/backend_imports.json`` summarising
available vs unreachable physics.

Background (2026-06-01): empirical probing surfaced two
structural alignment bugs:

  1. ``supported_physics()`` is a static list — it does NOT check
     whether the underlying library imports. The DUNE backend
     reports 15 supported physics even though ``import dune.fem``
     fails with a libibverbs version mismatch in the available
     python interpreter.

  2. Kratos has 31 physics in supported_physics(), but ~20 of them
     depend on applications NOT installed in the repo .venv
     (DEMApplication, MPMApplication, IgaApplication, etc.).
     Calling ``prepare_simulation(solver='kratos', physics='dem')``
     would return a template that immediately fails at runtime
     with ImportError on first ``import KratosMultiphysics.DEMApplication``.

This script produces an honest reachable-vs-unreachable view
that future regression tests can compare against. Does NOT
modify backend registration today — that is a follow-up refactor
(blocked on user decision).

Usage:
    python3 scripts/audit_backend_imports.py
"""

from __future__ import annotations

import json
import pathlib
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()


# ───────────────────────────────────────────────────────────────
# Configuration: which python interpreter each backend uses, and
# which python-import string actually proves the library works
# for that backend × physics.
# ───────────────────────────────────────────────────────────────

# RESOLVED, NOT ASSUMED.
#
# These used to be fixed paths: the repo's own `.venv`, and conda envs named
# `ofa-fenicsx` / `ofa-dune` / `ofa-dealii`. Every one of them is wrong on at
# least one host this runs on. A git worktree has no `.venv`, and the dolfinx
# env here is called `fenics` while the DUNE one is `dune-fem-env`. The audit
# then recorded `core_importable: false` with `core_error: "interpreter not
# found at <path>"` — for backends that import perfectly well through the
# interpreter the rest of the tree already knows about. Kratos was reported
# unavailable on a host where it imports with 28 applications.
#
# That is not a wrong path, it is a wrong KIND of answer: an instrument that
# could not look, filing a negative that reads exactly like a measurement.
# Same defect as a fixture runner recording `skipped` because an env var was
# unset, and as an auditor that examines nothing and prints a pass.
#
# Resolution order matches scripts/run_tier2_fixtures.py, so the two agree
# about what a backend's interpreter IS: explicit env var, then plausible
# locations, then the running interpreter.
_ENV_VAR = {
    "venv":        "OPENPASO_PYTHON",
    "kratos":      "KRATOS_PYTHON",
    "ofa-fenicsx": "FENICS_PYTHON",
    "ofa-dune":    "DUNE_PYTHON",
}
_CANDIDATES = {
    "venv": [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(HOME / "Schreibtisch" / "open-fem-agent" / ".venv" / "bin" / "python"),
    ],
    "ofa-fenicsx": [str(HOME / "miniconda3" / "envs" / n / "bin" / "python")
                    for n in ("fenics", "ofa-fenicsx", "fenicsx", "dolfinx")],
    "ofa-dune": [str(HOME / "miniconda3" / "envs" / n / "bin" / "python")
                 for n in ("dune-fem-env", "ofa-dune", "dune311")],
    "ofa-dealii": [str(HOME / "miniconda3" / "envs" / n / "bin" / "python")
                   for n in ("ofa-dealii", "dealii")],
    # Kratos gets its OWN slot. It used to share the repo venv, whose Kratos
    # wheel fails on this host with `GLIBC_2.32 not found` — so the audit
    # reported Kratos unavailable while the interpreter the rest of the tree
    # uses for it imports 28 applications happily.
    "kratos": ["/mnt/kratos-tier2/kv/bin/python",
               str(REPO_ROOT / ".venv" / "bin" / "python"),
               str(HOME / "Schreibtisch" / "open-fem-agent" / ".venv" / "bin" / "python")],
}


def _resolve(slot: str) -> str | None:
    """The interpreter for a slot, or None when this host has none."""
    var = _ENV_VAR.get(slot)
    if var and os.environ.get(var) and Path(os.environ[var]).is_file():
        return os.environ[var]
    for cand in _CANDIDATES.get(slot, []):
        if Path(cand).is_file():
            return cand
    if slot == "system":
        return sys.executable
    return None


PYTHON_PATHS = {slot: _resolve(slot)
                for slot in ("venv", "kratos", "ofa-fenicsx", "ofa-dealii",
                             "ofa-dune", "system")}

# Which python interpreter to use for each backend.
BACKEND_PYTHON = {
    "kratos":  "kratos",
    "skfem":   "venv",
    "ngsolve": "venv",
    "fenics":  "ofa-fenicsx",
    "dune":    "ofa-dune",
    "dealii":  None,   # C++ compile path; check Debug build instead
    "fourc":   None,   # Binary path check, not python import
}

# Kratos physics → required application name (from the bulk-promotion
# audit). When a physics is not in this map, only the core
# KratosMultiphysics import is required.
KRATOS_PHYSICS_APP = {
    "linear_elasticity":      "StructuralMechanicsApplication",
    "structural_dynamics":    "StructuralMechanicsApplication",
    "plasticity":             "ConstitutiveLawsApplication",
    "fluid":                  "FluidDynamicsApplication",
    "fluid_biomedical":       "FluidDynamicsApplication",
    "contact":                "ContactStructuralMechanicsApplication",
    "heat":                   "ConvectionDiffusionApplication",
    "heat_transient":         "ConvectionDiffusionApplication",
    "dem":                    "DEMApplication",
    "thermal_dem":            "DEMApplication",
    "swimming_dem":           "DEMApplication",
    "fem_to_dem":             "FemToDemApplication",
    "dem_structures_coupling":"DEMApplication",
    "mpm":                    "MPMApplication",
    "geomechanics":           "GeoMechanicsApplication",
    "poromechanics":          "GeoMechanicsApplication",
    "pfem_fluid":             "PfemFluidDynamicsApplication",
    "pfem":                   "PfemFluidDynamicsApplication",
    "iga":                    "IgaApplication",
    "rans":                   "RANSApplication",
    "shallow_water":          "ShallowWaterApplication",
    "rom":                    "RomApplication",
    "shape_optimization":     "ShapeOptimizationApplication",
    "topology_optimization":  "TopologyOptimizationApplication",
    "compressible_potential": "CompressiblePotentialFlowApplication",
    "compressible_flow":      "CompressiblePotentialFlowApplication",
    "cosimulation":           "CoSimulationApplication",
    "chimera":                "ChimeraApplication",
    "fsi":                    "MappingApplication",
    "wind_engineering":       "FluidDynamicsApplication",
    "fluid_hydraulics":       "FluidDynamicsApplication",
    "optimization":           "OptimizationApplication",
    "fluid_dynamics":         "FluidDynamicsApplication",
}

# Special checks for non-python backends.
# Candidate lists, not single guesses, for the same reason as PYTHON_PATHS:
# the old single paths (`Schreibtisch/dealii-debug`, `Schreibtisch/4C-src`)
# exist on no host this has ever run on, so both backends were recorded absent
# while both are installed and in daily use.
NONPYTHON_CHECKS = {
    "dealii": [
        Path(os.environ["DEALII_DIR"]) / "lib" / "libdeal_II.so"
        if os.environ.get("DEALII_DIR") else None,
        HOME / "dealii" / "build" / "lib" / "libdeal_II.so",
        HOME / "Schreibtisch" / "dealii-debug" / "lib" / "libdeal_II.g.so",
    ],
    "fourc": [
        Path(os.environ["FOURC_BINARY"]) if os.environ.get("FOURC_BINARY") else None,
        HOME / "4C" / "build" / "4C",
        HOME / "Schreibtisch" / "4C-src" / "4C" / "build" / "4C",
    ],
}


def _can_run(python_path: str, import_stmt: str, timeout: int = 30) -> tuple[bool, str]:
    """Subprocess-check whether `import_stmt` succeeds in `python_path`.

    Returns (ok, diagnostic). diagnostic is empty on success and the
    first ~200 chars of the import error otherwise.
    """
    # NOT CHECKED is not the same answer as NOT IMPORTABLE. Returning False
    # here made "this host has no interpreter mapped for that backend" and
    # "that library is broken" the same record, and a reader cannot tell them
    # apart. None means the question was never put to the library.
    if not python_path or not Path(python_path).is_file():
        return None, (f"NOT CHECKED: no interpreter at "
                      f"{python_path or '<unmapped>'} — set the backend's "
                      f"*_PYTHON environment variable. This is not evidence "
                      f"that the library is unavailable.")
    try:
        r = subprocess.run(
            [python_path, "-c", import_stmt],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    except Exception as e:
        return False, f"subprocess error: {e}"
    if r.returncode == 0:
        return True, ""
    err = (r.stderr or r.stdout or "").strip()
    # Filter MPI noise lines (shmem warnings etc.)
    err_lines = [l for l in err.splitlines()
                 if "shmem" not in l and "create_and_attach" not in l]
    return False, "\n".join(err_lines)[:240]


def audit_kratos() -> dict:
    """Kratos: core import + each physics app."""
    py = PYTHON_PATHS["kratos"]
    core_ok, core_err = _can_run(py, "import KratosMultiphysics")
    if core_ok is not True:
        # core_ok is None when there is no interpreter to ask. That is NOT the
        # same record as "Kratos is broken here", and collapsing the two is how
        # this file previously reported a working install as unavailable.
        return {
            "interpreter": py,
            "core_importable": core_ok,          # None = not checked
            "core_error": core_err,
            "checked": core_ok is not None,
            "physics_available": [],
            "physics_unreachable": [],
        }
    available = ["StructuralMechanicsApplication",
                 "FluidDynamicsApplication",
                 "ConvectionDiffusionApplication"]  # checked separately below
    available_apps = set()
    for app in set(KRATOS_PHYSICS_APP.values()):
        ok, _ = _can_run(py, f"import KratosMultiphysics.{app}")
        if ok:
            available_apps.add(app)
    physics_available = []
    physics_unreachable = []
    for physics, app in KRATOS_PHYSICS_APP.items():
        if app in available_apps:
            physics_available.append({"physics": physics, "app": app})
        else:
            physics_unreachable.append({"physics": physics, "app": app})
    return {
        "interpreter": py,
        "core_importable": True,
        "core_error": "",
        "available_apps": sorted(available_apps),
        "physics_available": sorted(physics_available,
                                      key=lambda x: x["physics"]),
        "physics_unreachable": sorted(physics_unreachable,
                                        key=lambda x: x["physics"]),
    }


def audit_simple_backend(name: str, import_stmt: str) -> dict:
    py_key = BACKEND_PYTHON.get(name)
    py = PYTHON_PATHS.get(py_key) if py_key else None
    if py is None:
        return {"backend": name, "interpreter": None,
                "importable": None, "error": "no python interpreter mapped"}
    ok, err = _can_run(py, import_stmt)
    return {"backend": name, "interpreter": py,
            "importable": ok, "error": err}


def audit_nonpython_backend(name: str) -> dict:
    cands = [c for c in (NONPYTHON_CHECKS.get(name) or []) if c is not None]
    if not cands:
        return {"backend": name, "target": None, "available": None,
                "error": "no nonpython check configured"}
    for c in cands:
        if Path(c).exists():
            return {"backend": name, "target": str(c),
                    "available": True, "error": ""}
    # Every candidate missing is a real negative — the search DID happen — but
    # say where it looked, so a reader can tell "not installed" from "installed
    # somewhere this list does not know about".
    return {"backend": name, "target": None, "available": False,
            "searched": [str(c) for c in cands],
            "error": (f"not found in any of {len(cands)} candidate paths; set "
                      f"{'DEALII_DIR' if name == 'dealii' else 'FOURC_BINARY'} "
                      f"if it lives elsewhere")}


def main():
    report = {
        "_comment": "Backend import-availability audit. Surfaces "
                    "the gap between supported_physics() and what "
                    "the underlying library can actually do. Use "
                    "as data for follow-up registry-validation work.",
        "kratos":  audit_kratos(),
        "skfem":   audit_simple_backend("skfem", "import skfem"),
        "ngsolve": audit_simple_backend("ngsolve", "import ngsolve"),
        "fenics":  audit_simple_backend("fenics", "import dolfinx"),
        "dune":    audit_simple_backend("dune", "import dune.fem"),
        "dealii":  audit_nonpython_backend("dealii"),
        "fourc":   audit_nonpython_backend("fourc"),
    }

    # Headline summary line.
    summary = {}
    k = report["kratos"]
    summary["kratos_physics_available"] = len(k.get("physics_available", []))
    summary["kratos_physics_unreachable"] = len(k.get("physics_unreachable", []))
    for be in ("skfem", "ngsolve", "fenics", "dune"):
        summary[f"{be}_importable"] = bool(report[be].get("importable"))
    for be in ("dealii", "fourc"):
        summary[f"{be}_available"] = bool(report[be].get("available"))
    report["summary"] = summary

    # An optional argv path lets callers (the test suite) write elsewhere.
    # Without it, the tracked snapshot under scan_results/ was overwritten on
    # EVERY test run: a committed measurement silently replaced by whatever
    # this machine says today, so the file drifted with each `pytest` and a
    # stale environment could clobber a good record.
    import sys as _sys
    out = (pathlib.Path(_sys.argv[1]) if len(_sys.argv) > 1
           else REPO_ROOT / "scripts" / "scan_results" / "backend_imports.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"audit written: {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
