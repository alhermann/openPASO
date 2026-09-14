"""Tool output must land where the CALLER works, not inside the install.

Every output directory used to be computed from the package's own location,
so one shared directory served every caller. An evaluation cell that followed
the documented workflow wrote its results outside its own sandbox, and the
grader — which reads the sandbox — recorded it as having produced nothing.
Because these are the openPASO tools, the cost fell entirely on the arm under
test: 8 of 14 coupled openPASO runs in one round went through that path.

The first repair fixed tools/consolidated.py and missed tools/coupling.py,
tools/simulation.py, tools/mesh_generation.py and tools/benchmark.py, which
each carried their own copy. This test exists so the next copy cannot hide:
it walks the tool layer for the pattern itself.
"""
import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"

# module -> (attribute holding the directory, env var that must control it)
CASES = [
    ("tools.consolidated", "_OUTPUT_DIR", "OPENPASO_OUTPUT_DIR"),
    ("tools.consolidated", "_COUPLING_DIR", "OPENPASO_COUPLING_DIR"),
    ("tools.coupling", "_COUPLING_DIR", "OPENPASO_COUPLING_DIR"),
    ("tools.simulation", "_OUTPUT_DIR", "OPENPASO_OUTPUT_DIR"),
    ("tools.mesh_generation", "_MESH_OUTPUT_DIR", "OPENPASO_MESH_DIR"),
    ("tools.benchmark", "_BENCHMARK_DIR", "OPENPASO_BENCHMARK_DIR"),
]


@pytest.mark.parametrize("module,attr,var", CASES)
def test_directory_follows_its_environment_variable(module, attr, var, tmp_path):
    """Each module must resolve its output dir from the caller's environment.

    Run in a subprocess: these are module-level constants, so the environment
    has to be set before import to mean anything.
    """
    target = tmp_path / "caller_dir"
    code = (f"import sys; sys.path.insert(0, {str(SRC)!r});"
            f"import importlib;"
            f"m = importlib.import_module({module!r});"
            f"print(getattr(m, {attr!r}))")
    env = dict(os.environ, **{var: str(target)})
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    resolved = out.stdout.strip().splitlines()[-1]
    assert resolved == str(target), (
        f"{module}.{attr} ignored {var}: resolved to {resolved}")


def test_no_tool_module_computes_an_output_dir_from_the_install():
    """The pattern itself, wherever it reappears."""
    bad = []
    pattern = re.compile(
        r"parents\[\d\]\s*/\s*\"(simulation_outputs|meshes|benchmarks)\"")
    for f in (SRC / "tools").glob("*.py"):
        for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                bad.append(f"{f.name}:{i}: {line.strip()}")
    assert not bad, (
        "output directories must come from core.output_paths.output_dir so "
        "they follow the caller:\n  " + "\n  ".join(bad))


def test_resolver_rejects_an_unknown_kind():
    sys.path.insert(0, str(SRC))
    from core.output_paths import output_dir
    with pytest.raises(KeyError):
        output_dir("not_a_real_output_kind")
