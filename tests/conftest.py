"""
Pytest configuration: ensure all solver backends are discoverable.

Environment variables (set these for your machine, or they'll be auto-detected):
  FOURC_ROOT   — path to 4C source tree
  FOURC_BINARY — path to 4C binary
  LD_LIBRARY_PATH — include 4C dependency libs if needed
"""
import os
import shutil

# 4C Multiphysics — auto-detect if not set
if not os.environ.get("FOURC_ROOT"):
    # Try common locations
    for candidate in [os.path.expanduser("~/4C"), "/opt/4C", "/usr/local/4C"]:
        if os.path.isdir(candidate):
            os.environ["FOURC_ROOT"] = candidate
            break

if not os.environ.get("FOURC_BINARY"):
    # Try finding in FOURC_ROOT/build or on PATH
    root = os.environ.get("FOURC_ROOT", "")
    for d in ["build", "build/release", "build/debug"]:
        candidate = os.path.join(root, d, "4C")
        if os.path.isfile(candidate):
            os.environ["FOURC_BINARY"] = candidate
            break
    else:
        p = shutil.which("4C")
        if p:
            os.environ["FOURC_BINARY"] = p

# 4C runtime dependencies (if present)
ld_path = os.environ.get("LD_LIBRARY_PATH", "")
for dep_lib in ["/opt/4C-dependencies/lib", os.path.join(os.environ.get("FOURC_ROOT", ""), "lib")]:
    if os.path.isdir(dep_lib) and dep_lib not in ld_path:
        os.environ["LD_LIBRARY_PATH"] = f"{dep_lib}:{ld_path}" if ld_path else dep_lib
        break

# PyVista off-screen rendering
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")


# ─────────────────────────────────────────────────────────────────────────
# IS THIS SOLVER REALLY HERE?
#
# Many tests need a specific solver. Skipping them when it is absent is right:
# the README tells a newcomer to install ONE solver, and a suite that then
# reports thirty failures reads as "this project is broken".
#
# But a skip that fires on a missing import string is worse than no skip at
# all, because it hides the case that actually matters: a solver that IS
# installed and does NOT work. So this asks openPASO's own probe whether the
# backend is present, and then, where openPASO ships a smoke test, RUNS it —
# a real Poisson solve, not a version string. Three outcomes:
#
#   absent   the solver is not on this machine        -> the test skips
#   working  it is here and it solved something       -> the test runs
#   broken   it is here and it did NOT solve          -> the test FAILS
#
# "broken" never skips. That is the whole point: a half-installed solver must
# be louder than a missing one, not quieter.
# ─────────────────────────────────────────────────────────────────────────
import functools
import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Backends for which openPASO ships a smoke test that actually solves.
_SMOKE = {
    "ngsolve": "smoke_ngsolve",
    "skfem": "smoke_skfem",
    "kratos": "smoke_kratos",
    "dune": "smoke_dune",
    "fourc": "smoke_fourc",
}


@functools.lru_cache(maxsize=None)
def backend_state(name: str) -> tuple[str, str]:
    """('absent'|'working'|'broken', detail). Probed once per session."""
    try:
        from core.backend import BackendStatus
        from core.registry import get_backend, load_all_backends
        load_all_backends()
        backend = get_backend(name)
    except Exception as exc:                     # noqa: BLE001
        return "absent", f"openPASO could not load its registry: {exc}"

    if backend is None:
        return "absent", f"{name} is not registered on this machine"

    try:
        status, detail = backend.check_availability()
    except Exception as exc:                     # noqa: BLE001
        return "broken", (f"{name}.check_availability() raised "
                          f"{type(exc).__name__}: {exc}")

    if status is BackendStatus.NOT_INSTALLED:
        return "absent", f"{name} is not installed ({detail})"
    if status is BackendStatus.MISCONFIGURED:
        # openPASO's own word for "it is here and it is wrong". Never a skip.
        return "broken", f"{name} is installed but misconfigured: {detail}"

    smoke_name = _SMOKE.get(name)
    if smoke_name is None:
        # No smoke test ships for this backend. check_availability already
        # imported it or ran its binary, which is weaker but not nothing.
        return "working", f"{name} available ({detail}); no smoke test ships for it"

    try:
        from core import smoke_tests
        result = getattr(smoke_tests, smoke_name)()
    except Exception as exc:                     # noqa: BLE001
        return "broken", (f"{name} smoke test raised "
                          f"{type(exc).__name__}: {exc}")
    if getattr(result, "passed", False):
        return "working", f"{name} solved its smoke problem"
    return "broken", (f"{name} reports itself available and then did NOT solve "
                      f"its smoke problem: {getattr(result, 'error', result)}")


def requires_backend(name: str):
    """Skip if the solver is absent; FAIL if it is present and broken."""
    state, detail = backend_state(name)
    if state == "broken":
        return pytest.mark.xfail(reason=detail, strict=True, run=True)
    return pytest.mark.skipif(state == "absent",
                              reason=f"{detail} — install it to run this test")


def need_backend(name: str) -> None:
    """Same rule, inside a test body or setUp."""
    state, detail = backend_state(name)
    if state == "broken":
        pytest.fail(detail)
    if state == "absent":
        pytest.skip(f"{detail} — install it to run this test")
