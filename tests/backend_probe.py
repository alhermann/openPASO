"""Is this solver really here?

Imported by tests that need a specific solver. It is a module rather than part
of conftest.py because pytest loads conftest itself, under a name a test file
cannot import -- `from conftest import ...` fails at collection. Fixtures and
hooks belong in conftest; this is a library.

WHY THREE STATES AND NOT TWO
---------------------------
Many tests need a specific solver. Skipping them when it is absent is right:
the README tells a newcomer to install ONE solver, and a suite that then
reports thirty failures reads as "this project is broken".

But a skip that fires on a missing import, or on the words "No module named"
appearing in an error, is worse than no skip at all, because it also swallows
the case that matters: a solver that IS installed and does NOT work. Then the
suite is green and the product is broken.

So this asks openPASO's own check_availability, which already separates
NOT_INSTALLED from MISCONFIGURED, and then -- where openPASO ships a smoke test
-- RUNS it. A real Poisson solve, not a version string. Three answers:

    absent    not on this machine                  -> the test skips
    working   here, and it solved something        -> the test runs
    broken    here, and it did NOT solve           -> the test FAILS

"broken" never skips. A half-installed solver must be louder than a missing
one, not quieter.
"""
from __future__ import annotations

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


def _one_line(detail: str, limit: int = 160) -> str:
    """A skip reason is a one-line summary. Probes hand back whole tracebacks."""
    text = " ".join(str(detail).split())
    for marker in ("Traceback (most recent call last):", "File \"<string>\""):
        cut = text.find(marker)
        if cut > 0:
            text = text[:cut].rstrip(" :")
            break
    return text if len(text) <= limit else text[:limit - 1] + "…"


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
        return "absent", f"{name} is not installed ({_one_line(detail)})"
    if status is BackendStatus.MISCONFIGURED:
        # openPASO's own word for "it is here and it is wrong". Never a skip.
        # The full detail is kept here: a broken backend is worth the noise.
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


def need_backend(name: str) -> None:
    """Skip if the solver is absent; FAIL if it is present and broken."""
    state, detail = backend_state(name)
    if state == "broken":
        pytest.fail(detail)
    if state == "absent":
        pytest.skip(f"{detail} — install it to run this test")


# There is deliberately no decorator form. A decorator has to return a mark, and
# no pytest mark means "fail this test now" -- xfail(strict=True) would turn a
# genuinely failing test into an expected failure, which is the opposite of what
# the broken state is for. need_backend() inside the test body has one behaviour
# per state and no silent mode.
