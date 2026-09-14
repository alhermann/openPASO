"""A skip must never be able to hide a solver that is installed and broken.

Skipping a test because its solver is missing is right: the README tells a
newcomer to install one solver, and a suite that then reports thirty failures
reads as "this project is broken". But the easy way to write that skip -- catch
ImportError, skip -- also swallows the case that actually matters, where the
solver IS there and does NOT work. Then the suite is green and the product is
broken, which is the worst of both.

So backend_probe.backend_state has three answers, not two, and this file holds it to
them. It is the gate on the gate: without it, nobody has ever watched the
"broken" branch fire.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backend_probe as C  # noqa: E402


def test_every_backend_answers_with_one_of_the_three_states():
    for name in ("skfem", "ngsolve", "kratos", "dune", "fourc",
                 "dealii", "febio", "fenics", "sparta"):
        state, detail = C.backend_state(name)
        assert state in ("absent", "working", "broken"), (name, state)
        assert detail, f"{name} gave no reason for '{state}'"


def test_a_broken_backend_fails_and_does_not_skip(monkeypatch):
    """The whole point. A solver that is present and does not work must FAIL."""
    monkeypatch.setattr(C, "backend_state",
                        lambda name: ("broken", "pretend it is misconfigured"))
    with pytest.raises(BaseException) as caught:   # pytest outcomes are BaseException
        C.need_backend("skfem")
    # pytest.skip raises Skipped; pytest.fail raises Failed. It must be Failed.
    assert caught.typename != "Skipped", (
        "a broken backend was skipped — that is exactly the hole this guards")
    assert "pretend it is misconfigured" in str(caught.value)


def test_an_absent_backend_skips(monkeypatch):
    monkeypatch.setattr(C, "backend_state",
                        lambda name: ("absent", "not installed here"))
    with pytest.raises(BaseException) as caught:   # pytest outcomes are BaseException
        C.need_backend("skfem")
    assert caught.typename == "Skipped"


def test_a_working_backend_does_nothing(monkeypatch):
    monkeypatch.setattr(C, "backend_state",
                        lambda name: ("working", "it solved its smoke problem"))
    C.need_backend("skfem")          # must not raise


def test_the_check_is_not_satisfied_by_an_import_alone():
    """dune is the case that proved this necessary.

    dune reported itself available and its smoke test said "No module named
    'dune'", because the smoke test ran in openPASO's own interpreter while
    DUNE lives in its own environment by design. An availability check that
    stopped at 'the backend says yes' would have called that working.
    """
    import inspect
    source = inspect.getsource(C.backend_state)
    assert "smoke_tests" in source, (
        "backend_state must run the shipped smoke test, not trust a status")
    assert "MISCONFIGURED" in source, (
        "backend_state must honour openPASO's own 'installed but wrong' state")
