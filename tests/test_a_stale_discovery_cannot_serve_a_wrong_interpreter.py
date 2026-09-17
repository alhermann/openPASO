"""A served path is the one the backend would use, never a stale guess recorded months ago.

Measured on another checkout: data/discovered_config.json (from May) recorded the server's own venv as
DUNE's "location" -- autodiscovery records the interpreter it probed with -- so {DUNE_PYTHON} in served
text became a Python that cannot import dune.fem. A clean checkout had the opposite problem: no record,
so "<set DUNE_PYTHON>" for a DUNE the backend finds by itself.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import core.host_paths as hp  # noqa: E402


def test_a_recorded_interpreter_is_never_served(monkeypatch):
    monkeypatch.delenv("DUNE_PYTHON", raising=False)
    monkeypatch.setattr(hp, "_from_backend", lambda token: None)          # the backend finds nothing
    monkeypatch.setattr(hp, "_from_discovery", lambda key: sys.executable)  # the stale record
    assert sys.executable not in hp.resolve("{DUNE_PYTHON}")
    assert "set DUNE_PYTHON" in hp.resolve("{DUNE_PYTHON}")


def test_the_backend_finder_wins_over_a_record(monkeypatch, tmp_path):
    real = tmp_path / "febio4"
    real.write_text("")
    monkeypatch.delenv("FEBIO_BINARY", raising=False)
    monkeypatch.setattr(hp, "_from_backend", lambda token: str(real) if token == "{FEBIO_BINARY}" else None)
    monkeypatch.setattr(hp, "_from_discovery", lambda key: "/nonexistent/febio4")
    assert hp.resolve("{FEBIO_BINARY}") == str(real)


def test_a_recorded_binary_that_no_longer_exists_is_not_served(monkeypatch):
    monkeypatch.delenv("FEBIO_BINARY", raising=False)
    monkeypatch.setattr(hp, "_from_backend", lambda token: None)
    monkeypatch.setattr(hp, "_from_discovery", lambda key: "/nonexistent/febio4")
    monkeypatch.setattr(hp.shutil, "which", lambda name: None)
    assert "set FEBIO_BINARY" in hp.resolve("{FEBIO_BINARY}")


def test_the_environment_variable_still_wins(monkeypatch):
    monkeypatch.setenv("DUNE_PYTHON", "/explicit/python")
    assert hp.resolve("{DUNE_PYTHON}") == "/explicit/python"


def test_loading_the_backends_does_not_promote_a_recorded_interpreter(monkeypatch, tmp_path):
    """Found by Copilot (Hereon PR #57): load_all_backends copied the recorded location into
    FENICS_PYTHON / DUNE_PYTHON, so the stale value looked user-set and bypassed every check."""
    import core.autodiscovery as ad
    from core.registry import load_all_backends
    monkeypatch.delenv("DUNE_PYTHON", raising=False)
    monkeypatch.delenv("FENICS_PYTHON", raising=False)
    monkeypatch.delenv("FOURC_BINARY", raising=False)
    stale = {"backends": {"dune": {"location": sys.executable},
                          "fenics": {"location": sys.executable},
                          "fourc": {"location": str(tmp_path / "gone" / "4C")}}}
    monkeypatch.setattr(ad, "load_discovered_config", lambda: stale)
    load_all_backends()
    import os
    assert "DUNE_PYTHON" not in os.environ and "FENICS_PYTHON" not in os.environ
    assert "FOURC_BINARY" not in os.environ, "a recorded binary that no longer exists was promoted"


def test_a_verified_server_interpreter_is_a_valid_answer(monkeypatch):
    """Also from Copilot: the FEniCSx finder returns the server's own Python only after dolfinx imports."""
    monkeypatch.delenv("FENICS_PYTHON", raising=False)
    import importlib
    real = importlib.import_module
    class _Fake:
        @staticmethod
        def _find_fenics_python():
            return sys.executable
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a: _Fake if name == "backends.fenics.backend" else real(name, *a))
    monkeypatch.setattr(hp, "_backend_works", lambda name: True)
    assert hp._from_backend("{FENICS_PYTHON}") == sys.executable


def test_a_located_path_the_backend_calls_unavailable_is_not_served(monkeypatch, tmp_path):
    """Copilot, Hereon PR #57: a finder that only locates must not be trusted on its own."""
    monkeypatch.delenv("FEBIO_BINARY", raising=False)
    fake = tmp_path / "febio4"
    fake.write_text("")
    import importlib
    real = importlib.import_module
    class _Fake:
        @staticmethod
        def _find_febio_binary():
            return str(fake)
    monkeypatch.setattr(importlib, "import_module",
                        lambda name, *a: _Fake if name == "backends.febio.backend" else real(name, *a))
    monkeypatch.setattr(hp, "_backend_works", lambda name: False)
    assert hp._from_backend("{FEBIO_BINARY}") is None
