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
