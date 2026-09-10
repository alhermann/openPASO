"""Natural task language must select the method-specific backend recipe."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.registry import get_backend, load_all_backends  # noqa: E402
from tools.consolidated import _fuzzy_match_physics  # noqa: E402


def _match(solver: str, query: str) -> str:
    load_all_backends()
    backend = get_backend(solver)
    assert backend is not None
    return _fuzzy_match_physics(backend, query)


def test_dune_sipg_advection_diffusion_does_not_route_to_reaction_diffusion():
    assert _match(
        "dune", "advection diffusion") == "dg_advection_diffusion"
    assert _match(
        "dune", "steady advection-diffusion SIPG") == \
        "dg_advection_diffusion"


def test_fourc_crank_nicolson_heat_selects_the_transient_mms_recipe():
    assert _match(
        "fourc", "transient heat conduction") == "thermo_transient_mms"
    assert _match(
        "fourc", "Crank-Nicolson transient heat") == "thermo_transient_mms"


def test_fenics_taylor_hood_phrases_do_not_route_to_primal_elasticity():
    for query in ("nearly incompressible elasticity",
                  "Taylor-Hood elasticity",
                  "mixed displacement pressure elasticity"):
        assert _match("fenics", query) == \
            "nearly_incompressible_elasticity"


def test_agent_is_told_not_to_erase_method_qualifiers():
    """The guidance lives in the PRODUCT (the server's own instructions, which
    the harness puts in front of the model verbatim) and in the coupled
    must-read -- not in harness prose of its own."""
    from core.instructions import INSTRUCTIONS
    from tools.consolidated import _COUPLING_MUST_READ
    for phrase in ("nearly incompressible Taylor-Hood elasticity",
                   "steady SIPG advection-diffusion",
                   "Crank-Nicolson transient heat"):
        assert phrase in INSTRUCTIONS
    assert "history_path" in _COUPLING_MUST_READ[:1500]
    assert "never retype" in _COUPLING_MUST_READ[:1500]
