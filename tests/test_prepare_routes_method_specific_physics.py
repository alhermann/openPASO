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
    # THE TWO MUST ARRIVE TOGETHER, NEAR THE TOP -- NOT WITHIN AN EXACT
    # CHARACTER COUNT. This read _COUPLING_MUST_READ[:1500] for both. The text
    # grew by a character or two and "never retype" came to start at 1,499, so
    # the slice cut the phrase in half and the test failed while the
    # instruction was exactly where it belongs, one line under history_path.
    # The property is that the instruction accompanies the argument it is
    # about, early in the must-read; assert that instead of a byte offset.
    opening = _COUPLING_MUST_READ[:3000]
    at = opening.find("history_path")
    assert at >= 0, "history_path is no longer named in the opening must-read"
    assert "never retype" in opening[at:at + 400], (
        "the instruction not to retype the measured history no longer "
        "accompanies the history_path argument it is about")
