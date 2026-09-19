"""The server's own instructions told the model about eight codes while nine shipped.

The instructions string is the FIRST thing a connected model reads, and it is
where the model learns which codes exist at all. It has said "across 8
independent FEM codes" and listed eight since before SPARTA was added, so the
ninth backend was invisible to every agent that did not go looking: the catalog
registers it, `discover(query='list')` reports it, the coupling contracts exist
for it, and the sentence that introduces the server does not mention it.

That is not cosmetic here. One coupled development problem couples FEniCSx with
SPARTA, and an agent reading "the catalog ships generators for 8 backends"
followed by a list without SPARTA has been told, in the server's own voice, that
its task names a code the server does not have.

MEASURED at the time of writing: the registry ships nine (dealii, dune, febio,
fenics, fourc, kratos, ngsolve, skfem, sparta); the instructions named eight and
omitted sparta.

THE CLAIM IS DERIVED, NOT COUNTED BY HAND. A number written into prose drifts
away from the thing it counts the moment the thing changes, and nothing tells
anybody -- which is exactly what happened. This test derives both the count and
the names from the registry, so the next backend to be added fails here rather
than quietly going unmentioned.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from core.registry import list_backends, load_all_backends       # noqa: E402

# How each backend is written in the instructions' own list. Only the spelling
# lives here; which backends must appear comes from the registry.
DISPLAY = {
    "fenics": "FEniCSx", "dealii": "deal.II", "fourc": "4C",
    "ngsolve": "NGSolve", "skfem": "scikit-fem", "kratos": "Kratos",
    "dune": "DUNE-fem", "febio": "FEBio", "sparta": "SPARTA",
}


def _shipped() -> set:
    load_all_backends()
    return {b["name"] for b in list_backends()}


def _instructions() -> str:
    from core.instructions import INSTRUCTIONS
    return INSTRUCTIONS


def test_every_shipped_backend_is_named_in_the_instructions():
    text = _instructions()
    missing = sorted(n for n in _shipped()
                     if DISPLAY.get(n, n) not in text)
    assert not missing, (
        "the server's instructions never mention " + ", ".join(missing)
        + ", so a model is told these codes do not exist. The registry ships "
        + str(sorted(_shipped())))


def test_the_number_in_the_prose_is_the_number_that_ships():
    text = _instructions()
    n = len(_shipped())
    claims = [int(m) for m in re.findall(r"(?:across|ships generators for)\s+(\d+)\s+(?:independent\s+)?(?:FEM\s+)?(?:codes|backends)", text)]
    assert claims, "the instructions no longer state how many codes there are"
    wrong = [c for c in claims if c != n]
    assert not wrong, (
        f"the instructions claim {wrong} code(s) while the registry ships {n}. "
        "A count written into prose drifts the moment a backend is added; "
        "state it from the registry or keep this test green by hand.")


def test_the_list_has_one_entry_per_shipped_backend():
    """A bullet each, so the count and the list cannot disagree with each other."""
    text = _instructions()
    bullets = re.findall(r"^- \*\*(.+?)\*\*", text, re.M)
    assert len(bullets) >= len(_shipped()), (
        f"{len(bullets)} backend bullets for {len(_shipped())} shipped backends: "
        + str(bullets))
