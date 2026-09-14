"""Every code the coupling corpus serves a participant for carries its measured deciding facts.

scikit-fem had none — the only backend in the corpus without them — while three coupled development
problems put it on one side. Measured 2026-09-14 on skfem 12.0.1: two of three step-trial workers died
on `laplace.assemble(basis=basis)` (TypeError: missing positional 'ubasis') and on
`basis.dofs(facet_indices=...)` (TypeError: 'Dofs' object is not callable), both of which a served fact
names. This pins the coverage rule, not one text."""
from __future__ import annotations
import sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

PARTICIPANTS = ROOT / "data" / "coupling_participants"
# the backend key each participant_<key>.py belongs to
SERVED_CODES = sorted({p.stem.removeprefix("participant_").split("_")[0]
                       for p in PARTICIPANTS.glob("participant_*.py")})


def test_the_corpus_serves_participants_for_the_codes_we_think_it_does():
    assert set(SERVED_CODES) >= {"fenics", "fourc", "dealii", "ngsolve", "skfem",
                                 "kratos", "dune", "febio", "sparta"}


@pytest.mark.parametrize("code", SERVED_CODES)
def test_a_served_code_carries_deciding_facts(code):
    from tools.consolidated import _DECIDING_FACTS
    if code in ("fsi", "tsi", "precice"):          # role/physics prefixes, not backends
        pytest.skip("not a backend key")
    assert _DECIDING_FACTS.get(code), (
        f"{code} serves a coupling participant but has no deciding facts: a coupled side "
        f"needs the code's measured API traps exactly as much as a single-code run does")


@pytest.mark.parametrize("alias,canonical", [("scikit-fem", "skfem"), ("scikitfem", "skfem"),
                                             ("deal.ii", "dealii"), ("4c", "fourc"),
                                             ("dolfinx", "fenics"), ("dune-fem", "dune")])
def test_the_name_a_task_uses_finds_the_facts(alias, canonical):
    from tools.consolidated import _DECIDING_FACTS
    assert _DECIDING_FACTS.get(alias) == _DECIDING_FACTS[canonical]


def test_the_skfem_facts_name_the_two_measured_deaths():
    from tools.consolidated import _DECIDING_FACTS
    f = _DECIDING_FACTS["skfem"]
    assert "assemble(basis=basis)" in f and "'ubasis'" in f
    assert "'Dofs' object is not callable" in f and "get_dofs(" in f
    assert "boundaries_only=False" in f          # an interface inside one mesh
    assert "probes(" in f and "interpolator(" in f   # off-node evaluation, which every task demands


def test_a_coupled_skfem_reply_carries_them():
    from core.registry import load_all_backends
    load_all_backends()
    from tools import consolidated as C
    C._MUST_READ_STATE["served"] = False
    first = C._get_coupling_knowledge("skfem", "", "")
    pointer = C._get_coupling_knowledge("scikit-fem", "", "")
    for out in (first, pointer):
        assert "'Dofs' object is not callable" in out
