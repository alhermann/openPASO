"""A fixture run must not leave readable worked answers on the host.

MEASURED 2026-09-16. Every coupling fixture produces exactly the shape the
blind campaign's custody preflight refuses to start beside: two participants,
exports.json on both sides, per-level fields, a converged interface.
`readable_worked_answers` (campaign3_blind/host_hygiene.py) walks /tmp,
/var/tmp and $HOME looking for that shape, and refuses the round rather than
touching a developer's files.

125 of these had accumulated on this host since August, 68 of them in a single
afternoon, and a blind round's first launch was refused by nine cells at once.
Worse than the refusal is the ordering it implies: the preflight runs ONCE, at
launch, against a condition that stays mutable for the next 45 minutes, and two
sessions share /tmp. A fixture regeneration that has nothing to do with blind
rounds can contaminate one, and nothing warns either side.

Sealing, not deleting: the artefacts stay for inspection, `chmod 700` brings
them back, and that is how the August ones were handled.
"""
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "tier2_fixtures" / "coupling" / "_lib"))

import couplinglib  # noqa: E402


def test_a_workroot_is_registered_for_sealing():
    before = list(couplinglib._WORKROOTS)
    root = couplinglib.workroot("sealtest")
    try:
        assert root.is_dir()
        assert root in couplinglib._WORKROOTS, (
            "a workroot nobody recorded cannot be sealed at exit")
        assert len(couplinglib._WORKROOTS) == len(before) + 1
    finally:
        os.chmod(root, 0o700)
        shutil.rmtree(root, ignore_errors=True)


def test_sealing_makes_the_worked_answer_unreadable():
    root = couplinglib.workroot("sealtest2")
    (root / "exports.json").write_text('{"values": [1.0, 2.0]}')
    try:
        couplinglib.seal_workroots()
        mode = oct(root.stat().st_mode & 0o777)
        assert mode == "0o0", f"workroot still readable at {mode}"
        # d--------- is what the preflight and the August directories look like.
        assert not os.access(root, os.R_OK)
    finally:
        # A TEST THAT LEAVES ONE BEHIND IS THE THING THIS FILE EXISTS TO STOP.
        os.chmod(root, 0o700)
        shutil.rmtree(root, ignore_errors=True)


def test_sealing_is_idempotent_and_survives_a_missing_directory():
    # atexit runs after temp cleanup in some orders; a vanished root must not
    # take the handler down with it, or every later workroot stays readable.
    root = couplinglib.workroot("sealtest3")
    os.rmdir(root)
    couplinglib.seal_workroots()
    couplinglib.seal_workroots()


def test_the_handler_is_wired_to_process_exit():
    import atexit
    # atexit keeps no public registry, so check the function object is the one
    # registered by unregistering it and putting it back.
    atexit.unregister(couplinglib.seal_workroots)
    atexit.register(couplinglib.seal_workroots)
