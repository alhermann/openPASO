"""One variable, two spellings.

The project was called OASiS and its environment variables were named
``OASIS_*``. It is now called openPASO and they are named ``OPENPASO_*``.
Both spellings have to keep working, because they are read across a process
boundary: the server writes the mesh keys into a participant's environment,
and the participant that reads them may be a script somebody wrote, saved and
still runs from a year ago. A rename that only touched this repository would
break those quietly, at the far end, with an empty value.

So at start-up every ``OASIS_X`` gains a twin ``OPENPASO_X`` and every
``OPENPASO_X`` gains a twin ``OASIS_X``, unless the twin is already set --
an explicitly set value always wins over one inferred from the other name.

This file is deliberately excluded from ``scripts/rename_to_openpaso.py``:
it is the one place that must go on saying both names.
"""
from __future__ import annotations

import os
from typing import MutableMapping

LEGACY_PREFIX = "OASIS_"
CURRENT_PREFIX = "OPENPASO_"


def _twin(name: str) -> str | None:
    """The other spelling of `name`, or None if it is not one of ours."""
    if name.startswith(CURRENT_PREFIX):
        return LEGACY_PREFIX + name[len(CURRENT_PREFIX):]
    if name.startswith(LEGACY_PREFIX):
        return CURRENT_PREFIX + name[len(LEGACY_PREFIX):]
    return None


def install_aliases(environ: MutableMapping[str, str] | None = None) -> list[str]:
    """Give every one of our variables both spellings. Returns the names added.

    Idempotent, and safe to call again after the environment has changed.
    """
    env = os.environ if environ is None else environ
    added: list[str] = []
    for name, value in list(env.items()):        # snapshot: we mutate as we go
        twin = _twin(name)
        if twin is not None and twin not in env:
            env[twin] = value
            added.append(twin)
    return added


def set_env(name: str, value: str,
            environ: MutableMapping[str, str] | None = None) -> None:
    """Set one of our variables under both spellings at once."""
    env = os.environ if environ is None else environ
    env[name] = value
    twin = _twin(name)
    if twin is not None:
        env[twin] = value


def get_env(name: str, default: str | None = None,
            environ: MutableMapping[str, str] | None = None) -> str | None:
    """Read one of our variables under either spelling, current name first."""
    env = os.environ if environ is None else environ
    if name in env:
        return env[name]
    twin = _twin(name)
    if twin is not None and twin in env:
        return env[twin]
    return default
