"""Where every input comes from, and the refusal to guess when one is missing.

Three location rules, each the fix for a defect that actually happened:

* **The answers are located by OPENPASO_BLIND_KEYS.** They live outside the
  repository on purpose. A hardcoded path here once pointed at a SECOND, older
  copy of the campaign, so grading found stale answers beside a stale grader.

* **The problems always come from THIS checkout.** Never the copy beside the
  answers — that is how a stale question sheet gets served.

* **Helper code is loaded from THIS checkout by absolute file path**, not from
  whatever `sys.path` happens to hold. `code_ran` once imported the evidence
  gate from a different worktree, so hash-freezing this repository froze
  NOTHING about the thresholds that actually executed at grading time.

And one refusal rule: **a missing `spec_public.json` is a HARD ERROR, never a
default.** The previous grader fell back to `dim = 2`, so the 3-D instances B4
and B6 would have been parsed with the wrong column count and graded on
nonsense that looked like a result. A grader that cannot say what the problem
is must stop, not guess. `GraderConfigError` means THE CELL OR THE SETUP is
broken; it is never an outcome for the submission.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

# grading/ -> campaign3_blind/ -> repository root
REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "campaign3_blind"
PROBLEMS = CAMPAIGN / "problems"


class GraderConfigError(RuntimeError):
    """The cell, the key custody, or the grader setup is broken.

    Deliberately NOT an outcome. An outcome describes the submission; this
    describes the experiment, and an experiment defect must halt grading loudly
    instead of being booked against the agent.
    """


# ── helper modules, pinned to this checkout ───────────────────────────────
_LOADED: dict[str, object] = {}


def _load_checkout_module(rel: str, alias: str):
    """Import a module from an absolute path inside THIS repository.

    `importlib` from an explicit file location, not `sys.path`: nothing another
    worktree or an installed package puts on the path can shadow it, and a test
    asserts the loaded file lives under this checkout.
    """
    if alias in _LOADED:
        return _LOADED[alias]
    path = REPO / rel
    if not path.is_file():
        raise GraderConfigError(f"required module {path} is missing from this "
                                f"checkout; the grader cannot run without it")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    _LOADED[alias] = mod
    return mod


def evidence_mod():
    """blind_eval.evidence — the canonical `NDOF = <n>` contract line and the
    per-code structured signatures — from THIS checkout."""
    return _load_checkout_module("src/blind_eval/evidence.py", "c3g_evidence")


def interface_mod():
    """blind_eval.interface — interface CSV reading and two-sided jumps —
    from THIS checkout."""
    return _load_checkout_module("src/blind_eval/interface.py", "c3g_interface")


def keyvault_mod():
    """blind_eval.keyvault — encrypted-key loading — from THIS checkout.
    Loaded lazily: plaintext keys must not require the cryptography package."""
    return _load_checkout_module("src/blind_eval/keyvault.py", "c3g_keyvault")


# ── locations ─────────────────────────────────────────────────────────────
def keys_dir(override: Path | str | None = None) -> Path:
    if override:
        return Path(override)
    env = os.environ.get("OPENPASO_BLIND_KEYS")
    return Path(env) if env else CAMPAIGN / "keys"


def problems_dir(override: Path | str | None = None) -> Path:
    """Where the question sheets come from.

    OPENPASO_BLIND_PROBLEMS IS HONOURED, AND WAS NOT. Keys have always been
    relocatable through OPENPASO_BLIND_KEYS; problems were hardcoded. That
    asymmetry produced a silent false pass: checking a freshly drawn instance
    with `OPENPASO_BLIND_PROBLEMS=problems_dev3 OPENPASO_BLIND_KEYS=.../keys_dev3
    check_grader_accepts.py C2` read the NEW key against the OLD task text, and
    reported "1/1 instances would accept a correct submission" — a green result
    from mismatched inputs. The give-away was the reported interface count, 44,
    against the 22 the new task states.

    A fresh draw into its own root is the documented way to avoid overwriting a
    spent instance (build_balanced --problems-root), so the grader has to be
    able to read one.
    """
    if override:
        return Path(override)
    env = os.environ.get("OPENPASO_BLIND_PROBLEMS")
    return Path(env) if env else PROBLEMS


# ── the public spec: mandatory, never defaulted ───────────────────────────
def load_spec(problem_id: str, problems: Path | str | None = None) -> dict:
    pdir = problems_dir(problems) / problem_id
    if not pdir.is_dir():
        raise GraderConfigError(
            f"no problem directory {pdir}: the question sheets always come "
            f"from this checkout, and this checkout has no {problem_id}")
    spec_path = pdir / "spec_public.json"
    if not spec_path.is_file():
        raise GraderConfigError(
            f"{problem_id} has no spec_public.json under {pdir}. This is a "
            f"HARD ERROR, not a case for a default: the grader once fell back "
            f"to dim = 2 here, and the 3-D instances B4 and B6 would have "
            f"been parsed with the wrong column count and graded on nonsense. "
            f"The builder must write the public spec before the cell can be "
            f"graded.")
    try:
        spec = json.loads(spec_path.read_text())
    except (json.JSONDecodeError, OSError) as ex:
        raise GraderConfigError(f"{spec_path} is unreadable or not JSON: {ex}")
    dim = spec.get("dim")
    if not isinstance(dim, int) or dim not in (2, 3):
        raise GraderConfigError(
            f"{problem_id}: spec_public.json carries no usable 'dim' "
            f"(got {dim!r}); refusing to guess the dimension")
    return spec


def load_task(problem_id: str, problems: Path | str | None = None) -> str:
    task = problems_dir(problems) / problem_id / "task.txt"
    if not task.is_file():
        raise GraderConfigError(
            f"{problem_id} has no task.txt under {task.parent}; a cell whose "
            f"task text is missing cannot be checked against its grading grid")
    return task.read_text(encoding="utf-8", errors="ignore")


# ── the sealed key ────────────────────────────────────────────────────────
def load_key(problem_id: str, keys: Path | str | None = None,
             passphrase: str | None = None) -> dict:
    kdir = keys_dir(keys) / problem_id
    plain, enc = kdir / "key.json", kdir / "key.json.enc"
    if plain.is_file():
        try:
            return json.loads(plain.read_text())
        except (json.JSONDecodeError, OSError) as ex:
            raise GraderConfigError(f"{plain} is unreadable or not JSON: {ex}")
    if enc.is_file():
        if passphrase is None:
            raise GraderConfigError(
                f"the key for {problem_id} is encrypted ({enc}) and no "
                f"passphrase was supplied. Grading an encrypted campaign "
                f"needs the passphrase typed at grading time; refusing to "
                f"grade without the key rather than grading against nothing.")
        try:
            return json.loads(
                keyvault_mod().decrypt_bytes(enc.read_bytes(), passphrase))
        except Exception as ex:
            raise GraderConfigError(
                f"could not decrypt {enc}: {type(ex).__name__}: {ex}")
    raise GraderConfigError(
        f"no key for {problem_id} under {kdir} (looked for key.json and "
        f"key.json.enc). Keys are located by OPENPASO_BLIND_KEYS "
        f"(currently {keys_dir(keys)}); a cell without a key cannot be "
        f"graded and must not be silently skipped.")


# ── spec/key coherence ────────────────────────────────────────────────────
def crosscheck_spec_key(problem_id: str, spec: dict, key: dict) -> None:
    """The public spec and the sealed key describe the same instance, or the
    build is broken and grading must stop."""
    if key.get("id") not in (None, problem_id):
        raise GraderConfigError(
            f"key id {key.get('id')!r} does not match problem {problem_id!r}: "
            f"the key on disk is for a different instance")
    for field in ("dim", "kind", "theoretical_order", "tol", "band", "mesh_N"):
        s, k = spec.get(field), key.get(field)
        if s is not None and k is not None and s != k:
            raise GraderConfigError(
                f"{problem_id}: spec_public.json and the key disagree on "
                f"{field!r} ({s!r} vs {k!r}). The question sheet and the "
                f"marking scheme describe different problems; this is a build "
                f"defect, not a submission property.")


def evidence_grade(problem_id: str, key: dict, spec: dict) -> tuple[int, list]:
    """The evidence grade of the cell, READ FROM THE KEY and stamped on the
    result. Grade semantics (DESIGN.md Amendment 2 §4):

        1 — exact manufactured solution: proves the answer is CORRECT
        2 — monolithic reference: proves the SPLIT did not change the answer
        3 — band-only: proves the value is in a pre-registered band, no more

    Keys older than the field (the FE1..FC2 generation) carry an exact
    solution but no `evidence_grade`; that IS the grade-1 definition, so it is
    inferred — loudly, with a note, never silently.
    """
    notes = []
    g = key.get("evidence_grade")
    if g is None:
        if key.get("grading") == "band-only":
            g = 3
        elif key.get("grading") == "reference":
            g = 2
        elif key.get("exact_solution") is not None:
            g = 1
            notes.append(
                "evidence_grade inferred as 1 from the presence of an exact "
                "solution: this key predates the evidence_grade field")
        else:
            raise GraderConfigError(
                f"{problem_id}: the key carries neither an evidence_grade nor "
                f"an exact solution nor a grading mode; the cell's evidence "
                f"class is undefined and the result would be unpoolable")
    sg = spec.get("evidence_grade")
    if sg is not None and sg != g:
        raise GraderConfigError(
            f"{problem_id}: spec_public.json says evidence_grade {sg} but the "
            f"key says {g}; the build is inconsistent")
    return int(g), notes


def parse_rel_tol(text, default: float = 1e-6) -> float:
    """'1e-6 relative' -> 1e-6. The spec states the interface tolerance as
    prose; take the leading float and fall back loudly-typed."""
    if text is None:
        return default
    if isinstance(text, (int, float)):
        return float(text)
    import re
    m = re.match(r"\s*([0-9.eE+-]+)", str(text))
    try:
        return float(m.group(1)) if m else default
    except ValueError:
        return default
