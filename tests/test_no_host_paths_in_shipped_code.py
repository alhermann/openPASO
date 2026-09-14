"""Nothing openPASO ships may carry the author's home directory.

This is a product rule, not a tidiness rule. The installed-version API
reference is served to the model that is driving the solvers. When it said

    run: /home/someone/miniconda3/envs/fenics/bin/python <script>.py

a model on anyone else's computer read that and used it verbatim, and the
command could not run. The path also published one person's username and
directory layout to everyone who cloned the repository.

Paths belong in core.host_paths as tokens, filled in at serve time from the
reader's own environment. This test fails if a literal home directory comes
back into the code that ships.

`campaign3_blind/`, captured solver logs and recorded transcripts are exempt:
they are records of runs that really happened on one machine, and rewriting a
record to look tidier would misstate it. They are not part of the product.
"""
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Any user's home directory, on Linux or macOS -- not just the author's. The
# rule is about paths that IDENTIFY A PERSON, so deliberately anonymous
# placeholders are allowed: writing "/home/user/4C" in a docstring is how you
# show the shape of a path without naming anybody. "conda" and "runner" appear
# inside real build-system messages that are quoted verbatim from a solver.
PLACEHOLDER_NAMES = {"user", "you", "youruser", "someuser", "username",
                     "...", "conda", "runner", "me", "USER"}
HOME_PATH = re.compile(r"(?:/home/|/Users/)([A-Za-z0-9._-]+)/")


def identifies_a_person(line: str) -> str | None:
    """The first home path in `line` that names someone, or None."""
    for match in HOME_PATH.finditer(line):
        if match.group(1) not in PLACEHOLDER_NAMES:
            return match.group(0)
    return None

# What the product actually ships and a stranger actually runs.
SHIPPED = ("src", "data", "langgraph_eval", "webui", "run_agent.py",
           "pyproject.toml", "README.md", "CONTRIBUTING.md", "CITATION.cff",
           ".env.example", "mcp_config.json")

# Records, not product: a log says what happened, and what happened had a path.
EXEMPT_PREFIXES = ("data/blind_",)


def tracked(paths) -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", *paths],
                         cwd=REPO, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def test_no_home_directory_is_shipped():
    offenders: list[str] = []
    for rel in tracked(SHIPPED):
        if rel.startswith(EXEMPT_PREFIXES):
            continue
        try:
            text = (REPO / rel).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            found = identifies_a_person(line)
            if found:
                offenders.append(f"{rel}:{number}: {found}…")
    assert not offenders, (
        "a home directory reached the shipped code. Replace it with a token "
        "from core.host_paths (see its docstring) so the reader's own path is "
        "filled in at serve time:\n  " + "\n  ".join(offenders[:25]))


def test_the_served_api_reference_resolves_every_token_it_uses():
    """A token that nothing fills in would be shown to the model verbatim."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from backends._installed_api import INSTALLED_API, render
    from core.host_paths import tokens

    known = set(tokens())
    for name in INSTALLED_API:
        rendered = render(name)
        # (?<!\$) so CMake's own ${DEAL_II_DIR} is not mistaken for our token
        left = set(re.findall(r"(?<!\$)\{[A-Z_]+\}", rendered))
        assert not left, (
            f"{name}: {sorted(left)} survived rendering. Every token must be "
            f"declared in core.host_paths; known tokens are {sorted(known)}")
