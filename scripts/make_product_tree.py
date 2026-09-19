#!/usr/bin/env python3
"""Reduce a development checkout of openPASO to what a user needs to run it.

The development repository (alhermann/openPASO) carries the test suite, the tier-2 fixtures, the
measurement harnesses, scan results and development notes. The product repository
(Hereon-InstituteMS/openPASO) carries only what installing, running and documenting openPASO needs.
This script is the one definition of that difference, so a product release is reproducible from any
development commit in one command instead of a hand-picked deletion list.

Run it on a disposable checkout, never on a working clone. The pass removes scripts/, this file
included, so check the result from the development checkout:

    python scripts/make_product_tree.py --repo <copy> --dry-run
    python scripts/make_product_tree.py --repo <copy>
    python scripts/make_product_tree.py --repo <copy> --check
"""
from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEV_REPO = "alhermann/openPASO"
PRODUCT_REPO = "Hereon-InstituteMS/openPASO"
DEV_SITE = "https://alhermann.github.io/openPASO"
PRODUCT_SITE = "https://hereon-institutems.github.io/openPASO"

# Development-only paths, matched with fnmatch against repository-relative paths.
REMOVE = (
    # webui/tests/* STAYS (Alexander, 2026-09-18). The browser interface makes checkable claims --
    # "Finished" means a solver ran AND openPASO verified it, no home directory reaches the browser,
    # Stop ends every process a run started -- and those fast tests are how a contributor on a
    # product clone checks them. They need only webui and langgraph_eval, both of which ship; the
    # live suites need a key and say so. tests/, scripts/, benchmarks/ and validation/ stay out:
    # they are the development evidence, which a user does not install.
    "tests/*", "scripts/*", "benchmarks/*", "validation/*",
    "docs/CONSOLIDATION.md",
    # data the server never reads: build-time ledgers, fingerprints, grading studies, recordings
    "data/convergence/*", "data/fingerprints/*", "data/execution_ledger_*.json",
    "data/coupled_*.json", "data/solver_recommendations.yaml", "data/webui_sessions/*",
    ".github/workflows/knowledge-freshness.yml",      # runs the test suite, which is not shipped
    "product_vid.md", "clear_history.sh", "check_solver_updates.sh",
    "ONBOARDING*", "HANDOFF*", "*.log", "ui/node_modules/*", "langgraph_eval/test_*.py",
    # a run's own folder: the prompt, the model output and someone's working directory. It is
    # gitignored, and cut here as well in case the ignore ever slips.
    "eval_interactive/*",
    # The web interface SHIPS. It was held out of the first sync (Hereon PR #55) only while its
    # rewrite was in flight; it goes to the product repository as its own pull request. Removing it
    # here again would delete it from the product on the next sync.
)

# Tests that SHIP: they exercise a path the product carries and need nothing development-only --
# no fixtures, no campaign data, no network. Everything else under tests/ stays in development.
# Each entry is a decision, which is why this is a list and not a pattern: webui/tests came in on
# 2026-09-18 (the interface makes checkable claims and a contributor on a product clone must be
# able to check them), and the web-search test came with Hereon PR #59, which put it under tests/
# while the code it covers (langgraph_eval/agent.py) ships. Without this entry the next product cut
# would delete a file the product repository had already merged.
KEEP_TESTS = ("webui/tests/*", "tests/test_web_search_reports_a_block.py")

# src/blind_eval: the product's own tools import exactly these two modules (verify_interface_flux,
# audit_results). The rest of the package is the evaluation campaign's grading harness.
BLIND_EVAL_KEEP = {"src/blind_eval/interface.py", "src/blind_eval/evidence.py", "src/blind_eval/__init__.py"}
BLIND_EVAL_INIT = '''"""Interface and run-evidence checks used by openPASO's verification tools.

interface   reads two sides' interface files and measures how far they disagree
evidence    reads a solver's own console output for proof that a run happened
"""
'''

CONTRIBUTING_PRODUCT = f"""# Contributing to openPASO

Thank you for your interest. openPASO is a community project under active development, and it gets
better every time someone reports what went wrong.

**Development happens in the development repository, <https://github.com/{DEV_REPO}>.** It carries
the test suite, the fixtures that back each served claim, and the measurement tooling. This
repository is the released product: what installing and running openPASO needs. Please open issues
here or there, and pull requests there.

The browser interface's fast tests ship with it and need no key:

```bash
pytest webui/tests/test_app.py -q
```

The full guide, in plain language, is on the website: <{PRODUCT_SITE}/contribute/>.

The rules a change has to meet:

- **Every improvement helps all simulations, not one example.** Templates use placeholders, never
  the dimensions of a specific case.
- Every new or changed parameter key or keyword is cited to the solver's own source, file and line.
- The pull request says what changed, which failure it catches, and how it was checked.
"""

# Text edits: (file, exact old text, new text). Each must match exactly once, so a changed source
# fails loudly instead of shipping a half-edited page.
EDITS = (
    ("README.md", f"git clone https://github.com/{DEV_REPO}.git", f"git clone https://github.com/{PRODUCT_REPO}.git"),
    ("docs/getting-started/install.md", f"git clone https://github.com/{DEV_REPO}.git",
     f"git clone https://github.com/{PRODUCT_REPO}.git"),
    ("CITATION.cff", f'repository-code: "https://github.com/{DEV_REPO}"',
     f'repository-code: "https://github.com/{PRODUCT_REPO}"'),
    ("mkdocs.yml", f"site_url: {DEV_SITE}/", f"site_url: {PRODUCT_SITE}/"),
    ("mkdocs.yml", f"repo_url: https://github.com/{DEV_REPO}", f"repo_url: https://github.com/{PRODUCT_REPO}"),
    ("CITATION.cff",
     "  This is the development repository of openPASO. The released product is\n"
     "  Hereon-InstituteMS/openPASO, first published as OASiS; the DOIs below archive\n"
     "  it under that name. Please cite the archived release.\n",
     "  openPASO was first published as OASiS; the DOIs below archive it under that\n"
     "  name. Please cite the archived release.\n"),
    # a comment in a kept module names a file of the evaluation campaign, which is not shipped
    ("src/blind_eval/evidence.py", "    # campaign3_blind/grading/evidence2.py for the run-log contract: a",
     "    # the evaluation grader for the run-log contract: a"),
)
EVERYWHERE = (("README.md", DEV_SITE, PRODUCT_SITE), ("check_install.py", DEV_SITE, PRODUCT_SITE))

PRIVATE = re.compile(r"(?:/home/|/Users/|(?<![\w.-])/media/)(?!user/|you/|youruser/|username/|runner/|conda/)"
                     r"[A-Za-z0-9._-]+/|Schreibtisch|qwen_uplift_test|ofa-v2|honest_rounds_|campaign3_blind")


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def tracked() -> list[str]:
    return [p for p in _git("ls-files", "-z").split("\0") if p]


def _removed(rel: str) -> bool:
    if any(fnmatch.fnmatch(rel, pat) for pat in KEEP_TESTS):
        return False
    if rel.startswith("src/blind_eval/") and rel not in BLIND_EVAL_KEEP:
        return True
    return any(fnmatch.fnmatch(rel, pat) for pat in REMOVE)


def _served_assets() -> set[str]:
    index = REPO / "webui" / "static" / "index.html"
    if not index.is_file():
        return set()
    return {"webui/static/" + m for m in re.findall(r'(assets/[^"\']+)', index.read_text())}


def _stale_assets(files) -> list[str]:
    served = _served_assets()
    return [p for p in files if p.startswith("webui/static/assets/") and p not in served
            and not p.endswith((".woff", ".woff2", ".ttf"))]


def _pyproject(text: str) -> str:
    """Drop the development extra, but SHIP A WAY TO RUN THE TESTS THAT SHIP.

    Measured on the cut itself: removing the dev extra and the pytest config left a product clone
    that carries webui/tests and tests/test_web_search_reports_a_block.py, tells the reader in
    CONTRIBUTING to run them, and has no pytest to run them with -- and no `pythonpath = src`, so
    even an installed pytest could not import the server. Shipping a test nobody can run is worse
    than not shipping it: it reads as a claim that was checked.
    """
    # the dev extra becomes a `test` extra: the same tools, for the tests that ship
    # `[^\]]*` stopped at the first "]", which since the dev extra gained "openpaso[webui]" is
    # INSIDE an entry rather than the end of the list -- a bracket read as a boundary, so the
    # substitution silently did not fire and the product shipped tests with no way to run them.
    # Match to the end of the line instead: this list is written on one line and the check below
    # fails loudly if it ever is not.
    text = re.sub(r"\n# Development\ndev = \[.*\]\n(?:#[^\n]*\n)*",
                  "\n# Running the tests that ship with openPASO: pip install -e \".[test]\"\n"
                  "test = [\"pytest>=7.0\", \"pytest-asyncio>=0.20\", \"openpaso[webui]\"]\n", text)
    text = re.sub(r"\n\[tool\.pytest\.ini_options\]\n.*?(?=\n\[build-system\])", "\n", text, flags=re.S)
    return text.replace("\n[build-system]", PRODUCT_TEST_CONFIG + "\n[build-system]", 1)


# What a product clone needs to run the tests it carries, and nothing more.
PRODUCT_TEST_CONFIG = """
[tool.pytest.ini_options]
# `pythonpath` is what lets a test import the server from src/ without installing a path hack.
pythonpath = ["src"]
testpaths = ["webui/tests/test_app.py", "tests"]
"""


def apply(dry: bool) -> int:
    files = tracked()
    remove = sorted({p for p in files if _removed(p)} | set(_stale_assets(files)))
    print(f"remove {len(remove)} of {len(files)} tracked files")
    for top in sorted({p.split('/')[0] for p in remove}):
        print(f"  {sum(1 for p in remove if p.split('/')[0] == top):5d}  {top}")
    if dry:
        return 0
    for i in range(0, len(remove), 500):
        _git("rm", "-q", "--", *remove[i:i + 500])
    for f, old, new in EDITS:
        path = REPO / f
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(f"{f}: text to edit found {text.count(old)} times -- source changed, update EDITS")
        path.write_text(text.replace(old, new))
    for f, old, new in EVERYWHERE:
        path = REPO / f
        path.write_text(path.read_text().replace(old, new))
    (REPO / "CONTRIBUTING.md").write_text(CONTRIBUTING_PRODUCT)
    (REPO / "src/blind_eval/__init__.py").write_text(BLIND_EVAL_INIT)
    (REPO / "pyproject.toml").write_text(_pyproject((REPO / "pyproject.toml").read_text()))
    _git("add", "--", "CONTRIBUTING.md", "src/blind_eval/__init__.py", "pyproject.toml",
         *{f for f, *_ in EDITS}, *{f for f, *_ in EVERYWHERE})
    return check()


def check() -> int:
    problems = []
    files = tracked()
    problems += [f"non-product path still tracked: {p}" for p in files if _removed(p)]
    problems += [f"built asset index.html does not serve: {p}" for p in _stale_assets(files)]
    problems += [f"index.html serves an untracked asset: {a}" for a in _served_assets() if a not in files]
    for p in files:
        if p == ".gitignore":          # an ignore rule names what must stay out; it cannot leak it
            continue
        try:
            text = (REPO / p).read_text(errors="strict")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if PRIVATE.search(line):
                problems.append(f"private path or campaign location: {p}:{n}")
    for doc in ("README.md", "CONTRIBUTING.md", "CITATION.cff", "mkdocs.yml", "check_install.py"):
        text = (REPO / doc).read_text() if (REPO / doc).is_file() else ""
        if DEV_REPO in text and doc != "CONTRIBUTING.md" or DEV_SITE in text:
            problems.append(f"{doc} sends users to the development repository or website")
    for doc in [p for p in files if p.endswith(".md")]:
        text = (REPO / doc).read_text(errors="ignore")
        for m in re.finditer(r"(?<![\w/.-])((?:tests|scripts|benchmarks|validation)/[\w./-]+)", text):
            problems.append(f"{doc} points at a removed path: {m.group(1)}")
    pyp = (REPO / "pyproject.toml").read_text()
    if "\ndev = [" in pyp:
        problems.append("pyproject.toml still carries the development extra")
    # The product DOES configure pytest -- for the tests it ships. What it must not do is point at
    # the development suite, or ship tests with no way to run them (measured on the cut: a product
    # clone carried webui/tests, told the reader to run them, and had no pytest).
    if any(p.startswith(("webui/tests", "tests/")) for p in tracked()):
        if "\ntest = [" not in pyp or "pytest" not in pyp:
            problems.append("tests ship but pyproject offers no way to install pytest")
        if "pythonpath" not in pyp:
            problems.append("tests ship but pyproject does not put src on the path for them")
        for line in pyp.splitlines():
            if line.startswith("testpaths"):
                named = re.findall(r'"([^"]+)"', line)
                gone = [n for n in named if not (REPO / n).exists()]
                if gone:
                    problems.append(f"testpaths names what the product does not carry: {gone}")
    # every product import of blind_eval must resolve inside the kept set
    for p in files:
        if p.endswith(".py") and not p.startswith("src/blind_eval/"):
            text = (REPO / p).read_text(errors="ignore")
            for m in re.finditer(r"from blind_eval(?:\.(\w+))? import ([\w, ()\n]+)", text):
                mods = [m.group(1)] if m.group(1) else [x.strip() for x in re.split(r"[,\s()]+", m.group(2).split(" as ")[0]) if x.strip()]
                for mod in mods:
                    if f"src/blind_eval/{mod}.py" not in files:
                        problems.append(f"{p} imports blind_eval.{mod}, which the product does not ship")
    for p in problems:
        print("  ✘", p)
    print(f"{'OK' if not problems else 'NOT READY'}: {len(problems)} problem(s)")
    return 1 if problems else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--check", action="store_true")
    ap.add_argument("--repo", type=Path, help="checkout to act on (default: the one holding this file)")
    a = ap.parse_args()
    global REPO
    if a.repo:
        REPO = a.repo.resolve()
    return check() if a.check else apply(dry=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
