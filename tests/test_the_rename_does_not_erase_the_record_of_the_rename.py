"""Replaying the rename must not destroy the sentences that name the old spelling.

scripts/rename_to_openpaso.py exists to be replayed on any branch that still carries the old name.
Replayed on the renamed tree it would have rewritten every sentence written to RECORD the rename --
"openPASO was first published as OASiS" becomes "openPASO was first published as openPASO", the
citation file loses the name the DOIs are archived under, and the test that proves both environment
spellings answer becomes a test of one spelling. Machinery destroying what it protects; found
2026-09-18 by running the script against the tree it had already renamed.

It also forced Hereon-InstituteMS/openPASO back to the old repository name. That was right until
2026-09-17 and wrong after it, because that repository was itself renamed.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("ren", ROOT / "scripts" / "rename_to_openpaso.py")
ren = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ren)


def test_the_history_sentences_survive_a_replay():
    for sentence in ("openPASO was first published as **OASiS**",
                     "first published as OASiS",
                     "so macht es auch OASiS"):
        assert ren.rewrite(sentence) == sentence, sentence


def test_the_product_repository_url_is_not_pushed_back_to_the_old_name():
    url = "https://github.com/Hereon-InstituteMS/openPASO"
    assert ren.rewrite(url) == url


def test_replaying_it_on_this_tree_changes_nothing():
    """The strongest form: the script is inert on a tree it has already renamed."""
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "rename_to_openpaso.py"), "--check"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
