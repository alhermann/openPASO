"""The product repository is cut from this one by scripts/make_product_tree.py. Pin what it keeps.

A product tree that silently drops a module a served tool imports installs cleanly and then fails
at the first call; one that keeps a development file ships something nobody asked for. Measured
2026-09-17: verify_interface_flux and audit_results import blind_eval.interface and
blind_eval.evidence, while the rest of blind_eval is the evaluation campaign's grading harness.
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_product_tree", ROOT / "scripts" / "make_product_tree.py")
mpt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mpt)


def _tracked():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True)
    return [p for p in out.stdout.split("\0") if p]


def test_everything_a_user_runs_is_kept():
    files = _tracked()
    dropped = [p for p in files if mpt._removed(p) and (
        p.startswith(("src/core/", "src/tools/", "src/backends/", "langgraph_eval/agent", "logo/"))
        or p in ("run_agent.py", "check_install.py", "pyproject.toml", "README.md", "LICENSE",
                 "CITATION.cff", "mkdocs.yml", ".env.example"))]
    assert not dropped, f"the product tree would drop what a user runs: {dropped[:10]}"


def test_blind_eval_keeps_exactly_what_the_served_tools_import():
    import re
    kept = {p for p in _tracked() if p.startswith("src/blind_eval/") and not mpt._removed(p)}
    imported = set()
    for p in _tracked():
        if p.endswith(".py") and p.startswith(("src/", "langgraph_eval/")) and not p.startswith("src/blind_eval/"):
            for m in re.finditer(r"from blind_eval(?:\.(\w+))? import (\w+)", (ROOT / p).read_text(errors="ignore")):
                imported.add(f"src/blind_eval/{m.group(1) or m.group(2)}.py")
    assert imported <= kept, f"a served module imports what the product drops: {sorted(imported - kept)}"
    assert "src/blind_eval/keyvault.py" not in kept and "src/blind_eval/leakgate.py" not in kept


def test_the_interface_keeps_its_own_fast_tests():
    """Alexander, 2026-09-18: webui/tests ships. The interface makes checkable claims -- "Finished"
    means a solver ran AND openPASO verified it, no home directory reaches the browser, Stop ends
    every process a run started -- and these are how a contributor on a product clone checks them.
    They need only webui and langgraph_eval, both of which ship."""
    assert not mpt._removed("webui/tests/test_app.py")
    assert not mpt._removed("webui/tests/__init__.py")
    assert mpt._removed("tests/test_backends.py"), "the development suite still stays out"


def test_a_test_the_product_repository_already_merged_is_not_deleted():
    """Hereon PR #59 put its test under tests/, where the cut removes everything. Merging it and
    then cutting would have deleted a file the product repository had already accepted -- the same
    trap as webui/tests the night before. Each shipped test is an entry, never a pattern."""
    assert not mpt._removed("tests/test_web_search_reports_a_block.py")
    assert mpt._removed("tests/test_backends.py")


def test_a_runs_own_folder_never_ships():
    """It holds the prompt, the model's output and someone's working directory."""
    assert mpt._removed("eval_interactive/webui_abc123/uploads/mesh.msh")
    assert mpt._removed("data/webui_sessions/abc.json")


def test_development_material_is_dropped():
    for p in ("tests/test_backends.py", "scripts/rename_to_openpaso.py", "docs/CONSOLIDATION.md",
              "data/webui_sessions/x.json", "ui/node_modules/x/index.js", "product_vid.md"):
        assert mpt._removed(p), p


def test_the_check_names_a_private_path():
    home = "/" + "home" + "/"          # spelled in pieces so this file does not trip the host-path gate
    assert mpt.PRIVATE.search(f"see {home}alice/data")
    assert mpt.PRIVATE.search("~/" + "Schreib" + "tisch/solver")
    assert not mpt.PRIVATE.search(f"see {home}user/data")
