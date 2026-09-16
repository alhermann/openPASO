"""The capture script drives the UI by data-testid.

`product_vid.md` section 5 records what happens without this test: the ids the
capture script looked for were renamed in the app, the script stopped matching
anything, and nothing failed. The old screenshots were still on disk and the
render still succeeded, so the film quietly froze at an old version of the UI.

Reading the contract from one file and asserting it here means a rename breaks
the suite instead of the film.
"""
import json
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / "static"


def _contract() -> list[str]:
    return json.loads((STATIC / "testids.json").read_text())["ids"]


def test_every_advertised_testid_exists_in_the_page():
    page = (STATIC / "index.html").read_text()
    missing = [i for i in _contract() if f'data-testid="{i}"' not in page]
    assert not missing, (
        f"testids.json advertises {missing}, which index.html does not carry. "
        "Either the markup was renamed and the contract was not, or the "
        "contract gained an id nobody tagged."
    )


def test_the_page_declares_no_testid_the_contract_does_not_know():
    import re
    page = (STATIC / "index.html").read_text()
    found = set(re.findall(r'data-testid="([a-z0-9-]+)"', page))
    unknown = sorted(found - set(_contract()))
    assert not unknown, (
        f"index.html carries {unknown}, absent from testids.json. Add them "
        "there so the capture script and this test agree on one list."
    )
