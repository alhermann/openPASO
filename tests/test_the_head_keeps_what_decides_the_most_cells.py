"""What must survive the coupled must-read's character cap, by measured cost.

The reply is trimmed to a fixed head. Anything added in the middle pushes an
equal number of characters off the end, and the source diff shows nothing:
you have to diff the SERVED text.

MEASURED, on my own change. Correcting the refusal guidance added 341
characters and silently evicted this from the head:

    "AND THE DRIVER ALREADY KEPT EACH LEVEL'S CONSOLE FOR YOU. `couple` saves
     every participant's captured output beside that side's exports.json as
     participant_output_level<k>.log ... The per-level execution log your task
     asks for is a COPY OF THAT FILE"

That passage is where an agent learns the console already exists and that the
deliverable is a verbatim copy of it. RUN_LOG_CONTRACT_UNMET is the single
largest malformed-submission cause on record -- 64 cells -- so of everything in
this block, that is among the least affordable to lose.

Each phrase below is pinned because losing it has a measured cost, not because
it reads well. Adding to this block is fine; pushing one of these out is not.
"""
import pathlib
import re
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = (ROOT / "src" / "tools" / "consolidated.py").read_text()
sys.path.insert(0, str(ROOT / "src"))

# MEASURE THE STRING THAT IS SERVED, NOT THE LITERAL THAT STARTS IT. This used a
# regex over the triple-quoted assignment, and the constant is EXTENDED after it
# (`_COUPLING_MUST_READ += "\n" + _PER_SIDE_NAMING`). The regex therefore
# under-measured the served block by 510 characters, so the ceiling below was
# passing on a block that had already grown past it -- a guard against silent
# growth that could not see the growth.
from tools.consolidated import (  # noqa: E402
    _COUPLING_HEAD_LIMIT, _COUPLING_MUST_READ)

HEAD_LIMIT = _COUPLING_HEAD_LIMIT
MUST_READ = _COUPLING_MUST_READ
assert HEAD_LIMIT == int(
    re.search(r"_COUPLING_HEAD_LIMIT\s*=\s*(\d+)", SRC).group(1))
HEAD = MUST_READ[:HEAD_LIMIT]


@pytest.mark.parametrize("phrase,why", [
    ("AND THE DRIVER ALREADY KEPT EACH LEVEL",
     "64 cells lost on RUN_LOG_CONTRACT_UNMET; this says the console already exists"),
    ("participant_output_level",
     "the exact filename the per-level run log is copied from"),
])
def test_the_head_still_carries_what_costs_most_to_lose(phrase, why):
    assert phrase in HEAD, (
        f"'{phrase}' has been pushed out of the first {HEAD_LIMIT} characters. {why}. "
        f"Something added earlier in the block evicted it -- shorten that, or move it "
        f"after this passage. The must-read is {len(MUST_READ)} characters long.")


def test_the_block_has_not_grown_without_anyone_noticing():
    """A ceiling with a reason: every character past the head is served only
    when the reply has room, and the block is already over it."""
    # 32_710 IS THE OLD CEILING RESTATED IN SERVED CHARACTERS, NOT A RELAXATION.
    # The previous 32_200 was measured against the triple-quoted literal alone,
    # which is 510 characters short of the string that is actually served. The
    # same ceiling, measured correctly, is 32_200 + 510. It has not been moved
    # to admit anything: at the time of writing the served block is 32_599, and
    # what matters is the two phrases below staying inside the head.
    assert len(MUST_READ) <= 32_710, (
        f"the coupled must-read is {len(MUST_READ)} characters against a "
        f"{HEAD_LIMIT}-character guaranteed head. Growth here is not free: it "
        f"evicts the tail silently. Measure the SERVED text before and after.")


def test_the_head_limit_itself_has_not_moved_silently():
    assert HEAD_LIMIT == 28000, (
        "the head limit changed; re-measure what now falls outside it before "
        "accepting this")
