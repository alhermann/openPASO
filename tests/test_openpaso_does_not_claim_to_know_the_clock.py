"""openPASO never tells the agent how much budget it has, because it cannot see it.

MEASURED. Five served strings in workspace_advisor.py opened with the
unconditional sentence "You have budget left now." Nothing in src/ can observe a
clock: there is no deadline, no step counter and no remaining-calls value
anywhere in the server. The only budget signal in the system is the harness
stamping "[clock: N min left of M]" onto run_bash replies.

The sentence was not merely unfounded, it was frequently false. Of the cells that
stall after one coupled level, 46% are at 95% or more of the wall when they stop,
and among the give-up cells the median first surrender comes with 15 of 45
minutes left -- these are runs being told they have room precisely when they are
out of it, by a server that is guessing.

What each of those findings had to say was true and is kept; only the claim about
the clock is gone.
"""
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Phrases that assert something about the agent's REMAINING budget. A finding may
# freely say that fixing a defect now is cheaper than fixing it later -- that is
# a statement about the defect, not about the clock.
_CLOCK_CLAIMS = (
    r"you have budget left",
    r"you have time left",
    r"there is still time",
    r"you have \w+ (?:minutes|calls|actions) left",
    r"plenty of (?:budget|time)",
)


def test_no_served_string_tells_the_agent_what_its_budget_is():
    offenders = []
    for f in sorted((ROOT / "src").rglob("*.py")):
        try:
            text = f.read_text()
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            # Comments record measurements about recorded cells and are not
            # served; only string content reaches an agent.
            if line.lstrip().startswith("#"):
                continue
            for pat in _CLOCK_CLAIMS:
                if re.search(pat, line, re.I):
                    offenders.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()[:90]}")
    assert not offenders, (
        "openPASO cannot observe the agent's clock, so it must not describe it:\n  "
        + "\n  ".join(offenders))
