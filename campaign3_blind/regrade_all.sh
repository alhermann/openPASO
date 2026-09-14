#!/usr/bin/env bash
# REGRADE THE WHOLE BACK-CATALOGUE UNDER ONE GATE — the current one.
#
# WHY. The two readable grades directories disagree with each other, and the
# newer one is the WRONG one:
#
#   C7_27b_BARE_seed2   grades_pre_repair: CORRECT (order 1.9757)
#                       grades_pre_naming_fixes: FABRICATED_NO_RUN
#   C8_27b_BARE_seed4   grades_pre_repair: COMPLETED_UNPHYSICAL (order 2.0170)
#                       grades_pre_naming_fixes: FABRICATED_NO_RUN
#
# Both fabrication labels come from `COUPLING_EVIDENCE_CONTRADICTED` with a
# measured ratio_cv of 7.9e-6, against a SYNTHETIC_RATIO_CV threshold of 1e-5
# that has since been tightened to 1e-12 — so under the current rule neither
# fires. Independent support, computed with the keys sealed: the new
# flux-from-field check recovers C8's two conductivities as 0.9999999 and
# 999.99992 from the agent's own submitted field. A run whose reported flux
# reproduces both of the task's conductivities to seven digits did not invent
# it.
#
# Consequence, stated plainly: NO coupled number and NO fabrication number may
# be quoted from either directory, including the SCORECARD's "bare completes
# none and fabricates". This pass is what makes them quotable. It costs no
# credits.
#
# Writes to a NEW directory. The two stale ones stay exactly as they are —
# they are the evidence that the gate moved, and overwriting them would erase
# the only record of the disagreement.
#
# CUSTODY. grade_round.py refuses to start unless the keys are SEALED, unseals,
# grades, and reseals in a `finally` block that also catches signals. This
# wrapper adds the one thing it cannot check for itself: that no agent is
# running. Unsealing the keys while an agent shell is alive would end the blind
# claim for every run in flight, whatever the agent did with the window.
#
# The passphrase is read from THIS SCRIPT'S STDIN, one line, and piped to the
# grader in memory. Never in argv, never in a file, never in the environment.
#
#   ./regrade_all.sh            # prompts on a terminal
#   ./regrade_all.sh < <(...)   # or one line on stdin
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OPENPASO_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OPENPASO_REPO=/home/alexander/Schreibtisch/ofa-v2
PY=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
# AS-RUN REGRADE 2026-09-03: graded against the problems tree the runs
# actually saw (OPENPASO_BLIND_PROBLEMS points at the 64ea4d4c extract; the
# ledgers' own task_sha256 match it 154/154 where a hash exists, and NO
# historical text demanded own-solver output, so the per-code gate takes
# the honest UNPROVEN branch for the whole back-catalogue instead of
# punishing runs for a clause added today).
OUTDIR=campaign3_blind/grades_asrun_2026_09_03
LOG=campaign3_blind/regrade_asrun_2026_09_03.log

# Every seed carrying a full or near-full matrix. Singleton seeds are scatter
# re-runs of one cell and are graded with the round that owns them.
# Round 9 (96/97) is deliberately absent: its launcher was stopped after live
# source edits mixed builds, and those legacy ledgers predate per-run source
# hashes. It is diagnostic material, not a quotable population.
SEEDS="2 3 4 5 6 7 8 9 10 11 14 15 34 35 40 43 50 60"

exec 9>/tmp/openpaso_regrade.lock
flock -n 9 || { echo "REFUSING: another regrade holds the lock" >&2; exit 1; }

live=$(pgrep -f 'run_blind[.]py' 2>/dev/null | wc -l)
if [ "$live" != "0" ]; then
  echo "REFUSING: $live agent run(s) in flight. Unsealing the keys beside a" >&2
  echo "live agent shell ends the blind claim for every run in flight." >&2
  exit 1
fi

if [ "$(stat -c %A "$OPENPASO_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: keys are not sealed ($(stat -c %A "$OPENPASO_BLIND_KEYS"))" >&2
  exit 1
fi

mkdir -p "$OUTDIR"
{
  echo ">>> regrade under the current gate, start $(date '+%F %T')"
  echo ">>> build $(git rev-parse --short HEAD), seeds $SEEDS"
} >> "$LOG"

# One line of stdin, forwarded to the grader's stdin. `read -r` keeps it out of
# argv; it is never echoed and never stored.
if [ -t 0 ]; then
  printf 'key passphrase: ' >&2
  read -r -s PHRASE
  echo >&2
else
  read -r PHRASE
fi
[ -n "$PHRASE" ] || { echo "no passphrase given" >&2; exit 2; }

printf '%s\n' "$PHRASE" | "$PY" campaign3_blind/grade_round.py \
    --seeds $SEEDS --model 27b --out "$OUTDIR/dev_grades_27b" --with-key \
    2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[1]}
unset PHRASE

state=$(stat -c %A "$OPENPASO_BLIND_KEYS")
echo ">>> keys after: $state" | tee -a "$LOG"
if ! bash campaign3_blind/shield_keys.sh status >> "$LOG" 2>&1; then
  bash campaign3_blind/shield_keys.sh seal | tee -a "$LOG"
  echo ">>> RESEALED ALL KEY STORES BY THE WRAPPER — the grader should have done this" | tee -a "$LOG"
fi
echo ">>> regrade done rc=$rc $(date '+%F %T')" | tee -a "$LOG"
exit "$rc"
