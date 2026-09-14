#!/usr/bin/env bash
# ROUND 8 — the re-baseline. What is OASiS worth after the rebuild?
#
# Every number we hold is stale. 27B OASiS 37.5% single-code was measured
# before five changes, each of which moves the arm under test:
#
#   1. the harness silently ended runs that had not finished — 112 of 813
#      across the campaign, and NOT arm-neutral: 19.5% of OASiS runs against
#      8.1% of bare. Every prior uplift was measured with OASiS losing one run
#      in five to us.
#   2. the interface flux recovery never converged (order 0.00 in max norm);
#      it is now 2.00 across four codes, in all 21 participants.
#   3. OPTION B: OASiS no longer serves a working solve. This is the one that
#      may COST us. An agent must now hand-write a two-code coupling, ~1000
#      lines, in 45 minutes. If the OASiS arm falls, that is a real finding
#      about what the knowledge alone is worth, and we need it now rather than
#      after another week of per-cell work.
#   4. the agent had no clock and threw away 59% of two runs believing it was
#      out of time. Both arms now see the remaining minutes.
#   5. the self-check was blind to every coupled submission — "AMBIGUOUS INPUT"
#      instead of "your field is zero" — so the gate never fired on the half of
#      the campaign that scores zero.
#
# THE QUESTIONS, in order of what they change:
#   * what did Option B cost the OASiS arm on single-code cells?
#   * is the coupled score still zero, and if a run fails, does it now fail
#     LOUDLY (near-zero field flagged before submission) instead of reaching
#     the grader as FABRICATED_NO_RUN?
#   * do runs still stop at ~40% of the clock now that they can see it?
#
# This is also the freeze gate: the criterion is a round that teaches OASiS
# nothing new (paper §3.2), not a score. Two fresh seeds, never used.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=campaign3_blind/round8_rebuild.log

if [ "$(stat -c %A "$OASIS_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OASIS_BLIND_KEYS"))" >> "$LOG"
  exit 1
fi

# Nothing CPU-heavy may run beside a measurement: a niced tier-2 fixture once
# ran at 1226% CPU next to a round and inflated its wall-clock and timeouts.
# `pgrep -fc` prints 0 AND exits 1 when nothing matches, so `|| echo 0` yields
# "0\n0" and this guard would refuse forever. `| wc -l` prints one number.
if [ "$(pgrep -f 'run_blind[.]py' 2>/dev/null | wc -l)" != "0" ]; then
  echo "REFUSING: agent runs already in flight; a round must have the machine" >> "$LOG"
  exit 1
fi

CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 \
C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"
SEEDS="14 15"
PAR=6                      # concurrent cells; the box has 32 threads

echo ">>> round 8 (rebuild re-baseline) start $(date '+%F %T')" >> "$LOG"
echo ">>> seeds $SEEDS, $(echo $CELLS | wc -w) cells, both arms" >> "$LOG"

# mapfile + </dev/null: a backgrounded child inside `while read ... done < list`
# inherits the loop's stdin and eats the list — that cost 15 of 17 cells once.
mapfile -t CELL_ARR <<< "$(echo $CELLS | tr ' ' '\n')"

running=0
for s in $SEEDS; do
  for c in "${CELL_ARR[@]}"; do
    [ -z "$c" ] && continue
    ( for arm in BARE MCP; do
        d="campaign3_blind/runs/${c}_27b_${arm}_seed${s}"
        [ -f "$d/ledger.json" ] && continue        # resume, never overwrite
        "$PY" campaign3_blind/run_blind.py --model 27b --conditions "$arm" \
             --problems "$c" --seed "$s" --phase development >> "$LOG" 2>&1
      done
      echo "<<< $c seed $s done $(date '+%F %T')" >> "$LOG" ) < /dev/null &
    running=$((running + 1))
    if [ "$running" -ge "$PAR" ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done
done
wait
echo ">>> round 8 complete $(date '+%F %T')" >> "$LOG"
