#!/usr/bin/env bash
# PROBE 2 — does the work now REACH the finish line?
#
# Three things landed since the seed-12 probe, and this measures all three on
# the same six coupled cells:
#   * the continuation: a turn ending with no tool call no longer ends the run.
#     Four of six seed-12 runs died that way, mid-sentence, clock unspent.
#   * the structural delivery: participants land in <work>/coupling as a side
#     effect of the knowledge call agents actually make. (In the seed-12 build
#     it wrote to the server's cwd and would never have reached the agent.)
#   * the flux recovery: all 12 participants now return the interface flux at
#     order ~2 instead of not converging.
#
# WATCH: runs that write RESULT.txt at all (seed 12: 2 of 6); the continuation
# count in each ledger; participant files present in the work dir.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OPENPASO_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OPENPASO_REPO=/home/alexander/Schreibtisch/ofa-v2
LOG=campaign3_blind/probe2.log
PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python

if [ "$(stat -c %A "$OPENPASO_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OPENPASO_BLIND_KEYS"))" >> "$LOG"; exit 1
fi

SEED=13
# mapfile + </dev/null: a backgrounded child inside `while read ... done < list`
# inherits the loop's stdin and eats the list (footgun #7, cost 15 of 17 cells).
mapfile -t CELLS <<< "C2
C7
C8
C9
C10
C11"

echo ">>> probe 2 start $(date '+%F %T')  seed=$SEED  cells=${#CELLS[@]}" >> "$LOG"
for c in "${CELLS[@]}"; do
  ( "$PY" campaign3_blind/run_blind.py --model 27b --conditions MCP \
        --problems "$c" --seed $SEED --phase development >> "$LOG" 2>&1
    echo "<<< cell $c done $(date '+%F %T')" >> "$LOG" ) < /dev/null &
done
wait
echo ">>> probe 2 complete $(date '+%F %T')" >> "$LOG"
