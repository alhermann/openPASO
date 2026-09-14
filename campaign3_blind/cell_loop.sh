#!/usr/bin/env bash
# ONE CELL, THREE SEEDS — the development loop, run per problem.
#
# The 128-run round is the wrong instrument for finding out WHY something
# fails. It costs ~130 credits and hours, mixes every cause together, and one
# candidate fix takes a whole round to test. This runs a single cell on three
# seeds (~4 credits, ~20 min), which is enough to tell a systematic failure
# from a one-off, and cheap enough to repeat after every fix.
#
# The loop it belongs to:
#   1. run this          2. read all three runs, find the ROOT cause
#   3. distil a GENERAL primitive, verify it by executing it locally
#   4. run this again on the SAME seeds
#   5. new failure -> repeat; nothing new -> this cell is done
#
# Then re-run two or three ALREADY-CONVERGED cells. A fix that only helps the
# cell it came from was not a primitive, and per-cell work is precisely how
# that mistake gets made. The paper's rule is that knowledge must be general;
# this script makes it cheap to keep checking that it is.
#
# It does NOT replace the freeze gate. Freezing still needs one full round that
# teaches nothing new — this just gets us there without buying a round per fix.
#
# usage: ./cell_loop.sh C1 [seedA seedB seedC] [ARM]
set -u
CELL="${1:?usage: cell_loop.sh CELL [s1 s2 s3] [ARM]}"
S1="${2:-21}"; S2="${3:-22}"; S3="${4:-23}"
ARM="${5:-MCP}"

cd /home/alexander/Schreibtisch/ofa-v2
export OPENPASO_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OPENPASO_REPO=/home/alexander/Schreibtisch/ofa-v2
PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG="campaign3_blind/cell_${CELL}_${ARM}.log"

if [ "$(stat -c %A "$OPENPASO_BLIND_KEYS")" != "d---------" ]; then
  echo "REFUSING: answer keys are not sealed ($(stat -c %A "$OPENPASO_BLIND_KEYS"))" | tee -a "$LOG"
  exit 1
fi

echo ">>> $CELL $ARM seeds $S1 $S2 $S3  start $(date '+%F %T')" >> "$LOG"
# mapfile + </dev/null: a backgrounded child inside `while read ... done < list`
# inherits the loop's stdin and eats the list — that cost 15 of 17 cells once.
mapfile -t SEEDS <<< "$S1
$S2
$S3"
for s in "${SEEDS[@]}"; do
  ( "$PY" campaign3_blind/run_blind.py --model 27b --conditions "$ARM" \
        --problems "$CELL" --seed "$s" --phase development >> "$LOG" 2>&1
    echo "<<< $CELL seed $s done $(date '+%F %T')" >> "$LOG" ) < /dev/null &
done
wait
echo ">>> $CELL $ARM complete $(date '+%F %T')" >> "$LOG"
