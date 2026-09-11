#!/bin/bash
# Honest-build re-run of the three coupled cells that have never passed (C3, C1, C2), fresh seeds.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OASIS_REPO=/home/alexander/Schreibtisch/ofa-v2
export OASIS_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
export OASIS_BLIND_PROBLEMS=/home/alexander/Schreibtisch/ofa-v2/campaign3_blind/problems
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OASIS_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
export PATH="/home/alexander/FEBio/bin:$PATH"
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
[ "$(stat -c %A "$OASIS_BLIND_KEYS")" = "d---------" ] || { echo "REFUSING: keys not sealed"; exit 1; }
PY=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/b1c8e459-ec06-467a-bad7-474c74f9d0f3/scratchpad/rerun_honest17.log
echo ">>> honest re-run (C3 5731-33, C1 6541-43, C2 6551-53) $(date '+%F %T')" >> "$LOG"
for job in "C3 5731" "C3 5732" "C3 5733" "C1 6541" "C1 6542" "C1 6543" "C2 6551" "C2 6552" "C2 6553"; do
  set -- $job
  ( $PY campaign3_blind/run_blind.py --model 27b --conditions MCP \
        --problems $1 --seed $2 --phase development >> "$LOG" 2>&1
    echo "<<< $1 MCP seed $2 done $(date '+%F %T')" >> "$LOG" ) < /dev/null &
done
wait
echo ">>> all nine done $(date '+%F %T')" >> "$LOG"
