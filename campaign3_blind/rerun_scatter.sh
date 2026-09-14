#!/bin/bash
# Re-run the 19 scatter-contaminated development runs (2026-08-15).
#
# These runs wrote deliverables OUTSIDE the sandbox (to /tmp or $HOME), so the
# grader — which by contract reads only work/ — saw nothing. The originals are
# quarantined WITH their out-of-sandbox evidence preserved; these re-runs use
# the confined write_file (writes outside the sandbox are refused with a
# corrective message, so the model self-corrects instead of scattering).
# 13 coupled MCP + 4 coupled BARE + 2 single BARE. Same model, seed, phase.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OPENPASO_REPO=/home/alexander/Schreibtisch/ofa-v2
export OPENPASO_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OPENPASO_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
P=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b.log
RUNS="C1:MCP C2:MCP C3:MCP C4:MCP C5:MCP C6:MCP C7:MCP C8:MCP C9:MCP C10:MCP C11:MCP C13:MCP C14:MCP C4:BARE C5:BARE C7:BARE C13:BARE FC2:BARE SP1:BARE"

echo ">>> scatter rerun start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 30; done; }
for r in $RUNS; do
  throttle
  c=${r%%:*}; a=${r##*:}
  $P campaign3_blind/run_blind.py --model 27b --conditions "$a" --problems "$c" \
      --seed 1 --phase development >> "$LOG" 2>&1 &
done
wait
echo ">>> scatter rerun complete $(date '+%F %T')" >> "$LOG"
