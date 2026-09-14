#!/bin/bash
# Development-phase driver: the 32 new cells, both arms, 27B, seed 1.
#
# INTERLEAVED ARMS: BARE then MCP back-to-back per cell, so background load
# hits both arms alike (the same fairness rule the previous campaign's
# ab_driver used). Throttled to 3 concurrent cells. Default per-run timeout —
# the pre-registered cap — no shortcuts.
#
# Development phase: results feed post-mortems and convergence checks, never
# the paper's table. Every run dir and ledger is kept.
set -u
cd /home/alexander/Schreibtisch/ofa-v2
export OPENPASO_REPO=/home/alexander/Schreibtisch/ofa-v2
export OPENPASO_BLIND_KEYS=/home/alexander/Schreibtisch/qwen_uplift_test/campaign3_blind/keys
set -a; . /home/alexander/Schreibtisch/qwen_uplift_test/.env; set +a
export OPENPASO_PYTHON=/home/alexander/Schreibtisch/open-fem-agent/.venv/bin/python
# NO X11. DISPLAY=:1 with a stale ~/.Xauthority made every graphical-capable
# solver emit "Invalid MIT-MAGIC-COOKIE-1 key" — 60 such lines across round 1,
# and C1 concluded in BOTH arms that "4C requires an X11 display" and treated
# it as fatal. These are batch solves; there is no display to want.
unset DISPLAY XAUTHORITY
export MPLBACKEND=Agg
export QT_QPA_PLATFORM=offscreen
P=/home/alexander/Schreibtisch/open-fem-agent/.venv-lg/bin/python
LOG=/tmp/claude-1001/-home-alexander-4C/dev27b.log
CELLS="FE1 FE2 DL1 DL2 NG1 NG2 SK1 SK2 KR1 KR2 DU1 DU2 FB1 FB2 FC1 FC2 SP1 SP2 C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 C14"

echo ">>> development 27b start $(date '+%F %T')" >> "$LOG"
throttle() { while [ "$(jobs -rp | wc -l)" -ge 3 ]; do sleep 30; done; }

pair() {  # one cell: BARE then MCP, back to back
  local c=$1
  $P campaign3_blind/run_blind.py --model 27b --conditions BARE --problems "$c" \
      --seed 1 --phase development >> "$LOG" 2>&1
  $P campaign3_blind/run_blind.py --model 27b --conditions MCP  --problems "$c" \
      --seed 1 --phase development >> "$LOG" 2>&1
  echo "<<< cell $c done $(date '+%F %T')" >> "$LOG"
}

for c in $CELLS; do
  throttle
  pair "$c" &
done
wait
echo ">>> development 27b complete $(date '+%F %T')" >> "$LOG"
grep -cE "^\[.*\] done" "$LOG" | xargs -I{} echo "runs logged: {}" >> "$LOG"
