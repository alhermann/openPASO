#!/usr/bin/env bash
# Start round 8 the moment the machine is free, and not before.
#
# The round driver refuses to run beside other agent runs on purpose: a niced
# tier-2 fixture once ran at 1226% CPU next to a round and inflated its
# wall-clock and its timeout count. C9 was mid-flight and healthy, so killing
# it would have thrown away spent credits and the information it is producing,
# for the sake of starting ~25 minutes sooner.
#
# Bracket trick in the pattern so this script's own cmdline cannot match it.
set -u
cd /home/alexander/Schreibtisch/ofa-v2/campaign3_blind
LOG=round8_rebuild.log
echo ">>> chain: waiting for the machine $(date '+%F %T')" >> "$LOG"
# `pgrep -fc PATTERN` PRINTS 0 AND EXITS 1 when nothing matches, so the usual
# `|| echo 0` fallback appends a SECOND zero: the value becomes "0\n0", which
# never equals "0". A wait loop written that way waits forever and a guard
# written that way refuses forever. This cost 36 minutes of an idle machine
# with the round sitting behind it. `pgrep -f ... | wc -l` always prints one
# number and always exits 0.
while [ "$(pgrep -f 'run_blind[.]py' 2>/dev/null | wc -l)" != "0" ]; do
  sleep 60
done
echo ">>> chain: machine free, starting round 8 $(date '+%F %T')" >> "$LOG"
exec ./round8_rebuild_driver.sh
