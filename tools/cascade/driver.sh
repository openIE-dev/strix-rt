#!/bin/bash
# tools/cascade/driver.sh KIND ARGS : detached driver (nohup). Energy sessions wait for a quiet box first (accuracy sessions do not need it);
# takes the GPU lock (perf level high, restored on exit), runs one session; if the in-session gate times out (exit 3) it waits for quiet again
# and retries (up to 20 times).
cd ~/strix-rt; D=logs/cascade; mkdir -p $D
for try in $(seq 1 20); do
  case $1 in acc|acc2) ;; *) python3 logs/cpubusy.py gate 1.0 21600 >> $D/gate.txt || { echo "$(date +%T) no quiet window in 6 h" >> $D/session.log; exit 3; } ;; esac
  GPU_LOCK_TAG="cascade $*" logs/gpu_lock.sh tools/comp/perf_high.sh bash tools/cascade/session.sh "$@"; rc=$?
  [ $rc -ne 3 ] && { echo "$(date +%T) driver $* rc=$rc" >> $D/session.log; exit $rc; }
  echo "$(date +%T) driver $* retry after gate timeout" >> $D/session.log
done
