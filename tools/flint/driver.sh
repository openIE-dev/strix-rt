#!/bin/bash
# tools/flint/driver.sh "NAME REP [acc|energy]" ... (acc sessions skip the quiet gate) : detached (nohup) driver. Per session: wait for a quiet box, take the GPU lock (queues behind any holder),
# perf level high, run tools/flint/session.sh; on gate timeout (exit 3) wait and retry (up to 20 times).
cd ~/strix-rt; D=logs/flint; mkdir -p $D
for s in "$@"; do for try in $(seq 1 20); do
  case "$s" in *acc) ;; *) python3 logs/cpubusy.py gate 1.0 21600 >> $D/gate.txt || { echo "$(date +%T) no quiet window in 6 h" >> $D/session.log; exit 3; } ;; esac
  GPU_LOCK_TAG="flint $s" logs/gpu_lock.sh tools/comp/perf_high.sh bash tools/flint/session.sh $s; rc=$?
  echo "$(date +%T) driver $s rc=$rc try=$try" >> $D/session.log; [ $rc -ne 3 ] && break
done; done
echo "$(date +%T) driver all done" >> $D/session.log
