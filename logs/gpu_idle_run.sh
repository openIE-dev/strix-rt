# usage: gpu_idle_run.sh LOG ENV... CMD ARGS... : runs `env ENV... CMD ARGS...` holding the exclusive GPU lock
# (logs/gpu_lock.sh, /tmp/strix-gpu.lock), output to LOG. Replaces the old pgrep-based idle wait.
cd ~/strix-rt
log=$1; shift
exec logs/gpu_lock.sh sh -c 'log=$1; shift; echo "[waited ${STRIX_GPU_LOCK_WAITED:-0}s for GPU lock; start $(date +%T)]" > $log
env "$@" >> $log 2>&1; rc=$?
echo "[end $(date +%T); exit $rc; ran under GPU lock]" >> $log; exit $rc' sh "$log" "$@"
