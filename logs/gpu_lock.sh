#!/bin/bash
# Exclusive GPU lock for benchmarks and heavy GPU jobs (Vulkan, ROCm/HIP/PyTorch, llama.cpp, NPU+GPU runs).
#   logs/gpu_lock.sh CMD [ARGS...]   run CMD while holding /tmp/strix-gpu.lock (waits for the current holder)
#   logs/gpu_lock.sh --status        show who holds the lock (exit 0 = free, 1 = held)
# Env: GPU_LOCK_WAIT=S   give up after S seconds (exit 75); default: wait forever
#      GPU_LOCK_TAG=txt  label shown in --status (default: the command line)
# Re-entrant: children of a holder see STRIX_GPU_LOCK_HELD=1 and run directly (no self-deadlock).
# The lock is a kernel flock: it is released when the holder and all its children exit, even on crash or kill -9.
L=/tmp/strix-gpu.lock; O=/tmp/strix-gpu.lock.owner
if [ "$1" = --status ]; then
  if flock -n "$L" true 2>/dev/null; then echo "free"; exit 0; else echo "held: $(cat $O 2>/dev/null)"; exit 1; fi
fi
[ $# -gt 0 ] || { sed -n 2,9p "$0"; exit 2; }
if [ -n "$STRIX_GPU_LOCK_HELD" ]; then exec "$@"; fi
[ -e "$L" ] || { (umask 000; : > "$L"); }
exec 9>>"$L" || exit 1
t0=$(date +%s)
if ! flock -x ${GPU_LOCK_WAIT:+-w "$GPU_LOCK_WAIT"} 9; then echo "gpu_lock: timed out after ${GPU_LOCK_WAIT}s; held: $(cat $O 2>/dev/null)" >&2; exit 75; fi
(umask 000; echo "pid $$ user $(id -un) since $(date +%T) (waited $(( $(date +%s) - t0 ))s): ${GPU_LOCK_TAG:-$*}" > "$O")
export STRIX_GPU_LOCK_HELD=1 STRIX_GPU_LOCK_WAITED=$(( $(date +%s) - t0 ))
"$@"; rc=$?
: > "$O"; exit $rc
