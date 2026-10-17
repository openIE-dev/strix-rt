# sourced by GPU benchmark scripts. Now backed by the exclusive GPU lock (logs/gpu_lock.sh, /tmp/strix-gpu.lock):
# sourcing this from a script re-executes that script under the lock, so the whole script runs exclusively.
# busy() / waitidle() are kept for compatibility: inside the lock nobody else can run, so busy() is false.
if [ -z "$STRIX_GPU_LOCK_HELD" ]; then
  case "$0" in *.sh) if [ -x "$0" ]; then exec ~/strix-rt/logs/gpu_lock.sh "$0" "$@"; fi
    exec ~/strix-rt/logs/gpu_lock.sh "${BASH_VERSION:+bash}${BASH_VERSION:-sh}" "$0" "$@" ;; esac
fi
busy() { [ -z "$STRIX_GPU_LOCK_HELD" ] && ! flock -n /tmp/strix-gpu.lock true 2>/dev/null; }
waitidle() { while busy; do sleep 3; done; }
