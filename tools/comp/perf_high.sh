#!/bin/bash
# tools/comp/perf_high.sh CMD..: run CMD with the GPU perf level forced high, restore auto on exit. Use inside gpu_lock.sh.
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
trap "echo auto | sudo tee $P >/dev/null" EXIT
echo high | sudo tee $P >/dev/null; echo "perf level: $(cat $P)"
"$@"
