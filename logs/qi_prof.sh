#!/bin/bash
# usage: logs/qi_prof.sh [env...]  (per-op profile at 1024, perf high then auto)
D=/sys/class/drm/card1/device; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
cd ~/strix-rt; env "$@" target/release/qi --size 1024 --stage prof 2>&1 | grep -v "warm"
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; sudo dmesg -T | grep timeout | tail -1 | cut -c1-30
