#!/bin/bash
# 720p session: perf level high, 5f then 9f (chunked VAE), 1 step each (functional check of memory/submission limits)
cd ~/strix-rt
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
for f in ${FR:-5 9}; do echo "== frames $f"; timeout 200 target-cos/release/cos gen --size 704x1280 --frames $f --steps 1 --out logs/cos/s720_$f 2>&1 | grep -v warm | tail -9; done
