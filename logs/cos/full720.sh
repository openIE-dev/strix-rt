#!/bin/bash
# 720p 9 frames, 12 steps, one session (perf high)
cd ~/strix-rt; echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
timeout 280 target-cos/release/cos gen --prompt "A red vintage car drives along a coastal cliff road at sunset, cinematic aerial shot" --size 704x1280 --frames 9 --steps 12 --out logs/cos/full720 2>&1 2>&1 | grep --line-buffered -v warm
