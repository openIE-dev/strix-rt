#!/bin/bash
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
echo high | sudo -n tee $P >/dev/null 2>&1 || echo high > $P 2>/dev/null; cat $P
for s in 1234 1235 1236; do timeout 900 target-ye/release/yue2 gen --ref ~/yue_ref/out_ref --seed $s --maxsem 750 --out logs/ye/gen_$s.wav; done
echo auto | sudo -n tee $P >/dev/null 2>&1 || echo auto > $P 2>/dev/null; cat $P
