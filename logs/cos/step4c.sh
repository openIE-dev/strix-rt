#!/bin/sh
# Step 4c: same per-layer A/B under perf level high (as bench1), default vs SAMEL
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
cat /sys/class/drm/card*/device/power_dpm_force_performance_level
for v in "COS_LTIME=1" "X=1" "COS_SAMEL=1"; do echo "== $v"; env $v timeout 70 $B 2>&1 | grep -E "step gpu|layer ms" | head -4; done
