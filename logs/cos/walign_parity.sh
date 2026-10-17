#!/bin/bash
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tf"
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
for v in QI_WALIGN=0 QI_WALIGN=1; do echo "== $v"; env $v timeout 120 $B 2>&1 | grep -vE "warm" | tail -22 | cut -c1-200; done
