#!/bin/bash
cd ~/strix-rt
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
for v in "X=1" "COS_ASK_REPREFILL=1"; do echo "== $v"; env $v timeout 100 target-cos/release/cos ask --ref ~/cosmos_ref/vit1 --question "Describe this image in one sentence." --new 32 2>&1 | grep -E "decode:|logits0|greedy|^answer|ref answer"; done
