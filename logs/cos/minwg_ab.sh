#!/bin/bash
cd ~/strix-rt; echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
for v in X=1 PF_MINWG=100000000 X=1 PF_MINWG=100000000; do echo "== $v"; env $v timeout 60 target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower 2>&1 | grep -E "step gpu" | head -2; done
V="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage vae --lat logs/cos/g256_run.lat.f32 --vref $HOME/cosmos_ref/v256run --out /tmp/vab"
for v in X=1 PF_MINWG=100000000 X=1 PF_MINWG=100000000; do echo "== vae $v"; env $v timeout 60 $V 2>&1 | grep -E "vae decode|PSNR" | head -2 | cut -c1-100; done
