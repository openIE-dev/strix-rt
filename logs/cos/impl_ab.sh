#!/bin/bash
cd ~/strix-rt; echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
V="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage vae --lat logs/cos/g256_run.lat.f32 --vref $HOME/cosmos_ref/v256run --out /tmp/vab"
for v in COS_IMPL=0 COS_IMPL=1 COS_IMPL=1 "COS_IMPL=1 COS_VAE_CHUNK=2" "COS_IMPL=0 COS_VAE_CHUNK=2"; do echo "== $v"; env $v timeout 60 $V 2>&1 | grep -E "vae decode|PSNR|rror|panic" | head -3 | cut -c1-130; done
