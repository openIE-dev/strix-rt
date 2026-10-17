#!/bin/bash
# VAE GEMM tile A/B at 256x256x9 (perf high); parity vs CPU fp32 ref each
cd ~/strix-rt; echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage vae --lat logs/cos/g256_run.lat.f32 --vref $HOME/cosmos_ref/v256run --out /tmp/vab"
for v in X=1 PF_MINWG=100000000 PF_SG=64 PF_MTILE=0 "PF_MINWG=100000000 PF_SG=64" X=1; do echo "== $v"; env $v timeout 60 $B 2>&1 | grep -E "vae decode|vae_ref" | cut -c1-120; done
