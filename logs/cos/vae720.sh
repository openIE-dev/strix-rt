#!/bin/bash
# VAE-only 720p decode of a saved latent: logs/cos/vae720.sh LAT FRAMES OUT
cd ~/strix-rt; echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
COS_VAE_TRACE=${TRACE:-} timeout 120 target-cos/release/cos vae --lat $1 --size 704x1280 --frames $2 --out $3 2>&1 | grep --line-buffered -v warm | tail -${TAILN:-40}
