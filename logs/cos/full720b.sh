#!/bin/bash
# 720p 9f 12 steps: gen (latent only), then VAE in its own process, same lock session, perf high
cd ~/strix-rt; echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null; trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level >/dev/null" EXIT
COS_NO_VAE=1 timeout 200 target-cos/release/cos gen --prompt "A red vintage car drives along a coastal cliff road at sunset, cinematic aerial shot" --size 704x1280 --frames 9 --steps 12 --out logs/cos/full720 2>&1 | grep --line-buffered -v warm
timeout 100 target-cos/release/cos vae --lat logs/cos/full720.lat.f32 --size 704x1280 --frames 9 --out logs/cos/full720 2>&1 | grep --line-buffered -v warm
