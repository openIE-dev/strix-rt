#!/bin/bash
# Temporal-chunk VAE parity at 256x256x9 (t=3) vs the CPU fp32 diffusers reference, then the 9-frame 720p clip.
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage vae --lat logs/cos/g256_run.lat.f32 --vref $HOME/cosmos_ref/v256run"
for c in 0 1 2; do echo "== COS_VAE_CHUNK=$c"; COS_VAE_CHUNK=$c timeout 60 $B --out logs/cos/vchunk$c 2>&1 | grep -E "vae decode|vae_ref|rror|panic"; done
