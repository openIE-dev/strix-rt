#!/bin/sh
cd ~/strix-rt; COS_VAE_TRACE=1 COS_VAE_CHUNK=1 timeout 30 target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage vae --lat logs/cos/g256_run.lat.f32 --vref $HOME/cosmos_ref/v256run --out logs/cos/vchunk1 2>&1 | grep -E "vae part|PSNR|rror|panic" | tail -6
