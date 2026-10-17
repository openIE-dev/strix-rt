#!/bin/bash
# Step 5 parity: tiled VAE attention forced on at 256x256 vs CPU fp32 diffusers VAE ref (naive path gave 85.25 dB).
cd ~/strix-rt
COS_VATTN_T=1 timeout 120 target-cos/release/cos --golden ~/cosmos_ref/g256 --stage vae --lat logs/cos/g256_run.lat.f32 --vref ~/cosmos_ref/v256run --out logs/cos/vattn_t
