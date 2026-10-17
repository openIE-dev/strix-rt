#!/bin/bash
# Step 5: 720p clip (704x1280 = 44x80 latent, 3520 px/frame at the VAE mid attention -> tiled cos_vattn_t), 5 frames, 12 steps (9 frames: VAE buffers 4.15 GB each > 4 GB, device lost).
# Guard: only if the tiled-attention parity run (logs/cos/vattn_t.log) reached >= 60 dB vs the CPU fp32 VAE reference.
cd ~/strix-rt
db=$(grep -o "vae_ref \[[^]]*\]: PSNR [0-9.]*" logs/cos/vattn_t.log | awk '{print $NF}')
if [ -z "$db" ] || [ "$(echo "$db < 60" | bc)" = 1 ]; then echo "skip: vattn_t parity '$db' dB"; exit 1; fi
echo "vattn_t parity $db dB"
free -g | head -2
timeout 285 target-cos/release/cos gen --prompt "A red vintage car drives along a coastal cliff road at sunset, cinematic aerial shot" --size 704x1280 --frames 5 --steps 12 --out logs/cos/gen720
