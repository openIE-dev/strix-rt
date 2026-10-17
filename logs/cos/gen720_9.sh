#!/bin/bash
# 720p 9 frames with the temporal-chunk VAE (auto chunk 2 above 3.5 GB buffers); guarded by logs/cos/vchunk.log parity
cd ~/strix-rt
db=$(grep -o "PSNR [0-9.]*" logs/cos/vchunk.log | head -1 | awk '{print $2}')
if [ -z "$db" ] || [ "$(echo "$db < 60" | bc)" = 1 ]; then echo "skip: chunk parity '$db'"; exit 1; fi
echo "chunk parity $db dB"
timeout 285 target-cos/release/cos gen --prompt "A red vintage car drives along a coastal cliff road at sunset, cinematic aerial shot" --size 704x1280 --frames 9 --steps 4 --out logs/cos/gen720_9
