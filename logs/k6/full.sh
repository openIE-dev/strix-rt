#!/bin/bash
cd ~/strix-rt; B=target-k6/release/k6; O=~/k6_ref/full
echo "== vae ref latents"; $B vae --lat ~/k6_ref/out/latents.npy --out $O/ref_vid.npy
echo "== vae ours latents"; $B vae --lat ~/k6_ref/ours2/latents.npy --out $O/ours_vid.npy
echo "== aud ours latents"; mkdir -p $O/aud; $B aud --alat ~/k6_ref/ours2/audio_latents.npy --aref $O/aud
