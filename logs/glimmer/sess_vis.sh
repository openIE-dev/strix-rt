#!/bin/bash
# one lock session: our ViT on test (512x512) and test_big (1400x980) vs llama.cpp mtmd dumps (ref_*.f32, made by mtmd_dump), cosine
cd ~/strix-rt; D=$MODELS/new/glimmer; V=logs/glimmer/mtmd
for im in test test_big; do
  timeout 120 target-glm/release/glm_vis $D/mmproj-Muse-Glimmer-30B-Q4_K_M.gguf $V/$im.rgb $V/our_$im.f32 $V/ref_$im.f32 2>&1 | grep -E "^run|COSINE|rror|panic"
done
