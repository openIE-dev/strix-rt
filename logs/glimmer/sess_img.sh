#!/bin/bash
# one lock session: vision-prompt parity (368 tokens: BOS, image_start, 361 image rows, image_end, newline, "This image shows") vs ref_llama_img.json
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf; V=logs/glimmer/mtmd
for emb in ref_test our_test; do for ch in 256 512; do
  echo "== embeddings $emb chunk $ch"
  IMG_EMB=$V/$emb.f32 IMG_POS=2 PF_CHUNK=$ch FORCED=1 timeout 100 target-glm/release/glm_run $M @logs/glimmer/p/img.ids 128 @logs/glimmer/p/img.ref 2>&1 | grep -E "FORCED|rror"
  IMG_EMB=$V/$emb.f32 IMG_POS=2 PF_CHUNK=$ch timeout 100 target-glm/release/glm_run $M @logs/glimmer/p/img.ids 128 @logs/glimmer/p/img.ref 2>&1 | grep -E "parity|prompt|rror"
done; done
