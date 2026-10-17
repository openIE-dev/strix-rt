#!/bin/bash
# one lock session: long-prompt prefill parity (chunk 256 and 512) vs llama.cpp
cd ~/strix-rt; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
for ch in 256 512; do for i in 0 1; do
  echo "== L$i chunk $ch"; MAX_CTX=8192 PF_CHUNK=$ch timeout 120 target-glm/release/glm_run $M @logs/glimmer/p/L$i.ids 48 @logs/glimmer/p/L$i.ref 2>&1 | grep -E "parity|prompt|rror|panic"
done; done
