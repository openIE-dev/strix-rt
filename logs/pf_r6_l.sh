#!/bin/sh
# Round 6: where does lfm2 prefill vs token-by-token diverge? L sweep and chunk size (PF_M)
cd ~/strix-rt; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; . logs/pf_busy.sh
for c in "1024 1024" "1536 1536" "2048 512" "2048 1024" "2048 2048"; do set -- $c; waitidle
  echo "L=$1 PF_M=$2: $(PF_M=$2 REPS=1 ./target/release/pf_lfm $L $1 4 2>&1 | grep -E "cosine|argmax|greedy" | tr "\n" " ")"; done
