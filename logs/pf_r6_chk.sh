#!/bin/sh
# Round 6: batched prefill vs token-by-token decode at L=2048 / 4096 (one chunk, PF_M=L) for all models, after the dispatch fix
cd ~/strix-rt; . logs/pf_busy.sh
Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; P2=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; P1=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
for l in 2048 4096; do for mb in "lfm2 pf_lfm $L" "q38 pf_q38 $Q" "pq2 pf_q38 $P2" "ptq1 pf_q38 $P1"; do set -- $mb; waitidle
  echo "$1 L=$l: $(PF_M=$l REPS=1 ./target/release/$2 $3 $l 8 2>&1 | grep -E "cosine|argmax|greedy|rror|panic" | tr "\n" " ")"; done; done
echo "[end $(date +%T)]"
