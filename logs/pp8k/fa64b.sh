#!/bin/bash
# LFM FA64 acceptance: teacher-forced agreement FA64=0 vs 1 on 5 real-text prompts (L=768), within 1 token
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
for i in 1 2 3 4 5; do for fa in 0 1; do echo "== p$i FA64=$fa $(PF_TOKS=logs/pp8k/lfm_p$i.txt PF_FA64=$fa PF_FORCED=1 REPS=1 timeout 60 $B/pf_lfm $L 768 32 2>&1 | grep -E "PF_FORCED|rror|panic")"; done; done; echo DONE
