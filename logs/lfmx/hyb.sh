#!/bin/bash
# PF_HYB=1 acceptance: teacher-forced L768 32 tok, HYB 0 vs 1; args = prompt numbers
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
for i in "$@"; do for h in 0 1; do echo "== p$i HYB=$h"; PF_TOKS=logs/pp8k/lfm_p$i.txt PF_HYB=$h PF_FORCED=1 REPS=1 timeout 60 $B/pf_lfm $L 768 32 2>&1 | grep -E "PF_FORCED|mismatch [0-9]|rror|panic"; done; done; echo DONE
