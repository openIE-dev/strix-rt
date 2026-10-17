#!/bin/bash
# FA64 acceptance: args = "prompt FA64 QLO" triples; teacher-forced L768 32 tok
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
for c in "$@"; do set -- $c; echo "== p$1 FA64=$2 QLO=$3"; PF_TOKS=logs/pp8k/lfm_p$1.txt PF_FA64=$2 PF_FA_QLO=$3 PF_FORCED=1 REPS=1 timeout 60 $B/pf_lfm $L 768 32 2>&1 | grep -E "PF_FORCED|mismatch [0-9]|rror|panic"; done; echo DONE
