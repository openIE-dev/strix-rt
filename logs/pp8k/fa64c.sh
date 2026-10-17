#!/bin/bash
# FA64 follow-up: margins at p2 mismatches (QLO=1 as in fa64b), then FA64=1 with QLO=0 on 5 prompts
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
r() { PF_TOKS=logs/pp8k/lfm_p$1.txt PF_FORCED=1 REPS=1 timeout 60 $B/pf_lfm $L 768 32 2>&1 | grep -E "PF_FORCED|mismatch [0-9]|rror|panic"; }
echo "== p2 FA64=1 QLO=1"; PF_FA64=1 PF_FA_QLO=1 r 2
echo "== p2 FA64=0"; PF_FA64=0 r 2
for i in 1 2 3 4 5; do echo "== p$i FA64=1 QLO=0"; PF_FA64=1 PF_FA_QLO=0 r $i; done; echo DONE
