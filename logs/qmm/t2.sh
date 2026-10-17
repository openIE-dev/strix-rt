#!/bin/bash
# LFM PF_QMM=1 smoke: logits vs ref, speed (alternating, cool-down), forced agreement on 2 prompts
cd ~/strix-rt; B=target-fa3/release; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf
for q in 1 0 1; do echo "== QMM=$q L512"; PF_QMM=$q REPS=3 timeout 60 $B/pf_lfm $L 512 4 2>&1 | grep -E "logits|pp512|rror|panic"; sleep 10; done
for i in 1 2; do echo "== p$i QMM=1"; PF_TOKS=logs/pp8k/lfm_p$i.txt PF_QMM=1 PF_FORCED=1 REPS=1 timeout 60 $B/pf_lfm $L 768 32 2>&1 | grep -E "PF_FORCED|mismatch [0-9]|rror|panic"; done; echo DONE
