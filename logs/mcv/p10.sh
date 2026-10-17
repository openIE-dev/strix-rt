#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for c in "4 1" "4 2" "4 4" "8 2" "8 4"; do set -- $c; echo "T=$1 RR=$2"; for d in ref_img ref_gold; do PF_GMV_T=$1 PF_GMV_RR=$2 PF_GMV_M=512 PF_TS=1 PREFILL=1 REPS=2 timeout 60 target-k6/release/mcv_ref $M ~/mcv_ref/$d 100 2>&1 | grep -E "TS|MATCH|rror|panic" | tail -2 | sed "s/^/$d /"; done; done
