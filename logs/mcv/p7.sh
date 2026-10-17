#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for k in none gdn_scan gdn_conv,gdn_convst gnorm rmsnorm; do echo "== skip $k"; PF_SKIPK=$k PF_TS=1 PREFILL=1 REPS=2 timeout 120 target-k6/release/mcv_ref $M ~/mcv_ref/ref_img 2 2>&1 | grep -E "TS" | tail -1; done
