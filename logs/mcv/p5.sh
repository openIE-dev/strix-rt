#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for d in ref_img ref_chess; do for r in 0 64 32 16; do echo "== $d R=$r"; PF_GSCAN_R=$r PREFILL=1 REPS=3 timeout 120 target-k6/release/mcv_ref $M ~/mcv_ref/$d 100 2>&1 | grep -E "prefill \(|MATCH"; done; done
