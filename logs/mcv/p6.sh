#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for d in ref_img ref_chess; do echo "== $d"; PF_TS=1 PREFILL=1 REPS=2 timeout 120 target-k6/release/mcv_ref $M ~/mcv_ref/$d 2 2>&1 | grep -E "TS|prefill \(" | tail -2; done
