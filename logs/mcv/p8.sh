#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for t in 0 2048 1024 512 8192; do echo "== SMT=$t"; PF_SMT_M=4096 PF_SMT=$t PF_TS=1 PREFILL=1 REPS=2 timeout 120 target-k6/release/mcv_ref $M ~/mcv_ref/ref_img 40 2>&1 | grep -E "TS|prefill \(|MATCH" | tail -3; done
