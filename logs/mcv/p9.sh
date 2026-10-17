#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for s in 1 2 4; do echo "== SPLITK=$s"; PF_SPLITK=$s PF_TS=1 PREFILL=1 REPS=4 timeout 120 target-k6/release/mcv_ref $M ~/mcv_ref/ref_img 100 2>&1 | grep -E "TS|prefill \(|MATCH|rror|panic" | tail -3; done
