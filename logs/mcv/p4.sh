#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for d in ref_img ref_gold ref_chess ref_txt; do echo "== $d"; PREFILL=1 REPS=5 timeout 120 target-k6/release/mcv_ref $M ~/mcv_ref/$d 100 2>&1 | grep -E "prefill \(|MATCH"; done
timeout 600 ~/llama.cpp/build-vulkan/bin/llama-bench -m $M -ngl 99 -fa 1 -p 23,91,226,482 -n 0 -r 3 2>&1 | grep pp
