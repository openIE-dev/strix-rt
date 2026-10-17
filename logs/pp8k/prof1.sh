#!/bin/bash
# pp8192 phase profile: MiniCPM (mcv_ref PF_TS=1) chunked GDN off/on
cd ~/strix-rt; B=target-comp/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for ch in 0 1; do echo "== PF_GDN_CHUNK=$ch"; PF_GDN_CHUNK=$ch PF_TS=1 PREFILL=1 REPS=2 MAX_CTX=8400 timeout 200 $B/mcv_ref $M logs/comp/mcvref/L8192 2 2>&1 | grep -E "TS ms|prefill|rror|panic"; done
echo DONE
