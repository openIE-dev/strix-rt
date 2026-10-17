#!/bin/bash
cd ~/strix-rt; B=target-fa2/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for fa in 0 1; do echo "== Q38 seed 5 L1024 FA=$fa $(PF_SEED=5 PF_FA=$fa PF_FORCED=1 REPS=1 timeout 400 $B/pf_q38 $Q 1024 32 2>&1 | grep -E "PF_FORCED|rror|panic")"; done; echo DONE
