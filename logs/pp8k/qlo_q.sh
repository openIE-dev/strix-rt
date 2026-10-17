#!/bin/bash
# PF_FA_QLO=0 acceptance, Qwen3.8 seeds (FA=0 baseline 32/32 each); arg: list of "seed L"
cd ~/strix-rt; B=target-fa3/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for c in "$@"; do set -- $c; echo "== Q38 seed $1 L$2 QLO=0 $(PF_SEED=$1 PF_FA=1 PF_FA_QLO=0 PF_FORCED=1 REPS=1 timeout ${QT:-150} $B/pf_q38 $Q $2 32 2>&1 | grep -E "PF_FORCED|rror|panic")"; done; echo DONE
