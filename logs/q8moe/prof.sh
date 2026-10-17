#!/bin/bash
cd ~/strix-rt; B=target-fa2/release; M=$MODELS/bench/Qwen3.6-35B-A3B-Q8_0.gguf
for L in L512 L8192; do echo "== $L"; PF_FA=1 PF_TS=1 PREFILL=1 REPS=2 MAX_CTX=8400 timeout 150 $B/mcv_ref $M logs/comp/mcvref/$L 2 2>&1 | grep -E "TS ms|median|rror" | tail -3; done; echo DONE
