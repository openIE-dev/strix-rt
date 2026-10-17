#!/bin/bash
# Qwen3.6 Q8 pp2048/pp8192: PF_FA_QLO 1 vs 0 (alternating), + MiniCPM L512 forced agreement with QLO=0
cd ~/strix-rt; B=target-fa3/release; M=$MODELS/bench/Qwen3.6-35B-A3B-Q8_0.gguf
for L in L2048 L8192; do for q in 1 0 1 0; do echo "== q36 $L QLO=$q $(PF_FA_QLO=$q PREFILL=1 REPS=2 MAX_CTX=8400 timeout 120 $B/mcv_ref $M logs/comp/mcvref/$L 2 2>&1 | grep -oE "median.*tok/s")"; done; done
echo DONE
