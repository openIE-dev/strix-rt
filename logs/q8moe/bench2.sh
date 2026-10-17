#!/bin/bash
# Qwen3.6-35B-A3B Q8_0: pp512/2048/8192 (batched prefill, FA on) + decode at depth (n_gen 64 after the prompt)
cd ~/strix-rt; B=target-fa2/release; M=$MODELS/bench/Qwen3.6-35B-A3B-Q8_0.gguf
for L in L512 L2048 L8192; do echo "== $L $(PF_FA=1 PREFILL=1 REPS=2 MAX_CTX=8400 timeout 150 $B/mcv_ref $M logs/comp/mcvref/$L 64 2>&1 | grep -E "median|decode|rror|panic" | tr "\n" " " | cut -c1-400)"; done
echo DONE
