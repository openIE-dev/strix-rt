#!/bin/bash
# clean-GPU pp8192: FA=0 vs FA=1, alternating, MiniCPM + Qwen3.8
cd ~/strix-rt; B=target-fa2/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for r in 1 2; do for fa in 0 1; do echo "== MiniCPM FA=$fa r$r $(PF_FA=$fa PREFILL=1 REPS=2 MAX_CTX=8400 timeout 120 $B/mcv_ref $M logs/comp/mcvref/L8192 2 2>&1 | grep -oE "median.*tok/s")"; done; done
for fa in 0 1 0 1; do echo "== Q38 FA=$fa $(PF_FA=$fa NOREF=1 REPS=2 timeout 150 $B/pf_q38 $Q 8192 2>&1 | grep -oE "best.*")"; done
echo DONE
