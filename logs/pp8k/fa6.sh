#!/bin/bash
cd ~/strix-rt; B=target-fa/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for c in "16 2 512 0" "16 2 512 512"; do echo "== $c"; timeout 20 $B/fa_check $c 2>&1 | tail -3; done
for pm in 512 256; do echo "== MiniCPM L512 FA=1 PF_M=$pm"; PF_FA=1 PF_M=$pm PREFILL=1 timeout 60 $B/mcv_ref $M logs/comp/mcvref/L512 8 2>&1 | grep -E "OURS|first pred|rror|panic"; done
echo "== MiniCPM L8192 FA=1"; PF_FA=1 PREFILL=1 REPS=2 MAX_CTX=8400 timeout 120 $B/mcv_ref $M logs/comp/mcvref/L8192 8 2>&1 | grep -E "OURS|first pred|rror|panic"
echo DONE
