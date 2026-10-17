#!/bin/bash
# MiniCPM L512: which first pred is right? stepwise reference vs batched prefill FA=0/1
cd ~/strix-rt; B=target-fa/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
echo "== stepwise"; timeout 120 $B/mcv_ref $M logs/comp/mcvref/L512 8 2>&1 | grep -E "OURS|rror|panic"
for fa in 0 1; do for pm in 512 128; do echo "== FA=$fa PF_M=$pm"; PF_FA=$fa PF_M=$pm PREFILL=1 timeout 60 $B/mcv_ref $M logs/comp/mcvref/L512 8 2>&1 | grep -E "OURS|first pred|rror|panic"; done; done
echo DONE
