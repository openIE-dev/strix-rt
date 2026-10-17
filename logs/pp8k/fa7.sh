#!/bin/bash
cd ~/strix-rt; B=target-fa/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
echo "== MiniCPM L512 CMP"; PF_FA_CMP=1 PREFILL=1 timeout 60 $B/mcv_ref $M logs/comp/mcvref/L512 4 2>&1 | grep -E "FA_CMP|OURS|first pred|rror|panic" | cut -c1-1500
echo DONE
