#!/bin/bash
cd ~/strix-rt; B=target-fa/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for fa in 0 1; do echo "== FA=$fa L512"; PF_FA=$fa PREFILL=1 timeout 60 $B/mcv_ref $M logs/comp/mcvref/L512 2 2>&1 | grep -E "TOP2|first pred|rror|panic"; done; echo DONE
