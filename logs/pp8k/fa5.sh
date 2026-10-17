#!/bin/bash
cd ~/strix-rt; B=target-fa/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for c in "1 256 L512" "1 384 L512" "0 512 L2048" "1 512 L2048"; do set -- $c; echo "== FA=$1 PF_M=$2 $3"; PF_FA=$1 PF_M=$2 PREFILL=1 timeout 60 $B/mcv_ref $M logs/comp/mcvref/$3 8 2>&1 | grep -E "OURS|first pred|rror|panic"; done; echo DONE
