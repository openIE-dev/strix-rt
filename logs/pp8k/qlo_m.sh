#!/bin/bash
# PF_FA_QLO=0 acceptance, MiniCPM: forced agreement vs the FA=0 greedy ids from fa8 (FA=0 baseline 48/48,48/48,48/48,48/48,13/13)
cd ~/strix-rt; B=target-fa3/release; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
for d in ~/mcv_ref/ref_chess ~/mcv_ref/ref_gold ~/mcv_ref/ref_txt logs/comp/mcvref/L512 logs/comp/mcvref/L2048; do n=$(basename $d)
  echo "== MiniCPM $n QLO=0 $(PF_FA=1 PF_FA_QLO=0 PREFILL=1 PFORCED=/tmp/fa8_$n.ids timeout 60 $B/mcv_ref $M $d 2>&1 | grep -E "PFORCED|rror|panic")"; done; echo DONE
