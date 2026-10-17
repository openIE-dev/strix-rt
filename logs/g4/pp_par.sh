#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for cfg in "X=0" "PF_QH_KD=4 PF_QH_WR=1" "PF_QH_KD=4 PF_QH_WR=1 PF_QH_NJ=1"; do
 for t in 26bc:26bc_vk 26b:26b_ref_vk; do P=${t%%:*}; R=${t##*:}; echo "== $cfg $P"
  env $cfg G4_PF=1 G4_QMMH=1 timeout 120 ./target-g4p/release/g4_run $M @logs/g4/${P}_prompt_ids.txt 64 @logs/g4/${R}_ids.txt 2>&1 | grep -E "^ids|parity" | cut -c1-400
  env $cfg G4_PF=1 G4_QMMH=1 FORCED=1 timeout 120 ./target-g4p/release/g4_run $M @logs/g4/${P}_prompt_ids.txt 64 @logs/g4/${R}_ids.txt 2>&1 | grep FORCED | cut -c1-300
 done; done
