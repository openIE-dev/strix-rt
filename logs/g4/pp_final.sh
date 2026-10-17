#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for L in 512 2048; do for cfg in "PF_QH_WR=2 PF_QH_KD=8" "X=new"; do for r in 1 2 3 4 5 6; do
 env $cfg G4_QMMH=1 G4_PF=1 MAX_CTX=4096 timeout 90 ./target-g4p/release/g4_run $M rand:$L 2 2>&1 | grep prompt | tr -d "(" | awk -v n="pp$L [$cfg]" '{print n, $6}'; done; done; done
