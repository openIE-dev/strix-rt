#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
run() { n=$1; shift; for r in 1 2 3 4 5; do env "$@" G4_QMMH=1 G4_PF=1 MAX_CTX=4096 timeout 90 ./target-g4p/release/g4_run $M rand:512 2 2>&1 | grep prompt | tr -d "(" | awk -v n=$n '{print n,$6}'; done; }
run new X=1
run mi1 PF_QMM_MI=1
run mi1wr2 PF_QMM_MI=1 PF_QH_WR=2
run kd2 PF_QH_KD=2
run new X=1
