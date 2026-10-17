#!/bin/bash
# one perf_high session; each config 5 reps pp512 (+ids of 16 gen tokens for parity at the end)
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
run() { # name envs...
  n=$1; shift
  for r in 1 2 3 4 5; do env "$@" G4_PF=1 MAX_CTX=4096 timeout 90 ./target-g4p/release/g4_run $M rand:512 2 2>&1 | grep prompt | sed "s/^/$n /" | cut -c1-100; done
}
run base G4_QMMH=1
run kd4 G4_QMMH=1 PF_QH_KD=4
run kd4nj1 G4_QMMH=1 PF_QH_KD=4 PF_QH_NJ=1
run kd4wr1 G4_QMMH=1 PF_QH_KD=4 PF_QH_WR=1
run kd4wr1nj1 G4_QMMH=1 PF_QH_KD=4 PF_QH_WR=1 PF_QH_NJ=1
run kd2nj1 G4_QMMH=1 PF_QH_KD=2 PF_QH_NJ=1
run kd4nj1_4nj1 G4_QMMH=1 PF_QH_KD=4 PF_QH_NJ=1 PF_QH4_NJ=1
run kd4 G4_QMMH=1 PF_QH_KD=4
