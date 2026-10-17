#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; B=target-g4p/release/g4_run
for tb in 1 2; do echo "TB=$tb"; G4_RLOG_TB=$tb G4_PF=1 G4_PFPROF=1 MAX_CTX=4096 timeout 60 $B $M rand:512 2 2>&1 | grep -E "moe_|G4_PFPROF" | cut -c1-70; done
G4_RLOG=0 G4_PF=1 G4_PFPROF=1 MAX_CTX=4096 timeout 60 $B $M rand:512 2 2>&1 | grep -E "RLOG|moe_rt" | cut -c1-70
