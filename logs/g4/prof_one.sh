#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for i in 1 2; do G4_PF=1 G4_PFPROF=1 MAX_CTX=4096 timeout 100 target-g4p/release/g4_run $M rand:512 2 2>&1 | grep -E "G4_PFPROF|ms  |prompt" | cut -c1-140; done
