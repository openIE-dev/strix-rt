#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
G4_PFPROF=1 G4_QMMH=1 G4_PF=1 MAX_CTX=4096 ./target-g4p/release/g4_run $M rand:512 2 2>&1 | grep -iE "moe|prompt|chunk" | cut -c1-120
