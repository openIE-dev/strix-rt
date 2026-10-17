#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for L in 512 2048; do for r in 1 2 3 4 5; do G4_QMMH=1 G4_PF=1 MAX_CTX=4096 timeout 60 ./target-g4p/release/g4_run $M rand:$L 2 | grep prompt | sed "s/^/pp$L /" | cut -c1-90; done; done
