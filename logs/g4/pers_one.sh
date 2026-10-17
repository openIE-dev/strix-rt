#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for p in 0 160; do echo "ids PERS=$p: $(PF_QH_PERS=$p G4_QMMH=1 G4_PF=1 MAX_CTX=4096 ./target-g4p/release/g4_run $M rand:512 16 2>&1 | grep -iE "ids|^\[|token" | tail -2 | md5sum)"; done
for r in 1 2; do for p in 0 80 160 240 480 0; do echo -n "PERS=$p "; for i in 1 2 3 4 5; do PF_QH_PERS=$p G4_QMMH=1 G4_PF=1 MAX_CTX=4096 ./target-g4p/release/g4_run $M rand:512 2 2>&1 | grep prompt | sed "s/.*in .*s (\([0-9.]*\) tok.*/\1/"; done | tr "\n" " "; echo; done; done
