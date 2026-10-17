#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for r in 1 2 3 4 5; do for d in 0 1; do G4_DQ2=$d bash logs/g4/esample.sh dq$d ./target-g4p/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 513 | sed "s/^/DQ2=$d /" | cut -c1-220; done; done
