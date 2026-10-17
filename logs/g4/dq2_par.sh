#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for d in 0 1; do echo "== DQ2=$d"; G4_DQ2=$d timeout 120 ./target-g4p/release/g4_run $M @logs/g4/26b_prompt_ids.txt 64 @logs/g4/26b_cont_ids.txt 2>&1 | tail -3 | cut -c1-250; G4_DQ2=$d FORCED=1 timeout 120 ./target-g4p/release/g4_run $M @logs/g4/26b_prompt_ids.txt 64 @logs/g4/26b_cont_ids.txt 2>&1 | tail -3| cut -c1-250; done
