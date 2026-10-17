#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for t in 26bc:26bc_vk 26b:26b_ref_vk; do P=${t%%:*}; R=${t##*:}; for d in 0 1; do echo "== $P DQ2=$d"; G4_DQ2=$d timeout 120 ./target-g4p/release/g4_run $M @logs/g4/${P}_prompt_ids.txt 64 @logs/g4/${R}_ids.txt 2>&1 | grep -E "parity"; G4_DQ2=$d FORCED=1 timeout 120 ./target-g4p/release/g4_run $M @logs/g4/${P}_prompt_ids.txt 64 @logs/g4/${R}_ids.txt 2>&1 | grep FORCED | cut -c1-300; done; done
