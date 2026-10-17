#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for cfg in X=1 PF_QMM_MI=1; do
 for r in 1 2 3 4; do env $cfg G4_QMMH=1 G4_PF=1 MAX_CTX=4096 ./target-g4p/release/g4_run $M rand:2048 2 2>&1 | grep prompt | tr -d "(" | awk -v n="pp2048 $cfg" '{print n,$6}'; done
 echo "== parity $cfg"
 env $cfg G4_PF=1 G4_QMMH=1 ./target-g4p/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt 2>&1 | grep -E "^ids" | md5sum
 env $cfg G4_PF=1 G4_QMMH=1 FORCED=1 ./target-g4p/release/g4_run $M @logs/g4/26bc_prompt_ids.txt 64 @logs/g4/26bc_vk_ids.txt 2>&1 | grep FORCED | cut -c1-250
done
