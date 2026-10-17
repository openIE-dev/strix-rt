#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; B=target-g4p/release/g4_run
for tb in 1 2 3 4 5; do for L in rand:512 rand:200 rand:90 rand:1500 @logs/g4/26b_prompt_ids.txt @logs/g4/26bc_prompt_ids.txt; do echo "TB=$tb $L $(G4_RLOG_TB=$tb G4_PF=1 MAX_CTX=4096 timeout 60 $B $M $L 48 | grep ^ids | md5sum | cut -c1-8)"; done
echo "TB=$tb $(G4_RLOG_TB=$tb G4_PF=1 G4_PFPROF=1 MAX_CTX=4096 timeout 60 $B $M rand:512 2 2>&1 | grep -E "moe_route" | cut -c1-60)"
for r in 1 2 3; do echo "TB=$tb $(G4_RLOG_TB=$tb G4_PF=1 MAX_CTX=4096 timeout 60 $B $M rand:512 2 | grep ^prompt | cut -c1-50)"; done; done
