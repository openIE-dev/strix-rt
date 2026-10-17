#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf
for s in 0 1 2 4 16 8 30 0; do G4_SKIP=$s bash logs/g4/esample.sh sk$s ./target-g4p/release/g4_run $M @logs/g4/e4b_prompt_ids.txt 513 | sed "s/^/skip$s /" | cut -c1-200; grep -o "decode 512.*tok/s" /tmp/esample_sk$s.out | head -1 | sed "s/^/skip$s /"; done
