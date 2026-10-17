#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
for B in target-q8 target-q8b; do echo "== $B"; RUNS=2 timeout 60 $B/release/go_run $M @$D/prompt_ids.txt 128 @$D/ref_ids.txt 2>&1 | grep -E "decode|parity"; done
echo DONE
