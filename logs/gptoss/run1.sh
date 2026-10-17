#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
timeout 200 $D/ref.sh
RUNS=3 timeout 200 target-q8/release/go_run $M @$D/prompt_ids.txt 128 @$D/ref_ids.txt 2>&1 | tail -8
echo DONE
