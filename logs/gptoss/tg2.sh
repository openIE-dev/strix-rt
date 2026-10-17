#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
for c in "GO_LPR=16" "GO_LPR=8" "GO_LPR=32" "GO_LPR=32 GEMV_PROJ=32,128 GEMV_OUT=32,128"; do echo "== $c"; env $c RUNS=2 timeout 60 target-gopf/go_run2 $M @$D/prompt_ids.txt 128 @$D/ref_ids.txt 2>&1 | grep -E "decode|parity" | tail -2; done; echo DONE
