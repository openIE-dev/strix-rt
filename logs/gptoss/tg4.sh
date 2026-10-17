#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
for c in "GO_NR=0" "GO_NR=1" "GO_NR=1 ATTN_NC=2"; do echo "== $c"; env $c RUNS=2 timeout 60 target-gopf/go_run4 $M @$D/prompt_ids.txt 128 @$D/ref_ids.txt 2>&1 | grep -E "decode|parity" | tail -2; done; echo DONE
