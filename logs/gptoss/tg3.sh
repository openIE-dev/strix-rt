#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
for c in "X=1" "ATTN_NC=4" "ATTN_NC=8" "GEMV_SG=64" "GEMV_PROJ=64,256 GEMV_OUT=64,256" "ZEROCOPY=dmabuf"; do echo "== $c"; env $c RUNS=2 timeout 60 target-gopf/go_run3 $M @$D/prompt_ids.txt 128 @$D/ref_ids.txt 2>&1 | grep -E "decode|parity" | tail -2; done; echo DONE
