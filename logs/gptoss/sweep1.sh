#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf; R="target-q8/release/go_run $M @$D/prompt_ids.txt 128"
for e in "DBG_LAYERS=0" "DBG_LAYERS=12" "X=1" "GEMV_HEAD=16,256" "GEMV_HEAD=32,256" "GEMV_PROJ=32,128" "ATTN_NC=4" "ZEROCOPY=dmabuf"; do echo "== $e"; env $e timeout 60 $R 2>&1 | grep decode; done
echo DONE
