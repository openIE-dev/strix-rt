#!/bin/bash
# gpt-oss batched prefill parity: GO_PF=0 vs 1 (nb 2,4); greedy ids vs llama.cpp ref
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
for c in "0 4" "1 1" "1 2" "1 4"; do set -- $c; echo "== GO_PF=$1 NB=$2"; GO_PF=$1 GO_PF_NB=$2 RUNS=2 timeout 60 target-gopf/go_run $M @$D/prompt_ids.txt 64 @$D/ref_ids.txt 2>&1 | grep -E "prompt|decode|parity|rror|panic"; done
echo DONE
