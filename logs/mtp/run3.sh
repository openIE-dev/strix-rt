#!/bin/bash
cd ~/strix-rt; B=target-q8/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; P="$(cat logs/q38_prompt_ids.txt)"; export ZEROCOPY=dmabuf
VPROF=1 timeout 120 $B/q38 $Q "$P" 8 2>&1 | grep verify
MTP=2,3 RUNS=2 timeout 200 $B/q38 $Q "$P" 128 2>&1 | grep -E "baseline|MTP k|spec k|rror"
echo DONE
