#!/bin/bash
cd ~/strix-rt; B=target-q8c/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; P="$(cat logs/q38_prompt_ids.txt)"; export ZEROCOPY=dmabuf
for nr in 1 2 4; do echo "== NBNR_Q=$nr"; GEMVQ_NBNR_Q=$nr VPROF=1 timeout 100 $B/q38 $Q "$P" 8 2>&1 | grep verify; done
for i in 1 2 3; do echo "== rep $i"; MTP_HEAD_Q=1 GEMVQ_NBNR_Q=2 MTP=2,3 timeout 120 $B/q38 $Q "$P" 128 2>&1 | grep -E "MTP k|spec k|rror"; done; echo DONE
