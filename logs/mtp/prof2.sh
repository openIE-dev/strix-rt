#!/bin/bash
# verify cost split: VSKIP_GEMV (no multi-column GEMVs) and VONE (multi-column GEMVs as 1 column)
cd ~/strix-rt; B=target-q8/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; P="$(cat logs/q38_prompt_ids.txt)"; export ZEROCOPY=dmabuf
for e in VSKIP_GEMV VONE; do echo "== $e"; env $e=1 VPROF=1 timeout 120 $B/q38 $Q "$P" 8 2>&1 | grep verify; done
for nr in 2 4; do echo "== GEMVQ_NBNR=$nr"; GEMVQ_NBNR=$nr VPROF=1 timeout 120 $B/q38 $Q "$P" 8 2>&1 | grep verify; done
echo DONE
