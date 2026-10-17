#!/bin/bash
# MTP A/B on Qwen3.8: GDN_M=0 (per-column step + snapshot copies) vs GDN_M=1 (multi-column step); verify GPU ms, spec k=1..3
cd ~/strix-rt; B=target-q8/release; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; P="$(cat logs/q38_prompt_ids.txt)"; export ZEROCOPY=dmabuf
for g in 0 1 2; do echo "== GDN_M=$g"
GDN_M=$g VPROF=1 timeout 120 $B/q38 $Q "$P" 8 2>&1 | grep verify
GDN_M=$g MTP=1,2,3 RUNS=2 timeout 240 $B/q38 $Q "$P" 128 2>&1 | grep -E "baseline|MTP k|spec k|rror"
done
for v in 65536 32768; do echo "== MTP_VOCAB=$v"; MTP_VOCAB=$v MTP=2,3 RUNS=2 timeout 200 $B/q38 $Q "$P" 128 2>&1 | grep -E "MTP k|spec k|rror"; done
echo DONE
