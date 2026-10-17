#!/bin/bash
# tools/comp/srt.sh KIND MODEL [extra env..]: strix-rt single-stream numbers (pp512/2048/8192, tg128, decode at depth 8192)
# KIND: q38 (Qwen3.8 / Bonsai 2 via crate qwen38), lfm (LFM2.5 via lfm2), mcv (qwen35moe via crate mcv)
cd ~/strix-rt; B=target-comp/release; K=$1; M=$2; export ZEROCOPY=dmabuf
case $K in
q38|lfm) PF=$B/pf_$K; [ $K = q38 ] && PF=$B/pf_q38
  for L in 512 2048; do NOREF=1 REPS=3 timeout 600 $PF $M $L 1 2>&1 | grep -E "prefill|rror"; done
  NOREF=1 REPS=2 DEPTH_TG=1 timeout 900 $PF $M 8192 129 2>&1 | grep -E "prefill|decode|rror"
  P="$(cat logs/${K}_prompt_ids.txt)"; RUNS=3 timeout 600 $B/$K $M "$P" 128 2>&1 | grep -E "prompt|rror" ;;
mcv) unset ZEROCOPY; [ -n "$MCV_ZC" ] && export ZEROCOPY=$MCV_ZC
  for L in 512 2048 8192; do PREFILL=1 REPS=3 MAX_CTX=8400 timeout 900 $B/mcv_ref $M logs/comp/mcvref/L$L 129 2>&1 | grep -E "prefill|decode|rror|panic"; done
  for r in 1 2 3; do PREFILL=1 timeout 600 $B/mcv_ref $M logs/comp/mcvref/L12 129 2>&1 | grep -E "decode|rror|panic"; done ;;
esac
