#!/bin/sh
# Round 6: Bonsai ternary via f16 coop-matrix GEMM (PF_TF16=1) vs int8 pf_tgemm (PF_TF16=0): correctness L=512, pp512/pp2048 (perf high)
cd ~/strix-rt; D=/sys/class/drm/card1/device; P2=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; P1=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
. logs/pf_busy.sh; waitidle; echo "[start $(date +%T)]"; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
for m in $P2 $P1; do for t in 0 1; do waitidle; echo "$(basename $m) TF16=$t: $(PF_TF16=$t REPS=2 ./target/release/pf_q38 $m 512 8 2>&1 | grep -E "cosine|argmax|greedy|panic|rror" | tr "\n" " ")"; done; done
run() { m=$1; l=$2; shift 2; waitidle; echo "$(basename $m) pp$l $* : $(env "$@" PF_M=$l REPS=5 NOREF=1 ./target/release/pf_q38 $m $l 1 2>&1 | grep ^prefill) [busy after: $(busy && echo YES || echo no)]"; }
for m in $P2 $P1; do for l in 512 2048; do run $m $l PF_TF16=0; run $m $l PF_TF16=1; done; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored $(cat $D/power_dpm_force_performance_level)"
