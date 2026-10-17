#!/bin/bash
cd ~/strix-rt; B=logs/r11/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r11
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level" EXIT
run() { tag=$1; shift; bash logs/gpu_idle_run.sh $D/$tag.log "$@"; }
# correctness (greedy 16 vs token-by-token)
for L in 512 2048; do PM=$L; [ $L = 512 ] && PM=512
  run chk_off_$L PF_M=$PM REPS=2 $B $M $L 16
  run chk_on_$L NPU_SPLIT=1 PF_M=$PM REPS=2 $B $M $L 16; done
# perf
for L in 512 2048; do
  run perf_off_$L PF_M=$L NOREF=1 REPS=7 $B $M $L 1
  run perf_on_$L NPU_SPLIT=1 NPU_STATS=1 PF_M=$L NOREF=1 REPS=7 $B $M $L 1; done
# split sweep
for g in 8704 9216 9728; do for d in 2560 3072; do
  run sw512_${g}_$d NPU_SPLIT=1 NPU_NN=5120x17408:$g,17408x5120:$d PF_M=512 NOREF=1 REPS=7 $B $M 512 1; done; done
for g in 9216 9728; do for d in 2560 3072; do
  run sw2048_${g}_$d NPU_SPLIT=1 NPU_NN=5120x17408:$g,17408x5120:$d PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1; done; done
run ffnonly512 NPU_SPLIT=1 NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down PF_M=512 NOREF=1 REPS=7 $B $M 512 1
run ffnonly2048 NPU_SPLIT=1 NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1
echo DONE > $D/DONE
