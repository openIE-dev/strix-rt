#!/bin/bash
cd ~/strix-rt; B=logs/r11/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r11
while [ ! -f $D/DONE ]; do sleep 10; done
echo high | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level
trap "echo auto | sudo tee /sys/class/drm/card1/device/power_dpm_force_performance_level" EXIT
run() { tag=$1; shift; bash logs/gpu_idle_run.sh $D/$tag.log "$@"; }
run sw512_8704_2560 NPU_SPLIT=1 NPU_NN=5120x17408:8704,17408x5120:2560 PF_M=512 NOREF=1 REPS=7 $B $M 512 1
for r in b c; do
 for L in 512 2048; do run perf_off_${L}$r PF_M=$L NOREF=1 REPS=7 $B $M $L 1
  run perf_on_${L}$r NPU_SPLIT=1 PF_M=$L NOREF=1 REPS=7 $B $M $L 1
  run noattn_${L}$r NPU_SPLIT=1 NPU_SPLIT_OPS=attn_gate,attn_output,ffn_gate,ffn_up,ffn_down PF_M=$L NOREF=1 REPS=7 $B $M $L 1; done; done
echo DONE > $D/DONE2
