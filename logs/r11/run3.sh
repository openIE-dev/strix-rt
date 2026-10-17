#!/bin/bash
cd ~/strix-rt; B=logs/r11/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r11
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
run() { tag=$1; shift; echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log "$@"; echo "[perf level after: $(cat $P)]" >> $D/$tag.log; }
for L in 512 2048; do run chk_tuned_$L NPU_SPLIT=1 PF_M=$L REPS=2 $B $M $L 16; done
for r in a b; do for L in 512 2048; do
 run fin_off_${L}$r PF_M=$L NOREF=1 REPS=7 $B $M $L 1
 run fin_on_${L}$r NPU_SPLIT=1 NPU_STATS=1 PF_M=$L NOREF=1 REPS=7 $B $M $L 1; done; done
echo auto | sudo tee $P; echo DONE > $D/DONE3
