#!/bin/bash
cd ~/strix-rt; B=logs/r12/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r12
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
run() { tag=$1; shift; echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log "$@"; echo "[perf level after: $(cat $P)]" >> $D/$tag.log; }
F=NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down
for r in c d; do
 run s2off_$r PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1
 for g in 4096 6144; do for d in 1536 2048; do
  run s2f_${g}_${d}_$r NPU_SPLIT=1 NPU_STATS=1 $F NPU_NN=5120x17408:$g,17408x5120:$d PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1; done; done
 run s2gu_6144_$r NPU_SPLIT=1 NPU_STATS=1 NPU_SPLIT_OPS=ffn_gate,ffn_up NPU_NN=5120x17408:6144 PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1
done
echo auto | sudo tee $P; echo DONE > $D/DONE_SW3
