#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r20/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r20
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.* && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r20: $1" -- $D/$1.log $D/$1.json && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r20-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
# is the pp4096 TB greedy miss numerics (int8 activations on the NPU) or a bug? same int8 path on the GPU / split
while [ ! -f $D/DONE ]; do sleep 10; done
run i8chk4096 PF_TF16=0 PF_M=2048 REPS=2 $B $M 4096 16
run splitchk4096 NPU_SPLIT=1 PF_M=2048 REPS=2 $B $M 4096 16
echo DONE > $D/DONE2
