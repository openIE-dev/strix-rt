#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r19/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r19
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r19: $1" -- $D/$1.log && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r19-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
while [ ! -f $D/DONE2 ]; do sleep 10; done
run offchk4096 PF_M=2048 REPS=2 $B $M 4096 16
for r in c d; do run tbm4096_$r NPU_TB=1 PF_M=4096 NOREF=1 REPS=3 $B $M 4096 1; done
echo DONE > $D/DONE3
