#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r24/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r24
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.* && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r24: $1" -- $D/$1.* && return; sleep 3; done; }
run() { tag=$1; shift; logs/wait_quiet.sh true >/dev/null; GPU_LOCK_TAG=r24-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
# round 24: NPU_AUTO default (no NPU env) at PF_M=2048 by prompt length: greedy + speed vs NPU_AUTO=0
while [ ! -f logs/r23/DONE ]; do sleep 20; done
for L in 1024 2048 4096; do
 run auto_chk$L PF_M=2048 NPU_STATS=1 REPS=2 $B $M $L 16
 run auto$L PF_M=2048 NOREF=1 REPS=3 $B $M $L 1
 run gpu$L NPU_AUTO=0 PF_M=2048 NOREF=1 REPS=3 $B $M $L 1
done
echo DONE > $D/DONE
