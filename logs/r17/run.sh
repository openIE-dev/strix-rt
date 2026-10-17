#!/bin/bash
# round 17: pp512 NPU column sweep (gate/up) + attention projections (K=5120 shapes), each under the GPU lock
cd ~/strix-rt; B=logs/r16/pfq_bin2; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r17
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r17: $1" -- $D/$1.log && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r17-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
X="PF_M=512 NOREF=1 REPS=7 $B $M 512 1"; S="NPU_SPLIT=1 NPU_STATS=1"
A1="5120x10240:5632,5120x6144:3584,5120x12288:6656"; A2="5120x10240:3584,5120x6144:2048,5120x12288:4096"
for r in a b; do
 run off_$r $X
 for g in 3584 4096 5632 6144 8704; do run gu${g}_$r $S NPU_NN=5120x17408:$g $X; done
 run att1_$r $S NPU_NN=5120x17408:4096,$A1 $X
 run att2_$r $S NPU_NN=5120x17408:4096,$A2 $X
done
run chk_att2 $S NPU_NN=5120x17408:4096,$A2 PF_M=512 REPS=2 $B $M 512 16
echo DONE > $D/DONE
