#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r21/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r21
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.* && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r21: $1" -- $D/$1.* && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r21-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
# round 21: block-size / partial-FFN variants of NPU_TB (speed), pp2048 and pp4096
while ! grep -q DONE ~/npu/kernel/r21build.log; do sleep 10; done
V() { tag=$1; shift; for r in a b; do run ${tag}_$r "$@" NOREF=1 REPS=4 $B $M $L 1; done; }
for L in 2048 4096; do
 V tb$L NPU_TB=1 PF_M=2048
 V tba512_$L NPU_TB=1 NPU_TB_A=512 PF_M=2048
 V tbp_$L NPU_TB=1 PF_M=2048 NPU_NN=5120x17408:8704,17408x5120:2560
 V tba512m3_$L NPU_TB=1 NPU_TB_A=512 PF_M=1536
done
echo DONE > $D/DONE
