#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r22/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r22
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.* && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r22: $1" -- $D/$1.* && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r22-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
# round 22: NPU_TB=3 (A,B on NPU alternating, C on GPU): greedy + speed
run chk3_2048 NPU_TB=3 NPU_STATS=1 PF_M=2048 REPS=2 $B $M 2048 16
for r in a b; do
 run tb3_2048_$r NPU_TB=3 PF_M=2048 NOREF=1 REPS=4 $B $M 2048 1
 run tb3m3_2048_$r NPU_TB=3 NPU_TB_A=1024 PF_M=3072 NOREF=1 REPS=4 $B $M 2048 1
 run tb2_2048_$r NPU_TB=1 PF_M=2048 NOREF=1 REPS=4 $B $M 2048 1
 run off2048_$r PF_M=2048 NOREF=1 REPS=4 $B $M 2048 1
 run tb3_4096_$r NPU_TB=3 PF_M=2048 NOREF=1 REPS=3 $B $M 4096 1
 run tb3m4_4096_$r NPU_TB=3 NPU_TB_A=1024 PF_M=4096 NOREF=1 REPS=3 $B $M 4096 1
done
echo DONE > $D/DONE
