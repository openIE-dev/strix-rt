#!/bin/bash
# round 14: NPU_FUSE (gate/up share one sync point) vs round-13 path, under the GPU lock
cd ~/strix-rt; B=logs/r14/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r14
while [ ! -f logs/dq/DONE ]; do sleep 10; done
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r14: $1" -- $D/$1.log && return; sleep 3; done; }
run() { tag=$1; shift
  logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
F=NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down
run chk2048 NPU_SPLIT=1 NPU_STATS=1 PF_M=2048 REPS=2 $B $M 2048 16
run chk512 NPU_SPLIT=1 NPU_STATS=1 $F NPU_NN=5120x17408:4096,17408x5120:1536 PF_M=512 REPS=2 $B $M 512 16
for r in a b; do
 run off2048_$r PF_M=2048 NOREF=1 REPS=5 $B $M 2048 1
 run fuse2048_$r NPU_SPLIT=1 NPU_STATS=1 PF_M=2048 NOREF=1 REPS=5 $B $M 2048 1
 run nofuse2048_$r NPU_SPLIT=1 NPU_FUSE=0 PF_M=2048 NOREF=1 REPS=5 $B $M 2048 1
 run off512_$r PF_M=512 NOREF=1 REPS=7 $B $M 512 1
 run fuse512_$r NPU_SPLIT=1 NPU_STATS=1 $F NPU_NN=5120x17408:4096,17408x5120:1536 PF_M=512 NOREF=1 REPS=7 $B $M 512 1
 run nofuse512_$r NPU_SPLIT=1 NPU_FUSE=0 $F NPU_NN=5120x17408:4096,17408x5120:1536 PF_M=512 NOREF=1 REPS=7 $B $M 512 1
done
echo DONE > $D/DONE
