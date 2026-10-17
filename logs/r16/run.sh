#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r16/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r16
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r16: $1" -- $D/$1.log && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r16-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
F=NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down; N5="NPU_NN=5120x17408:4096,17408x5120:1536"
run chk512 NPU_SPLIT=1 NPU_STATS=1 $F $N5 PF_M=512 REPS=2 $B $M 512 16
run chk2048 NPU_SPLIT=1 NPU_STATS=1 PF_M=2048 REPS=2 $B $M 2048 16
for r in a b; do
 run off512_$r PF_M=512 NOREF=1 REPS=7 $B $M 512 1
 run kc512_$r NPU_SPLIT=1 NPU_STATS=1 $F $N5 PF_M=512 NOREF=1 REPS=7 $B $M 512 1
 run sw512_$r NPU_SPLIT=1 NPU_KCHUNK=0 $F $N5 PF_M=512 NOREF=1 REPS=7 $B $M 512 1
 run gu512_$r NPU_SPLIT=1 NPU_SPLIT_OPS=ffn_gate,ffn_up NPU_NN=5120x17408:4096 PF_M=512 NOREF=1 REPS=7 $B $M 512 1
 run off2048_$r PF_M=2048 NOREF=1 REPS=5 $B $M 2048 1
 run kc2048_$r NPU_SPLIT=1 NPU_STATS=1 PF_M=2048 NOREF=1 REPS=5 $B $M 2048 1
 run sw2048_$r NPU_SPLIT=1 NPU_KCHUNK=0 PF_M=2048 NOREF=1 REPS=5 $B $M 2048 1
done
echo DONE > $D/DONE
