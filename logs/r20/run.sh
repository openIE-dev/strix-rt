#!/bin/bash
# round 16: NPU_KCHUNK (ffn_down as K=5120 chunks, one xclbin) vs K switch, greedy + pp512/pp2048, each under the GPU lock
cd ~/strix-rt; B=logs/r20/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r20
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.log $D/$1.json && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r20: $1" -- $D/$1.log $D/$1.json && return; sleep 3; done; }
run() { tag=$1; shift; GPU_LOCK_TAG=r20-$tag logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo auto | sudo tee $P >/dev/null"; commit $tag; }
# round 20: energy per token (energy recorder), pp2048: REPS=6 minus REPS=1 isolates 5 prefills (load/fold cancel)
while [ ! -f logs/r19/DONE3 ]; do sleep 10; done
T() { tag=$1; shift; run $tag tools/energy/record.py record --hz 1000 --idle 3 --json $D/$tag.json -- env "$@"; }
for cfg in "off:NPU_SPLIT=0" "split:NPU_SPLIT=1" "tb:NPU_TB=1"; do n=${cfg%%:*}; e=${cfg#*:}
 for r in 1 6; do T e_${n}_$r $e PF_M=2048 NOREF=1 REPS=$r $B $M 2048 1; done; done
echo DONE > $D/DONE
