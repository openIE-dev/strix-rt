#!/bin/bash
# round 13: NPU_SPLIT default greedy check + pp2048 FFN-only sweep vs f16 baseline, each run under the GPU lock
cd ~/strix-rt; B=logs/r13/pfq; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r13
P=/sys/class/drm/card1/device/power_dpm_force_performance_level
commit() { for i in 1 2 3 4 5; do git add $D/$1.log && git -c user.name="David Charlot" -c user.email=you@example.com commit -qm "logs/r13: $1" -- $D/$1.log && return; sleep 3; done; }
run() { tag=$1; shift
  logs/gpu_lock.sh sh -c "echo high | sudo tee $P >/dev/null; bash logs/gpu_idle_run.sh $D/$tag.log $*; echo \"[perf level during: \$(cat $P)]\" >> $D/$tag.log; echo auto | sudo tee $P >/dev/null"
  echo "[gfx timeouts so far since start: $(sudo journalctl -k --since "$T0" --no-pager | grep -c "ring gfx.*timeout")]" >> $D/$tag.log; commit $tag; }
T0="$(date "+%F %T")"
run chk_default_2048 NPU_SPLIT=1 NPU_STATS=1 PF_M=2048 REPS=2 $B $M 2048 16
F=NPU_SPLIT_OPS=ffn_gate,ffn_up,ffn_down
for r in a b; do
 run off_$r PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1
 for g in 4096 6144; do for d in 1536 2048; do
  run f_${g}_${d}_$r NPU_SPLIT=1 NPU_STATS=1 $F NPU_NN=5120x17408:$g,17408x5120:$d PF_M=2048 NOREF=1 REPS=7 $B $M 2048 1; done; done
done
echo DONE > $D/DONE
