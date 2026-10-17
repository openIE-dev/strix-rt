#!/bin/bash
# round 21: where does NPU_TB host CPU time go? user/sys seconds for REPS=1 vs REPS=4 (3 pp2048 prefills)
cd ~/strix-rt; B=logs/r20/pfq_bin; M=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; D=logs/r21
for cfg in "off:NPU_SPLIT=0" "tb:NPU_TB=1"; do n=${cfg%%:*}; e=${cfg#*:}; for r in 1 4; do
 echo "== $n $r" >> $D/cpu.log
 /usr/bin/time -f "TIME user %U sys %S wall %e" env $e PF_M=2048 NOREF=1 REPS=$r $B $M 2048 1 2>&1 | grep -E "TIME|best" >> $D/cpu.log
done; done
perf_ok=$(command -v perf); echo "perf: $perf_ok" >> $D/cpu.log
