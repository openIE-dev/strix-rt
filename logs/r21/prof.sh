#!/bin/bash
# round 21: who burns core power during NPU_TB prefill? system-wide view while prefills run (after weight fold)
cd ~/strix-rt; D=logs/r21
NPU_TB=1 PF_M=2048 NOREF=1 REPS=8 logs/r21/pfq_bin $MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf 2048 1 > $D/prof_run.log 2>&1 &
P=$!; sleep 22
mpstat -P ALL 5 1 > $D/prof_mpstat.log 2>&1
sudo perf record -a -g -o /tmp/r21perf.data -- sleep 5 > /dev/null 2>&1
sudo perf report -i /tmp/r21perf.data --no-children --sort comm,dso,sym --stdio 2>/dev/null | grep -v "^#" | grep "%" | head -40 > $D/prof_perf.txt
top -b -H -n 1 -w 200 | head -30 > $D/prof_top.txt
wait $P
