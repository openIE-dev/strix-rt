#!/bin/bash
# Part B rerun (inside gpu_lock): box not quiet today (other users' 20-32-core jobs) -> gate 60 s max, busy cores logged per window.
cd ~/strix-rt; O=logs/beam/abm; B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4; Q=$MODELS/new/beam/q3
R=/usr/local/bin/tools/energy/record.py; export BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304 BEAM_PREFILL=1
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
for rep in 1 2; do for a in 0 1 2; do for n in 200 40; do t=nl8_abm${a}_n${n}_r$rep
  [ -s $O/$t.json ] && continue
  python3 logs/cpubusy.py gate 0.8 60 >> $O/gate.txt || echo "gate timeout $t" >> $O/gate.txt
  b0=$(python3 logs/cpubusy.py snap)
  BEAM_ABM=$a BEAM_QDIR=$Q BEAM_NL=8 BEAM_CACHE=224 BEAM_WARMSEQ=logs/beam/seq.tsv BEAM_WARMT=16 $R record --hz 20 --idle 5 --json $O/$t.json -- $B $D $IDS $n > $O/$t.log 2>&1
  b1=$(python3 logs/cpubusy.py snap)
  echo "$t $b0 $b1" >> $O/busy_raw.txt
done; done; done
echo "B done $(date +%T)" >> $O/status
