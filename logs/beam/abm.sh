#!/bin/bash
# logs/beam/abm.sh (inside gpu_lock): add-before-multiply 3-bit GEMV (BEAM_ABM=1 bit-plane, 2 code-bin) vs default MAC (0).
# A: teacher-forced top-20 on 2 eval prompts, full model q3 (exactness).  B: BEAM_NL=8 + 224-slot cache (GPU-bound), differential energy N=200-N=40, interleaved, 2 reps.
cd ~/strix-rt; O=logs/beam/abm; mkdir -p $O; B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4; Q=$MODELS/new/beam/q3
R=/usr/local/bin/tools/energy/record.py; export BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304 BEAM_PREFILL=1
grep -P "^(science_chat|sql)\t" logs/beam/seq.tsv | while IFS=$'\t' read -r name seq pl; do for a in 0 1 2; do
  [ -s $O/$name.abm$a.log ] || BEAM_ABM=$a BEAM_QDIR=$Q BEAM_FORCE=$((pl-1)) $B $D $seq > $O/$name.abm$a.log 2>&1; done; done
echo "A done $(date +%T)" >> $O/status
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
for rep in 1 2; do for a in 0 1 2; do for n in 200 40; do t=nl8_abm${a}_n${n}_r$rep
  python3 logs/cpubusy.py gate 0.8 900 >> $O/gate.txt || echo "gate timeout $t" >> $O/gate.txt
  BEAM_ABM=$a BEAM_QDIR=$Q BEAM_NL=8 BEAM_CACHE=224 BEAM_WARMSEQ=logs/beam/seq.tsv BEAM_WARMT=16 $R record --hz 20 --idle 5 --json $O/$t.json -- $B $D $IDS $n > $O/$t.log 2>&1
done; done; done
echo "B done $(date +%T)" >> $O/status
