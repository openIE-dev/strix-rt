#!/bin/bash
# speed + differential energy: direct IO (no page cache), nvfp4/q2/q3, with and without managed expert cache (held-out warmed)
cd ~/strix-rt; O=logs/beam/bench; mkdir -p $O; rm -f $O/*
B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
export BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304 BEAM_PREFILL=1
# cache correctness (same tokens as baseline?)
BEAM_CACHE=64 $B $D 151669,872,151671,3838,374,279,6722,315,9625,30,151670,151669,77091 48 > $O/cache_correct.log 2>&1
run() { # name qdir cache
  local name=$1 q=$2 ck=$3
  for rep in 1 2; do for n in 40 2; do
    python3 logs/cpubusy.py gate 0.8 900 >> $O/gate.txt || echo "gate timeout $name $n $rep" >> $O/gate.txt
    env ${q:+BEAM_QDIR=$q} ${ck:+BEAM_CACHE=$ck BEAM_WARMSEQ=logs/beam/seq.tsv BEAM_WARMT=16} tools/energy/record.py record --hz 20 --idle 5 --json $O/${name}_n${n}_r$rep.json -- $B $D $IDS $n > $O/${name}_n${n}_r$rep.log 2>&1
  done; done
}
run nvfp4 "" ""
run q2 $MODELS/new/beam/q2 ""
run q3 $MODELS/new/beam/q3 ""
run nvfp4_c64 "" 64
run q2_c128 $MODELS/new/beam/q2 128
run q3_c96 $MODELS/new/beam/q3 96
echo BENCHDONE >> $O/gate.txt
