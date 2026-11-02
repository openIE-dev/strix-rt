#!/bin/bash
# cached-run differential energy: warm-up (identical in both windows, ~100 s) cancels in long(N=200)-short(N=40); idle baseline 5 s; gate before each run
cd ~/strix-rt; O=logs/beam/cachee; mkdir -p $O; rm -f $O/*
B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
export BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304 BEAM_PREFILL=1
run() { local name=$1 q=$2 ck=$3 nl=$4 ns=$5
  for rep in 1 2; do for n in $nl $ns; do
    python3 logs/cpubusy.py gate 0.8 900 >> $O/gate.txt || echo "gate timeout $name $n $rep" >> $O/gate.txt
    env ${q:+BEAM_QDIR=$q} ${ck:+BEAM_CACHE=$ck BEAM_WARMSEQ=logs/beam/seq.tsv BEAM_WARMT=16} tools/energy/record.py record --hz 20 --idle 5 --json $O/${name}_n${n}_r$rep.json -- $B $D $IDS $n > $O/${name}_n${n}_r$rep.log 2>&1
  done; done; echo "$name $nl $ns" >> $O/cfg.txt
}
run nvfp4_c64 "" 64 200 40
run q3_c96 $MODELS/new/beam/q3 96 200 40
run q2_c128 $MODELS/new/beam/q2 128 200 40
run nvfp4 "" "" 30 6
echo CEDONE >> $O/gate.txt
