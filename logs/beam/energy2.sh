#!/bin/bash
# differential energy, 2 reps each: warm page cache (default reader) and sustained-cold (O_DIRECT, no cache). long=40 new, short=2 new.
cd ~/strix-rt; O=logs/beam/energy2; mkdir -p $O; rm -f $O/*
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
for mode in warm direct; do
 [ $mode = direct ] && export BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304
 target-beam/release/beam $MODELS/new/beam/nvfp4 $IDS 40 > /dev/null 2>&1   # pre-warm / settle
 for rep in 1 2; do for n in 40 2; do
  python3 logs/cpubusy.py gate 0.8 900 >> $O/gate.txt || echo "gate timeout $mode $n $rep" >> $O/gate.txt
  tools/energy/record.py record --hz 20 --idle 5 --json $O/${mode}_n${n}_r$rep.json -- target-beam/release/beam $MODELS/new/beam/nvfp4 $IDS $n > $O/${mode}_n${n}_r$rep.log 2>&1
 done; done
done
echo ENERGYDONE >> $O/gate.txt
