#!/bin/bash
# differential energy: long (40 new) vs short (2 new) decode, same prompt, page cache warm from earlier runs
cd ~/strix-rt; O=logs/beam/energy; mkdir -p $O
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
for rep in 1 2; do for n in 40 2; do
  python3 logs/cpubusy.py gate 0.8 600 >> $O/gate.txt || echo "gate timeout" >> $O/gate.txt
  tools/energy/record.py record --hz 20 --idle 5 --json $O/n${n}_r$rep.json -- target-beam/release/beam $MODELS/new/beam/nvfp4 $IDS $n > $O/n${n}_r$rep.log 2>&1
done; done
echo ENERGYDONE >> $O/gate.txt
