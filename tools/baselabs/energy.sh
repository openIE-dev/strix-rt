#!/bin/bash
# Differential J/decision for Base Labs models via bl_run: windows lo (BL_REPS=LO) and hi (BL_REPS=HI) over REF items,
# each behind the quiet-box gate and tools/energy/record.py, like tools/cascade/session.sh. Run under logs/gpu_lock.sh.
# Usage: energy.sh NAME MODEL.gguf REF.jsonl LO HI REP
cd ~/strix-rt; N=$1; G=$2; R=$3; D=logs/baselabs/energy; n=$(( $(wc -l < $R) - 1 ))
for w in lo hi; do [ $w = lo ] && p=$4 || p=$5; tag=${N}_r$6.$w
  python3 logs/cpubusy.py gate 1.0 600 >> $D/gate.txt || { echo "gate timeout $tag"; exit 3; }
  BL_REPS=$p /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $D/$tag.json -- target-dm/release/bl_run $G $R > $D/$tag.out 2>&1
  echo "$p $n" > $D/$tag.meta; done
