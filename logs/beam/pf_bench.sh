#!/bin/bash
cd ~/strix-rt; O=logs/beam/pf; mkdir -p $O
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
B=target-beam/release/beam; M=$MODELS/new/beam/nvfp4
for cfg in "0 0" "1 12" "2 12" "3 16" "0 0" "2 12"; do set -- $cfg
  BEAM_EVICT=1 BEAM_PF_DEPTH=$1 BEAM_PF_TOP=$2 $B $M $IDS 24 > $O/cold_d$1_t$2_$(date +%s).log 2>&1
done
echo PFDONE > $O/done
