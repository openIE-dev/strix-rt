#!/bin/bash
cd ~/strix-rt; O=logs/beam/pre; mkdir -p $O; rm -f $O/*.log
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
B=target-beam/release/beam; M=$MODELS/new/beam/nvfp4
BEAM_EVICT=1 $B $M $IDS 1 > $O/tokbytok.log 2>&1
BEAM_EVICT=1 BEAM_PREFILL=1 $B $M $IDS 1 > $O/prefill.log 2>&1
# longer prompt (40 tokens) : chat prompt repeated words
L=$IDS,$(python3 -c "print(','.join(str(x) for x in [785,6722,315,9625,374,6722,315,15344,374,6722,315,9856,374]*2))")
BEAM_EVICT=1 BEAM_PREFILL=1 $B $M $L 1 > $O/prefill_long.log 2>&1
BEAM_EVICT=1 $B $M $L 1 > $O/tokbytok_long.log 2>&1
echo done > $O/done
