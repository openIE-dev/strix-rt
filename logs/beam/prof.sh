#!/bin/bash
# baseline (NVFP4) greedy generation + expert routing trace for 12 diverse prompts
cd ~/strix-rt; O=logs/beam/prof; mkdir -p $O; rm -f $O/*.log $O/done
while IFS=$'\t' read -r name ids; do
  BEAM_PREFILL=1 BEAM_TRACE=1 target-beam/release/beam $MODELS/new/beam/nvfp4 $ids 48 > $O/$name.log 2>&1
done < logs/beam/prompts.tsv
echo done > $O/done
