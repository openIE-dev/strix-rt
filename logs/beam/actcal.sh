#!/bin/bash
cd ~/strix-rt; mkdir -p logs/beam/act; B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4
for name in paris_raw paris_chat math_chat code_py code_chat prose; do
  seq=$(awk -F'\t' -v n=$name '$1==n{print $2}' logs/beam/seq.tsv)
  BEAM_ACTOUT=logs/beam/act/$name.f32 BEAM_FORCE=999 $B $D $seq > logs/beam/act/$name.log 2>&1
done
echo done > logs/beam/act/done
