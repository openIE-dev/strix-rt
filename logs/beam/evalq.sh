#!/bin/bash
cd ~/strix-rt; O=logs/beam/evalq; mkdir -p $O; B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4
while IFS=$'\t' read -r name seq pl; do
  BEAM_FORCE=$((pl-1)) $B $D $seq > $O/$name.base.log 2>&1
done < logs/beam/seq.tsv
for q in q2 q3; do
 while IFS=$'\t' read -r name seq pl; do
  BEAM_QDIR=$MODELS/new/beam/$q BEAM_FORCE=$((pl-1)) $B $D $seq > $O/$name.$q.log 2>&1
 done < logs/beam/seq.tsv
 while IFS=$'\t' read -r name ids; do
  BEAM_QDIR=$MODELS/new/beam/$q BEAM_PREFILL=1 $B $D $ids 48 > $O/$name.$q.free.log 2>&1
 done < logs/beam/prompts.tsv
done
echo done > $O/done
