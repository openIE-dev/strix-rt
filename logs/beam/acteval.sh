#!/bin/bash
cd ~/strix-rt; while ! grep -q EXIT logs/beam/rq/act_q2.log; do sleep 20; done
O=logs/beam/evalq; B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4; M=$MODELS/new/beam
for name in science_chat history german sql logic_chat json; do
  seq=$(awk -F'\t' -v n=$name '$1==n{print $2}' logs/beam/seq.tsv); pl=$(awk -F'\t' -v n=$name '$1==n{print $3}' logs/beam/seq.tsv)
  BEAM_QDIR=$M/act/q2 BEAM_FORCE=$((pl-1)) $B $D $seq > $O/$name.actq2.log 2>&1
done
echo done > $O/actdone
