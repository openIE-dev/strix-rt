#!/bin/bash
cd ~/strix-rt; O=logs/beam/evalq; B=target-beam/release/beam; D=$MODELS/new/beam/nvfp4; M=$MODELS/new/beam
BN="science_chat history german sql logic_chat json"
for hk in 64 32; do
 for name in $BN; do
  seq=$(awk -F'\t' -v n=$name '$1==n{print $2}' logs/beam/seq.tsv); pl=$(awk -F'\t' -v n=$name '$1==n{print $3}' logs/beam/seq.tsv)
  BEAM_QDIR=$M/q2 BEAM_QHI=$M/q3 BEAM_QHOT=logs/beam/hotA$hk.txt BEAM_FORCE=$((pl-1)) $B $D $seq > $O/$name.mix$hk.log 2>&1
 done
done
IDS=151669,872,151671,785,6722,315,9625,374,151670,151669,77091
BEAM_QDIR=$M/q2 BEAM_QHI=$M/q3 BEAM_QHOT=logs/beam/hotA64.txt BEAM_DIRECT=1 BEAM_RD_THREADS=16 BEAM_RD_CH=4194304 BEAM_PREFILL=1 $B $D $IDS 40 > $O/speed.mix64.log 2>&1
echo done > $O/mixdone
