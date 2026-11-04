#!/bin/bash
cd ~/strix-rt; O=logs/beam/cbdbg; M=$MODELS/new/beam; B=$M/cbB/q3; Bn=target-beam/release/beam; D=$M/nvfp4; E=logs/beam/evalq
for name in science_chat history german sql logic_chat json; do
  seq=$(awk -F"\t" -v n=$name "\$1==n{print \$2}" logs/beam/seq.tsv); pl=$(awk -F"\t" -v n=$name "\$1==n{print \$3}" logs/beam/seq.tsv)
  BEAM_QDIR=$B BEAM_CBLUT=logs/beam/cbq/lut.txt BEAM_FORCE=$((pl-1)) $Bn $D $seq > $E/$name.cbB.log 2>&1; echo "eval $name $(date -u +%T)" >> $O/evalB.status
done
NAMES="science_chat history german sql logic_chat json" python3 tools/beam_evalq.py q3 cbA cbB > $O/evalq.fixed.txt 2>&1; echo done >> $O/evalB.status
