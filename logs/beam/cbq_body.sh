#!/bin/bash
cd ~/strix-rt; O=logs/beam/cbq; S=$O/status; M=$MODELS/new/beam; A=$M/cbA/q3; B=$M/cbB/q3; L0=20; L1=29
~/.cargo/bin/cargo build --release -p beam --target-dir target-beam > $O/build.log 2>&1 || { echo "build FAIL" >> $S; exit 1; }
echo "built $(date -u +%T)" >> $S
mkdir -p $A $B; : > $O/lut.txt
for l in $(seq 3 51); do f=$(printf L%02d.bin $l); if [ $l -lt $L0 ] || [ $l -gt $L1 ]; then ln -sfn $M/q3/$f $A/$f; ln -sfn $M/q3/$f $B/$f; fi; done
~/venvs/ml/bin/python tools/beam_requant_cb.py $L0 $L1 $A $B $O/lut.txt --device cuda > $O/requant.log 2>&1 || { echo "requant FAIL" >> $S; exit 1; }
echo "requant $(date -u +%T)" >> $S
Bn=target-beam/release/beam; D=$M/nvfp4; E=logs/beam/evalq
# exactness of the LUT path: uniform LUT (empty file) must reproduce cbA on sql
: > $O/empty.txt
for name in science_chat history german sql logic_chat json; do
  seq=$(awk -F'\t' -v n=$name '$1==n{print $2}' logs/beam/seq.tsv); pl=$(awk -F'\t' -v n=$name '$1==n{print $3}' logs/beam/seq.tsv)
  BEAM_QDIR=$A BEAM_FORCE=$((pl-1)) $Bn $D $seq > $E/$name.cbA.log 2>&1
  BEAM_QDIR=$B BEAM_CBLUT=$O/lut.txt BEAM_FORCE=$((pl-1)) $Bn $D $seq > $E/$name.cbB.log 2>&1
  [ $name = sql ] && BEAM_QDIR=$A BEAM_CBLUT=$O/empty.txt BEAM_FORCE=$((pl-1)) $Bn $D $seq > $E/$name.cbAlut.log 2>&1
  echo "eval $name $(date -u +%T)" >> $S
done
NAMES="science_chat history german sql logic_chat json" python3 tools/beam_evalq.py q3 cbA cbB > $O/evalq.txt 2>&1
cmp <(grep '^F ' $E/sql.cbA.log) <(grep '^F ' $E/sql.cbAlut.log) && echo "LUT path bit-identical to uniform on sql" >> $O/evalq.txt || echo "LUT path DIFFERS on sql" >> $O/evalq.txt
echo "done $(date -u +%T)" >> $S
