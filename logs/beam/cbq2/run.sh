#!/bin/bash
# repair after the beam_requant_cb.py symlink write-through: regenerate original q3, rebuild cbA/cbB as real files, re-eval
cd ~/strix-rt; O=logs/beam/cbq2; S=$O/status; M=~/models/new/beam; P=~/venvs/ml/bin/python; R=$M/q3.regen
st() { echo "$* $(date -u +%T)" >> $S; }
set -o pipefail
mkdir -p $R
# 1) determinism: regenerated L20 must be byte-identical to the untouched original q3/L20.bin
$P logs/beam/cbq2/beam_requant_1e867d4.py 3 $R 20 20 --device cuda > $O/regen20.log 2>&1 || { st "regen20 FAIL"; exit 1; }
cmp $R/L20.bin $M/q3/L20.bin && st "regen L20 identical to original" || { st "regen L20 DIFFERS - abort"; exit 1; }
rm $R/L20.bin
# 2) regenerate the overwritten layers
$P logs/beam/cbq2/beam_requant_1e867d4.py 3 $R 3 19 --device cuda > $O/regen_a.log 2>&1 || { st "regen 3-19 FAIL"; exit 1; }; st "regen 3-19"
$P logs/beam/cbq2/beam_requant_1e867d4.py 3 $R 30 51 --device cuda > $O/regen_b.log 2>&1 || { st "regen 30-51 FAIL"; exit 1; }; st "regen 30-51"
for f in $R/L*.bin; do b=$(basename $f); [ -L $M/q3/$b ] && { st "q3/$b is a link - abort"; exit 1; }; mv -f $f $M/q3/$b; done; rmdir $R
( cd $M/q3; ls -la; sha256sum L*.bin ) > $O/q3.manifest; st "q3 restored ($(ls $M/q3/L*.bin | wc -l) files)"
# 3) keep the old all-layers-cb model as cbBall; cbA becomes real copies
mv $M/cbB $M/cbBall; mkdir -p $M/cbB/q3
for l in $(seq 3 51); do b=$(printf L%02d.bin $l)
  if [ $l -lt 20 ] || [ $l -gt 29 ]; then
    for d in cbA cbB; do cp $M/q3/$b $M/$d/q3/.$b.tmp && mv -f $M/$d/q3/.$b.tmp $M/$d/q3/$b; done
  fi; done
st "cbA/cbB copies; links left: $(find $M/cbA $M/cbB $M/cbBall -type l | wc -l)"
# 4) fixed requant 20-29
$P tools/beam_requant_cb.py $M/cbA/q3 $M/cbB/q3 $O/lut.txt --layers 20-29 --device cuda > $O/requant.log 2>&1 || { st "requant FAIL"; exit 1; }; st "requant 20-29"
# 5) exactness + eval
~/.cargo/bin/cargo build --release -p beam --target-dir target-beam > $O/build.log 2>&1 || { st "build FAIL"; exit 1; }
Bn=target-beam/release/beam; D=$M/nvfp4; E=logs/beam/evalq; : > $O/empty.txt
for name in science_chat history german sql logic_chat json; do
  seq=$(awk -F"\t" -v n=$name "\$1==n{print \$2}" logs/beam/seq.tsv); pl=$(awk -F"\t" -v n=$name "\$1==n{print \$3}" logs/beam/seq.tsv); F=$((pl-1))
  BEAM_QDIR=$M/q3 BEAM_FORCE=$F $Bn $D $seq > $E/$name.q3.log 2>&1
  BEAM_QDIR=$M/cbA/q3 BEAM_FORCE=$F $Bn $D $seq > $E/$name.cbA.log 2>&1
  BEAM_QDIR=$M/cbB/q3 BEAM_CBLUT=$O/lut.txt BEAM_FORCE=$F $Bn $D $seq > $E/$name.cbB.log 2>&1
  BEAM_QDIR=$M/cbBall/q3 BEAM_CBLUT=logs/beam/cbq/lut.txt BEAM_FORCE=$F $Bn $D $seq > $E/$name.cbBall.log 2>&1
  if [ $name = sql ]; then
    BEAM_QDIR=$M/cbA/q3 BEAM_FORCE=$F $Bn $D $seq > $O/sql.cbA2.log 2>&1
    BEAM_QDIR=$M/cbA/q3 BEAM_CBLUT=$O/empty.txt BEAM_FORCE=$F $Bn $D $seq > $O/sql.cbAlut.log 2>&1
    { cmp <(grep "^F " $E/sql.cbA.log) <(grep "^F " $O/sql.cbA2.log) && echo "cbA rerun identical" || echo "cbA rerun DIFFERS"
      cmp <(grep "^F " $E/sql.cbA.log) <(grep "^F " $O/sql.cbAlut.log) && echo "uniform LUT path identical to cbA" || echo "uniform LUT path DIFFERS"; } > $O/exact.txt
  fi
  st "eval $name"
done
NAMES="science_chat history german sql logic_chat json" python3 tools/beam_evalq.py q3 cbA cbB cbBall > $O/evalq.txt 2>&1
st done
