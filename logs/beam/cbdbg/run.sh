#!/bin/bash
# LUT-path debug: determinism of cbA and uniform-LUT exactness on sql. Usage: run.sh TAG
cd ~/strix-rt; O=logs/beam/cbdbg; T=$1; M=$MODELS/new/beam; A=$M/cbA/q3; Bn=target-beam/release/beam; D=$M/nvfp4
~/.cargo/bin/cargo build --release -p beam --target-dir target-beam > $O/build.$T.log 2>&1 || { echo "build FAIL" > $O/$T.res; exit 1; }
seq=$(awk -F"\t" "\$1==\"sql\"{print \$2}" logs/beam/seq.tsv); pl=$(awk -F"\t" "\$1==\"sql\"{print \$3}" logs/beam/seq.tsv)
[ -f $O/sql.cbA2.log ] || BEAM_QDIR=$A BEAM_FORCE=$((pl-1)) $Bn $D $seq > $O/sql.cbA2.log 2>&1
BEAM_QDIR=$A BEAM_CBLUT=logs/beam/cbq/empty.txt BEAM_FORCE=$((pl-1)) $Bn $D $seq > $O/sql.lut.$T.log 2>&1
E=logs/beam/evalq
{ cmp -s <(grep "^F " $E/sql.cbA.log) <(grep "^F " $O/sql.cbA2.log) && echo "cbA rerun identical" || echo "cbA rerun DIFFERS";
  cmp -s <(grep "^F " $E/sql.cbA.log) <(grep "^F " $O/sql.lut.$T.log) && echo "LUT[$T] identical to cbA" || echo "LUT[$T] DIFFERS"; } > $O/$T.res
