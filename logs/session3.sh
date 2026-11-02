#!/bin/bash
# session2.sh KIND OUTDIR NWIN K [args]: per window: strict gate (<2 busy cores system-wide), tools/energy/record.py record (hz 20, idle 5, csv+json), K back-to-back clips, busy-core average over the window.
# KIND: cos SIZE STEPS | yue
cd ~/strix-rt; kind=$1; O=$2; nw=$3; K=$4; shift 4; export O; mkdir -p $O
for w in $(seq 1 $nw); do
  python3 logs/cpubusy.py gate 0.8 1200 >> $O/gate.txt || { echo "w$w gate timeout" >> $O/gate.txt; continue; }
  b0=$(python3 logs/cpubusy.py snap)
  if [ $kind = cos ]; then cmd="for i in \$(seq 1 $K); do logs/cos/one_clip.sh w${w}_c\$i $1 $2; done"
  else cmd="for i in \$(seq 1 $K); do s=\$(date +%s.%N); timeout 400 target-ye/release/yue2 gen --ref $HOME/yue_ref/out_ref --seed 1234 --out /tmp/yue_e${w}_\$i.wav > $O/w${w}_c\$i.log 2>&1; echo w${w}_c\$i rc=\$? start \$s wall \$(echo \"\$(date +%s.%N) - \$s\" | bc) >> $O/walls.txt; done"; fi
  tools/energy/record.py record --hz 20 --idle 5 --csv $O/w$w.csv --json $O/w$w.json -- bash -c "$cmd" > /dev/null 2>&1
  b1=$(python3 logs/cpubusy.py snap)
  python3 - $b0 $b1 >> $O/busy.txt <<PY
import sys; a=list(map(int,sys.argv[1:3])); b=list(map(int,sys.argv[3:5])); print("w$w avg busy cores incl. our run (window+idle): %.2f" % (32*(b[0]-a[0])/(b[1]-a[1])))
PY
  uptime >> $O/load.txt
done
