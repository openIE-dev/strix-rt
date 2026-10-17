#!/bin/bash
# tools/comp/lb.sh BINDIR OUTTAG MODEL: llama-bench pp512/2048/8192 + tg128, then tg128 at depth 8192; JSON in logs/comp/json/
B=$1; T=$2; M=$3; O=~/strix-rt/logs/comp/json; mkdir -p $O
$B/llama-bench -m $M -ngl 999 -fa on -b 2048 -ub 512 -p 512,2048,8192 -n 128 -r 3 -o json > $O/$T.json 2> $O/$T.err; echo "pp/tg rc $?"
$B/llama-bench -m $M -ngl 999 -fa on -b 2048 -ub 512 -p 0 -n 128 -d 8192 -r 3 -o json > $O/$T.d8192.json 2> $O/$T.d8192.err; echo "depth rc $?"
python3 -c "import json,sys;[print(sys.argv[1],r[\"n_prompt\"],r[\"n_gen\"],r[\"n_depth\"],round(r[\"avg_ts\"],2),round(r[\"stddev_ts\"],2)) for f in sys.argv[2:] for r in json.load(open(f))]" $T $O/$T.json $O/$T.d8192.json
