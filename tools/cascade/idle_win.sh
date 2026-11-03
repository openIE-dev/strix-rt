#!/bin/bash
# One idle attempt (run under logs/gpu_lock.sh): idle_win.sh none|srv N. Keeps the window only if avg busy cores DURING it < 1.0.
cd ~/strix-rt; D=logs/cascade/floor; M=$MODELS/decision; t=$1; n=$2
python3 logs/cpubusy.py gate 1.0 120 >> $D/gate.txt || { sleep 0; exit 3; }
if [ $t = srv ]; then target-dm/release/kev_serve $M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin 127.0.0.1:8131 kev-0.8b > $D/server_idle.log 2>&1 & SP=$!
  trap "kill $SP 2>/dev/null" EXIT; for i in $(seq 60); do curl -s http://127.0.0.1:8131/v1/models >/dev/null && break; sleep 1; done; sleep 5
  python3 logs/cpubusy.py gate 1.0 120 >> $D/gate.txt || exit 3; fi
a=$(python3 logs/cpubusy.py snap); tools/energy/record.py record --hz 20 --idle 5 --json /tmp/iw.json -- sleep 60 > /tmp/iw.out 2>&1; b=$(python3 logs/cpubusy.py snap)
busy=$(python3 -c "a,b=\"$a\".split(),\"$b\".split();print(round(32*(float(b[0])-float(a[0]))/max(float(b[1])-float(a[1]),1),2))")
echo "$(date -u +%H:%M:%S) idle_$t try busy=$busy" >> $D/idle_tries.txt
if python3 -c "import sys;sys.exit(0 if $busy<1.0 else 1)"; then mv /tmp/iw.json $D/idle_${t}_c$n.json; mv /tmp/iw.out $D/idle_${t}_c$n.out; echo "$(date -u +%H:%M:%S) idle_${t}_c$n busy=$busy" >> $D/idle_clean.txt; fi
