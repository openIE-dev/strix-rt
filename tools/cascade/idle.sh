#!/bin/bash
# Idle power, kev_serve kev-0.8b loaded vs no server, 3 alternating 60 s gated windows. Run under logs/gpu_lock.sh.
cd ~/strix-rt; D=logs/cascade/floor; M=$MODELS/decision
win() { python3 logs/cpubusy.py gate 1.0 600 >> $D/gate.txt || exit 3; t=$1; shift; /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $D/$t.json -- "$@" > $D/$t.out 2>&1; }
for r in 1 2 3; do
  win idle_none_r$r sleep 60
  target-dm/release/kev_serve $M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin 127.0.0.1:8131 kev-0.8b > $D/server_idle.log 2>&1 & SP=$!
  for i in $(seq 60); do curl -s http://127.0.0.1:8131/v1/models >/dev/null && break; sleep 1; done; sleep 5
  win idle_srv_r$r sleep 60; kill $SP; wait $SP 2>/dev/null; sleep 5
done; echo IDLE DONE
