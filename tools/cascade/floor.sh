#!/bin/bash
# Run under logs/gpu_lock.sh. (a) curl smoke test of POST /v1/decisions; (b) differential energy of K=8 vs K=1 questions per
# state (lo 10 states, hi 50 states); (c) idle power 60 s with kev_serve loaded vs no server. Output logs/cascade/floor/.
cd ~/strix-rt; D=logs/cascade/floor; mkdir -p $D; M=$MODELS/decision; U=http://127.0.0.1:8131
srv() { target-dm/release/kev_serve $M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin 127.0.0.1:8131 kev-0.8b > $D/server.log 2>&1 & SP=$!; for i in $(seq 60); do curl -s $U/v1/models >/dev/null && return; sleep 1; done; }
win() { python3 logs/cpubusy.py gate 1.0 600 >> $D/gate.txt || { echo "gate timeout $1"; exit 3; }; t=$1; shift; /usr/local/bin/tools/energy/record.py record --hz 20 --idle 5 --json $D/$t.json -- "$@" > $D/$t.out 2>&1; }
srv
curl -s $U/v1/decisions -H "Content-Type: application/json" -d "{\"model\":\"kev-0.8b\",\"input\":\"Refund my order now, this is the third time I am asking!\",\"questions\":[{\"type\":\"predicate\",\"name\":\"angry\",\"instructions\":\"Is the customer angry?\"},{\"type\":\"choice\",\"name\":\"team\",\"instructions\":\"Which team should handle it?\",\"choices\":[{\"value\":\"billing\",\"description\":\"payments and refunds\"},{\"value\":\"shipping\"},{\"value\":\"tech\"}]},{\"type\":\"score\",\"name\":\"urgency\",\"instructions\":\"How urgent is it?\",\"levels\":[\"low\",\"medium\",\"high\"]}]}" > $D/curl.json; echo >> $D/curl.json
curl -s $U/v1/decisions -H "Content-Type: application/json" -d "{\"input\":\"x\",\"questions\":[]}" >> $D/curl.json; echo >> $D/curl.json
for r in 1 2; do for k in 8 1; do win k${k}_r$r.lo python3 tools/cascade/floor.py $U $k 10; win k${k}_r$r.hi python3 tools/cascade/floor.py $U $k 50; done; done
win idle_srv python3 -c "import time; time.sleep(60)"
kill $SP; wait $SP 2>/dev/null; sleep 3
win idle_none python3 -c "import time; time.sleep(60)"
echo FLOOR DONE
