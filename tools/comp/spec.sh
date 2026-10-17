#!/bin/bash
# tools/comp/spec.sh BINDIR TAG MODEL PROMPT(json: string or id list) [server flags..]: llama-server greedy 128-token decode, 3 runs,
# prints decode tok/s and draft acceptance from the response timings. Server killed by its exact PID.
B=$1; T=$2; M=$3; P=$4; shift 4; PORT=18080
$B/llama-server -m $M -ngl 999 -fa on -c 4096 -np 1 --port $PORT --no-webui "$@" > ~/strix-rt/logs/comp/server_$T.log 2>&1 & SP=$!
trap "kill $SP 2>/dev/null; wait $SP 2>/dev/null" EXIT
for i in $(seq 300); do curl -sf localhost:$PORT/health >/dev/null && break; kill -0 $SP 2>/dev/null || { echo "SPEC $T: server exited"; tail -3 ~/strix-rt/logs/comp/server_$T.log; exit 1; }; sleep 1; done
for r in 0 1 2 3; do
  curl -s localhost:$PORT/completion -d "{\"prompt\": $P, \"n_predict\": 128, \"temperature\": 0, \"ignore_eos\": true, \"cache_prompt\": false}" |
  python3 -c "import json,sys;d=json.load(sys.stdin);t=d[\"timings\"];print(\"SPEC $T run $r: decode\",round(t[\"predicted_per_second\"],2),\"tok/s\",t[\"predicted_n\"],\"tok, draft\",t.get(\"draft_n\"),\"accepted\",t.get(\"draft_n_accepted\"))"
done
