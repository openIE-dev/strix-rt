#!/bin/bash
# hidden.sh T1...: collect hidden states for the probe router (GPU lock, no energy gate)
cd ~/strix-rt; M=$MODELS/decision; K=target-dm/release/kev_serve; D=logs/cascade/hidden; mkdir -p $D
for t in "$@"; do case $t in kev08) a="$M/kev-0.8b-bf16.gguf $M/kev-0.8b-head.bin"; m=kev-0.8b;; kev4b) a="$M/kev-4b-bf16.gguf $M/kev-4b-head.bin"; m=kev-4b;;
  kev9q8) a="$M/kev-9b-q8_0.gguf $M/kev-9b-head.bin"; m=kev-9b;; esac
  [ -s $D/$t.jsonl ] && continue
  logs/gpu_lock.sh sh -c "$K $a 127.0.0.1:8130 $m > $D/server_$t.log 2>&1 & P=\$!; for i in \$(seq 120); do curl -sf localhost:8130/healthz >/dev/null && break; sleep 1; done; python3 tools/cascade/hidden_run.py http://127.0.0.1:8130 $m $D/$t.jsonl.tmp && mv $D/$t.jsonl.tmp $D/$t.jsonl; kill \$P"
done; echo "$(date +%T) hidden done $*" >> logs/cascade/session.log
