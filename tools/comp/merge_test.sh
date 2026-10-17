#!/bin/bash
exec 9>&-  # never let children (servers, curls) inherit the gpu_lock flock fd
# unified strix-serve: 3 models in one process, route by "model"
cd ~/strix-rt; P=18103; B=target-serve/release/strix-serve; M=$MODELS
MODELS=${MODELS:-"$M/Qwen3.8-27B-UD-Q4_K_M.gguf $M/LFM2.5-24B-A2B-Q4_K_M.gguf $M/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf"}; NAMES=${NAMES:-q38,lfm,mcv}
NEED=${NEED_GB:-60}; AV=$(awk "/MemAvailable/{print int(\$2/1048576)}" /proc/meminfo); echo "MemAvailable ${AV} GB (need ${NEED})"; [ $AV -ge $NEED ] || { echo SKIP: not enough free memory; exit 3; }
timeout 240 $B $MODELS --aliases $NAMES --slots 4 --ctx 4096 --port $P > logs/serve/merge.server.log 2>&1 &
S=$!; for i in $(seq 150); do curl -s --max-time 60 localhost:$P/health >/dev/null && break; sleep 1; done
curl -s --max-time 60 localhost:$P/v1/models; echo
for m in ${NAMES//,/ } nope; do
  echo "== $m"; curl -s --max-time 60 -w " [%{http_code}]" localhost:$P/v1/chat/completions -d "{\"model\":\"$m\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colors.\"}]}" | cut -c1-400; echo
done
# concurrent across models
for m in ${NAMES//,/ }; do curl -s --max-time 60 localhost:$P/v1/chat/completions -d "{\"model\":\"$m\",\"max_tokens\":24,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"Name three primary colors.\"}]}" > /tmp/mc_$m.json & W="$W $!"; done; wait $W
for m in ${NAMES//,/ }; do echo "conc $m: $(python3 -c "import json;print(json.load(open('/tmp/mc_$m.json'))['choices'][0]['message']['content'][:80])" 2>&1)"; done
kill $S; wait $S 2>/dev/null; grep "strix-serve\]" logs/serve/merge.server.log
