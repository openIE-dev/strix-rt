#!/bin/bash
# llama.cpp Vulkan greedy reference ids for gpt-oss-20b (raw prompt, no template): logs/gptoss/{prompt,ref}_ids.txt
cd ~/strix-rt; V=~/llama-latest/build-vulkan/bin; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf; PORT=8391; D=logs/gptoss
$V/llama-server -m $M -ngl 999 -fa on -c 4096 -np 1 --port $PORT --no-webui > $D/ref_server.log 2>&1 &
SP=$!
for i in $(seq 120); do curl -s localhost:$PORT/health | grep -q ok && break; sleep 1; done
TXT="The history of the Roman Empire spans more than a thousand years. In this essay, we will explore"
curl -s localhost:$PORT/tokenize -d "{\"content\": \"$TXT\", \"add_special\": false}" | python3 -c "import json,sys; print(\",\".join(map(str,json.load(sys.stdin)[\"tokens\"])))" > $D/prompt_ids.txt
P=$(cat $D/prompt_ids.txt)
curl -s localhost:$PORT/completion -d "{\"prompt\": [$P], \"n_predict\": 128, \"temperature\": 0, \"top_k\": 1, \"samplers\": [\"top_k\"], \"return_tokens\": true, \"cache_prompt\": false}" > $D/ref.json
python3 -c "import json; d=json.load(open(\"$D/ref.json\")); print(\",\".join(map(str,d[\"tokens\"])))" > $D/ref_ids.txt
kill $SP; wait $SP 2>/dev/null
echo "prompt $(cat $D/prompt_ids.txt | tr , \"\\n\" | wc -l) ref $(cat $D/ref_ids.txt | tr , \"\\n\" | wc -l)"
