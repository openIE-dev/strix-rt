#!/bin/bash
# torch-profiler traces of vLLM gpt-oss-20b at C=4 and C=5 (the 4->5 step). Run via run.sh (wait_quiet+gpu_lock+perf_high).
cd ~/strix-rt; PORT=18001; P=$HOME/strix-rt/logs/comp/vprof; mkdir -p $P; chmod 777 $P
CID=$(docker run -d --rm --device /dev/kfd --device /dev/dri --group-add video --group-add render --ipc=host --security-opt seccomp=unconfined \
  -v $MODELS/hf:/models -v $P:/prof -p $PORT:8000 --entrypoint /opt/venv/bin/vllm docker.io/kyuz0/vllm-therock-gfx1151:latest \
  serve /models/gpt-oss-20b --served-model-name m --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.6 \
  --profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/prof\"}")
echo "PROF container $CID"; stop(){ docker logs $CID > $P/server.log 2>&1; docker stop $CID >/dev/null; }; trap stop EXIT
t0=$(date +%s); until curl -sf localhost:$PORT/v1/models >/dev/null; do sleep 5; [ $(( $(date +%s)-t0 )) -gt 1500 ] && { echo "PROF timeout"; exit 1; }; done
for C in 4 5; do
  ~/.local/bin/uv run -q tools/comp/conc_bench.py http://localhost:$PORT m $C --out $P/warm.c$C.json >/dev/null 2>&1
  curl -sf -X POST localhost:$PORT/start_profile; mkdir -p $P/c$C
  ~/.local/bin/uv run -q tools/comp/conc_bench.py http://localhost:$PORT m $C --out $P/c$C.json > $P/c$C.out 2>&1
  curl -sf -X POST localhost:$PORT/stop_profile; sleep 20; mv $P/*.json.gz $P/c$C/ 2>/dev/null; find $P -newer $P/c$C.out -name "*trace*" -not -path "*/c[45]/*" -exec mv {} $P/c$C/ \;
  echo "PROF C=$C: $(tail -c 300 $P/c$C.out)"
done
