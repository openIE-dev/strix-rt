#!/bin/bash
# start vLLM on MODEL without --rm, capture logs on exit/ready, remove container by exact id
M=$1; L=~/strix-rt/logs/comp/serve/diag_$M.log
CID=$(docker run -d --device /dev/kfd --device /dev/dri --group-add video --group-add render --ipc=host --security-opt seccomp=unconfined \
  -v $MODELS/hf:/models -p 18002:8000 --entrypoint /opt/venv/bin/vllm docker.io/kyuz0/vllm-therock-gfx1151:latest \
  serve /models/$M --served-model-name m --host 0.0.0.0 --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.6)
for i in $(seq 200); do sleep 5; curl -sf localhost:18002/v1/models >/dev/null && { echo "DIAG $M ready"; break; }; [ "$(docker inspect -f {{.State.Running}} $CID)" = true ] || { echo "DIAG $M exited"; break; }; done
docker logs $CID > $L 2>&1; docker stop $CID >/dev/null 2>&1; docker rm $CID >/dev/null
