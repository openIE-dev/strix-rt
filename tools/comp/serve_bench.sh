#!/bin/bash
exec 9>&-  # never let children inherit the gpu_lock flock fd
# tools/comp/serve_bench.sh TAG KIND MODEL [server args..]   KIND: vllm | llama-vk | llama-hip | flm
# Starts the server, waits for /v1/models, runs conc_bench at C=1,2,4,8 (each under tools/energy/record.py record), stops the server
# (llama/flm: kill by exact PID; vllm: docker stop by exact container id).
cd ~/strix-rt; T=$1; K=$2; M=$3; shift 3; PORT=18000; O=logs/comp/serve; mkdir -p $O; L=$O/$T.server.log; NAME=m
case $K in
vllm) CID=$(docker run -d --device /dev/kfd --device /dev/dri --group-add video --group-add render --ipc=host --security-opt seccomp=unconfined \
        -v $MODELS/hf:/models ${VLLM_DOCKER_EXTRA} -p $PORT:8000 --entrypoint /opt/venv/bin/vllm docker.io/kyuz0/vllm-therock-gfx1151:latest \
        serve /models/$M --served-model-name m --host 0.0.0.0 --port 8000 "$@"); echo "container $CID"
      stop() { docker logs $CID > $L 2>&1; docker stop $CID >/dev/null; docker rm $CID >/dev/null; } ; alive() { docker inspect -f "{{.State.Running}}" $CID 2>/dev/null | grep -q true; } ;;
llama-vk) ~/llama-latest/build-vulkan/bin/llama-server -m $M --alias m -ngl 999 -fa on --port $PORT --no-webui "$@" > $L 2>&1 & SP=$!
      stop() { kill $SP; wait $SP; } ; alive() { kill -0 $SP 2>/dev/null; } ;;
llama-hip) ( source tools/comp/rocm713.env; exec ~/llama-latest/build-rocm713/bin/llama-server -m $M --alias m -ngl 999 -fa on --port $PORT --no-webui "$@" ) > $L 2>&1 & SP=$!
      stop() { kill $SP; wait $SP; } ; alive() { kill -0 $SP 2>/dev/null; } ;;
flm) flm serve $M --port $PORT "$@" > $L 2>&1 & SP=$!; NAME=$M
      stop() { kill $SP; wait $SP; } ; alive() { kill -0 $SP 2>/dev/null; } ;;
esac
trap stop EXIT
t0=$(date +%s); until curl -sf --max-time 10 localhost:$PORT/v1/models >/dev/null; do alive || { echo "SERVE $T: server died"; tail -n 20 $L 2>/dev/null; docker logs ${CID:-x} 2>&1 | tail -n 20; exit 1; }; [ $(( $(date +%s) - t0 )) -gt 1500 ] && { echo "SERVE $T: start timeout"; exit 1; }; sleep 3; done
echo "SERVE $T: ready after $(( $(date +%s) - t0 )) s"
DC="1 2 4 8"; [ $K = flm ] && DC="1 2"  # FLM serializes requests (C=1,2,4 identical 8.01 tok/s on gpt-oss)
for C in ${CONC:-$DC}; do
  tools/energy/record.py record --hz 10 --json $O/$T.c$C.energy.json -- ~/.local/bin/uv run -q tools/comp/conc_bench.py http://localhost:$PORT $NAME $C --out $O/$T.c$C.json > $O/$T.c$C.out 2>&1
  python3 -c "import json,sys;r=json.load(open(sys.argv[1]))[0];d=json.load(open(sys.argv[2]));ch={c[\"fabric\"]:c for c in d[\"breakdown\"][\"channels\"]};n=12*$C*r[\"mean_ctoks\"];print(\"SERVE $T C=$C:\",json.dumps({k:(round(v,2) if isinstance(v,float) else v) for k,v in r.items()}),\"| socket J/out-tok (all 12 waves incl. prefill)\",round(ch[\"socket\"][\"joules\"]/n,3))" $O/$T.c$C.json $O/$T.c$C.energy.json || tail -n 5 $O/$T.c$C.out
done
