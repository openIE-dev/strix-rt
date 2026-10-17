#!/bin/bash
exec 9>&-  # never let children (servers, curls) inherit the gpu_lock flock fd
# tools/comp/serve1_test.sh MODEL [IMG]: start strix-serve1, run serve1_test.py, stop the server (exact PID). Run under gpu_lock.
cd ~/strix-rt; M=$1; PORT=18101; L=logs/serve/s1_$(basename $M .gguf).server.log; mkdir -p logs/serve
ZEROCOPY=dmabuf target/release/strix-serve1 $M --port $PORT --alias m --ctx 4096 > $L 2>&1 & SP=$!
trap 'kill $SP; wait $SP' EXIT
t0=$(date +%s); until curl -sf --max-time 10 localhost:$PORT/v1/models >/dev/null; do kill -0 $SP 2>/dev/null || { echo "server died"; tail -20 $L; exit 1; }; [ $(( $(date +%s) - t0 )) -gt 600 ] && exit 1; sleep 2; done
echo "ready after $(( $(date +%s) - t0 )) s"
python3 tools/comp/serve1_test.py http://localhost:$PORT $2
