#!/bin/bash
exec 9>&-  # never let children inherit the gpu_lock flock fd
# tools/comp/serve_eq.sh MODEL [N] [MAXT] [server args..]: start strix-serve, run serve_eq.py, stop the server (exact PID). Run under gpu_lock.
cd ~/strix-rt; M=$1; N=${2:-8}; T=${3:-48}; shift 3; PORT=18100; L=logs/serve/eq_$(basename $M .gguf).server.log; mkdir -p logs/serve
ZEROCOPY=dmabuf SERVE_LOG=1 target-serve/release/strix-serve $M --port $PORT --alias m "$@" > $L 2>&1 & SP=$!
trap 'kill $SP; wait $SP' EXIT
t0=$(date +%s); until curl -sf --max-time 10 localhost:$PORT/v1/models >/dev/null; do kill -0 $SP 2>/dev/null || { echo "server died"; tail -20 $L; exit 1; }; [ $(( $(date +%s) - t0 )) -gt 600 ] && exit 1; sleep 2; done
echo "ready after $(( $(date +%s) - t0 )) s"
python3 tools/comp/serve_eq.py http://localhost:$PORT $N $T
