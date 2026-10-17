#!/bin/sh
# usage: pf_idle_run.sh CMD...   waits until no other GPU job (q38/npu/llama/bench/pf) runs, then runs CMD; reports contention
busy() { pgrep -f "release/(q38|qtest|lfm|npu_|tern_|bench|pf_)|npu/target[a-z_]*/release|llama-bench|llama-cli|build-.*/bin/" >/dev/null; }
w=0; while busy && [ $w -lt ${IDLE_WAIT:-600} ]; do sleep 3; w=$((w+3)); done
echo "[waited ${w}s for idle GPU$(busy && echo ', STILL BUSY')]"
"$@"
echo "[other GPU job at end: $(busy && echo YES || echo no)]"
