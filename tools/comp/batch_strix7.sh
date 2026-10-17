#!/bin/bash
# strix-serve round 7: prefill priority (SERVE_PF_PASSES=2 packed passes per decode step) on top of NB 8. Waits for round 6 (pid arg).
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
while kill -0 ${1:-0} 2>/dev/null; do sleep 15; done
SERVE_PF_PASSES=2 $R serve $S strix_q38_r7 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
echo BATCH_STRIX7_DONE
