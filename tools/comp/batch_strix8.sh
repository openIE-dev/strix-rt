#!/bin/bash
# rounds 6b/7b after the aux-size fix (NB 8 sq_commit scratch for columns 4..7 ran past aux): eq, NB 8 bench, NB 8 + PF_PASSES=2 bench
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
GPU_LOCK_TAG="strix-serve eq6b" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r6b.log 2>&1
grep -q "8/8" logs/serve/eq_q38_r6b.log || { echo EQ_FAIL; exit 1; }
$R serve $S strix_q38_r6b strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
SERVE_PF_PASSES=2 $R serve $S strix_q38_r7b strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
echo BATCH_STRIX8_DONE
