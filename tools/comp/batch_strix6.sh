#!/bin/bash
# strix-serve round 6: gemvq NB up to 8 (one weight stream for 8 decode columns). Waits for round 5 (pid arg), rebuilds
# target-serve, step times NB 4 vs 8, eq recheck, q38 concurrency.
cd ~/strix-rt; R=tools/comp/run.sh; S=tools/comp/serve_bench_strix.sh; MD=$MODELS
while kill -0 ${1:-0} 2>/dev/null; do sleep 15; done
source ~/.cargo/env; CARGO_TARGET_DIR=target-serve cargo build --release -p serve --bin strix-serve --bin serve_steps 2>&1 | tail -2
for nb in 4 8; do GPU_LOCK_TAG=serve_steps_nb$nb logs/wait_quiet.sh logs/gpu_lock.sh env SERVE_NB=$nb ZEROCOPY=dmabuf timeout 120 target-serve/release/serve_steps $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 600 > logs/serve/steps_q38_nb$nb.log 2>&1; done
GPU_LOCK_TAG="strix-serve eq6" logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/serve_eq.sh $MD/Qwen3.8-27B-UD-Q4_K_M.gguf 8 48 > logs/serve/eq_q38_r6.log 2>&1
$R serve $S strix_q38_r6 strix $MD/Qwen3.8-27B-UD-Q4_K_M.gguf --slots 8 --ctx 2048
echo BATCH_STRIX6_DONE
