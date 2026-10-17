#!/bin/bash
# decode power experiment (q38, multi step m=1..4, 150 steps each with package power sampled): perf level high vs auto,
# and fewer GEMV lanes per row. -> logs/serve/dec_power_q38.log. Run under gpu_lock.
cd ~/strix-rt; M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; B=target-serve/release/serve_steps
run() { echo "== $1"; shift; env SPOW=150 ZEROCOPY=dmabuf "$@" timeout 120 $B $M 4 600 2>&1 | grep -E "^m=|power"; }
run "default, perf high" tools/comp/perf_high.sh
run "default, perf auto"
run "lanes/2 (PROJ 8,64 OUT 16,128 SWIGLU 16,128 DOWN 8,64), perf high" GEMV_PROJ=8,64 GEMV_OUT=16,128 GEMV_SWIGLU=16,128 GEMV_DOWN=8,64 tools/comp/perf_high.sh
run "lanes/2, perf auto" GEMV_PROJ=8,64 GEMV_OUT=16,128 GEMV_SWIGLU=16,128 GEMV_DOWN=8,64
