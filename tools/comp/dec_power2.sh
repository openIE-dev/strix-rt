#!/bin/bash
# decode power sweep 2 (perf high): GEMV lanes/workgroup variants, m=1..4 -> logs/serve/dec_power2_q38.log. Run under gpu_lock.
cd ~/strix-rt; M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; B=target-serve/release/serve_steps
run() { echo "== $1"; shift; env SPOW=150 ZEROCOPY=dmabuf "$@" tools/comp/perf_high.sh timeout 120 $B $M 4 600 2>&1 | grep -E "power"; }
run "lanes/4 (4,64 8,128 8,128 4,64)" GEMV_PROJ=4,64 GEMV_OUT=8,128 GEMV_SWIGLU=8,128 GEMV_DOWN=4,64
run "lanes/2 ls x2 (8,128 16,256 16,256 8,128)" GEMV_PROJ=8,128 GEMV_OUT=16,256 GEMV_SWIGLU=16,256 GEMV_DOWN=8,128
run "lanes x2 (32,128 32,256 32,256 32,128)" GEMV_PROJ=32,128 GEMV_OUT=32,256 GEMV_SWIGLU=32,256 GEMV_DOWN=32,128
run "lanes/2 + head 4,256" GEMV_PROJ=8,64 GEMV_OUT=16,128 GEMV_SWIGLU=16,128 GEMV_DOWN=8,64 GEMV_HEAD=4,256
