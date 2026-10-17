#!/bin/bash
# decode power sweep 3 (perf high): more rows per workgroup (fewer x re-reads), m=1..2 -> logs/serve/dec_power3_q38.log. Under gpu_lock.
cd ~/strix-rt; M=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; B=target-serve/release/serve_steps
run() { echo "== $1"; shift; env SPOW=150 ZEROCOPY=dmabuf "$@" tools/comp/perf_high.sh timeout 100 $B $M 2 600 2>&1 | grep -E "power"; }
run "A r9 (8,128 16,256 16,256 8,128)" GEMV_PROJ=8,128 GEMV_OUT=16,256 GEMV_SWIGLU=16,256 GEMV_DOWN=8,128
run "B (8,256 16,512 16,512 8,256)" GEMV_PROJ=8,256 GEMV_OUT=16,512 GEMV_SWIGLU=16,512 GEMV_DOWN=8,256
run "C (4,128 8,256 8,256 4,128)" GEMV_PROJ=4,128 GEMV_OUT=8,256 GEMV_SWIGLU=8,256 GEMV_DOWN=4,128
run "D (8,256 x4)" GEMV_PROJ=8,256 GEMV_OUT=8,256 GEMV_SWIGLU=8,256 GEMV_DOWN=8,256
run "A again" GEMV_PROJ=8,128 GEMV_OUT=16,256 GEMV_SWIGLU=16,256 GEMV_DOWN=8,128
