#!/bin/bash
cd ~/strix-rt
while kill -0 $1 2>/dev/null; do sleep 15; done
GPU_LOCK_TAG=pf_load logs/wait_quiet.sh logs/gpu_lock.sh tools/comp/perf_high.sh tools/comp/pf_load.sh > logs/serve/pf_load_q38.log 2>&1
