#!/bin/bash
# 2026-10-30 evening quiet rerun: ABM B, spec, NPU (each step takes gpu_lock itself).
cd ~/strix-rt; echo "start $(date -u +%T)" > logs/quiet_rerun.status
GPU_LOCK_TAG="beam abm B quiet" bash logs/gpu_lock.sh bash logs/beam/abm_b.sh; echo "abm $(date -u +%T)" >> logs/quiet_rerun.status
bash tools/spec/driver.sh > logs/spec/driver.log 2>&1; echo "spec $(date -u +%T)" >> logs/quiet_rerun.status
bash tools/npu/driver.sh > logs/npu/driver.log 2>&1; echo "npu $(date -u +%T)" >> logs/quiet_rerun.status
