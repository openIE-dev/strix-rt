#!/bin/bash
# tools/comp/run.sh TAG CMD..: wait_quiet -> gpu_lock -> perf high -> CMD, output appended to logs/comp/TAG.log
cd ~/strix-rt; T=$1; shift
echo "=== $(date -u +%FT%TZ) $*" >> logs/comp/$T.log
GPU_LOCK_TAG="comp $T" logs/wait_quiet.sh logs/gpu_lock.sh timeout -k 10 ${SESS_TIMEOUT:-290} tools/comp/perf_high.sh "$@" >> logs/comp/$T.log 2>&1
echo "=== rc $? $(date -u +%FT%TZ)" >> logs/comp/$T.log
