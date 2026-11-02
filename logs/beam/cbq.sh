#!/bin/bash
# Per-layer codebook quality check (Nov 2). Waits for the decision track's idle loop, then holds gpu_lock for build+requant+eval.
cd ~/strix-rt; O=logs/beam/cbq; mkdir -p $O; S=$O/status
echo "queued $(date -u +%T)" > $S
while pgrep -f "[t]ools/cascade/idle_loop.sh" >/dev/null || pgrep -f "[c]ascade/.*driver|[c]ascade/session" >/dev/null; do sleep 60; done
echo "start $(date -u +%T)" >> $S
GPU_LOCK_TAG="beam cbq (per-layer codebook eval)" logs/gpu_lock.sh bash logs/beam/cbq_body.sh
echo "end $(date -u +%T) rc=$?" >> $S
