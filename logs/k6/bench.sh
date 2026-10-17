#!/bin/bash
# k6 perf-pinned timings (run under logs/gpu_lock.sh): perf level high, medians, restore auto.
cd ~/strix-rt; B=target-k6/release/k6; P=/sys/class/drm/card1/device/power_dpm_force_performance_level; O=~/k6_ref/bench; mkdir -p $O/aud
echo high | sudo tee $P >/dev/null; echo "perf level: $(cat $P)  start $(date +%T)"
trap "echo auto | sudo tee $P >/dev/null; echo restored: \$(cat $P)" EXIT
echo "== te x3"; K6_REPS=3 $B te
echo "== gen 10 steps"; $B gen --ref ~/k6_ref/out2 --out $O/gen
for i in 1 2 3; do echo "== vae $i"; $B vae --lat $O/gen/latents.npy --out $O/vid.npy; done
echo "== aud x3"; K6_REPS=3 $B aud --alat $O/gen/audio_latents.npy --aref $O/aud
echo "end $(date +%T)"
