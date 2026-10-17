#!/bin/bash
# Reference DiT per-step timing, perf level pinned high (3 steps, DiT only), restore auto.
P=/sys/class/drm/card1/device/power_dpm_force_performance_level; echo high | sudo tee $P >/dev/null; echo "perf level: $(cat $P)"
trap "echo auto | sudo tee $P >/dev/null; echo restored: \$(cat $P)" EXIT
mkdir -p ~/k6_ref/refbench
K6_DITONLY=1 K6_STEPS=3 K6_OUT=~/k6_ref/refbench ~/qi_ref/.venv_rocm/bin/python ~/k6_ref/ref.py > /dev/null 2>&1
cat ~/k6_ref/refbench/timing_dit.json
