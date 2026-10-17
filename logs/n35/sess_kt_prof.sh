#!/bin/bash
# one lock session: kernel tests, then 1-step profile (perf level high)
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf
timeout 60 target/release/n35_kt $M 1 2>&1 | tail -5 || exit 1
tools/comp/perf_high.sh logs/n35/sess_prof.sh
