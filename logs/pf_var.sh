#!/bin/sh
# Variance study: for perf level auto / high, run llama-bench + strix-rt pp512 (qwen38) with GPU telemetry.
cd ~/strix-rt; D=/sys/class/drm/card1/device; B=~/llama.cpp/build-vulkan/bin/llama-bench; Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf
for LV in auto high; do
  echo $LV | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "== level $(cat $D/power_dpm_force_performance_level)"
  sh logs/pf_gpumon.sh logs/pf_mon_${LV}_llama.csv & P=$!
  $B -m $Q -p 512 -n 0 -r 5 2>&1 | grep pp512; kill $P
  sh logs/pf_gpumon.sh logs/pf_mon_${LV}_pf.csv & P=$!
  REPS=6 NOREF=1 ./target/release/pf_q38 $Q 512 1 2>&1 | grep "^prefill"; kill $P
done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored: $(cat $D/power_dpm_force_performance_level)"
