#!/bin/sh
# One-session pp512: llama.cpp Vulkan vs strix-rt prefill at perf level auto and high, GPU telemetry per run
# (logs/pf_mon_<level>_<run>.csv); restores power_dpm_force_performance_level=auto at the end.
cd ~/strix-rt; D=/sys/class/drm/card1/device; B=~/llama.cpp/build-vulkan/bin/llama-bench
Q=$MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf; L=$MODELS/LFM2.5-24B-A2B-Q4_K_M.gguf; P2=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf; P1=$MODELS/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf
mon() { sh logs/pf_gpumon.sh logs/pf_mon_${LV}_$1.csv & MP=$!; }
sum() { kill $MP; awk -F, 'NR>1{n++; c+=$2; if(!mn||$2<mn)mn=$2; p+=$5; if($6>tm)tm=$6} END{printf "    [sclk avg %d min %d MHz, power avg %d W, temp max %d C]\n", c/n, mn, p/n, tm}' logs/pf_mon_${LV}_$1.csv; }
for LV in ${LEVELS:-auto high}; do
  echo $LV | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "== perf level $(cat $D/power_dpm_force_performance_level)"
  mon llq; echo "llama.cpp qwen38: $($B -m $Q -p 512 -n 0 -r 5 2>&1 | grep pp512 | awk -F'|' '{print $(NF-1)}')"; sum llq
  mon pfq; REPS=6 NOREF=1 ./target/release/pf_q38 $Q 512 1 2>&1 | grep "^prefill"; sum pfq
  mon lll; echo "llama.cpp lfm2: $($B -m $L -p 512 -n 0 -r 5 2>&1 | grep pp512 | awk -F'|' '{print $(NF-1)}')"; sum lll
  mon pfl; REPS=6 NOREF=1 ./target/release/pf_lfm $L 512 1 2>&1 | grep "^prefill"; sum pfl
  mon pfb2; REPS=4 NOREF=1 ./target/release/pf_q38 $P2 512 1 2>&1 | grep "^prefill"; sum pfb2
  mon pfb1; REPS=4 NOREF=1 ./target/release/pf_q38 $P1 512 1 2>&1 | grep "^prefill"; sum pfb1
done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null; echo "restored perf level: $(cat $D/power_dpm_force_performance_level)"
