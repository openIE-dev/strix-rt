#!/bin/bash
# nh_bench.sh OUTDIR : strix-rt vs llama.cpp Vulkan, pp512 + tg, with SoC power (hwmon power1_input, 50 ms) per phase.
# Run inside wait_quiet + gpu_lock (+ perf_high). Energy/token = mean steady power / throughput.
cd ~/strix-rt; O=$1; mkdir -p $O; M=${M:-$MODELS/new/nemotron3/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf}; TG=${TG:-256}
H=$(ls -d /sys/class/drm/card1/device/hwmon/hwmon* | head -1); LB=~/llama-latest/build-vulkan/bin/llama-bench
mon() { while :; do echo "$(date +%s%3N) $(cat $H/power1_input)"; sleep 0.05; done > $O/$1.pw & MP=$!; }
stop() { kill $MP; wait $MP 2>/dev/null; }
P="$(cat logs/nh/p512.ids)"; S="[1, 1784, 6330, 1307, 1278, 9718, 18344, 19570, 1454]"
sleep 5; mon idle; sleep 3; stop
mon s_pp; PF=1 RUNS=12 timeout 60 target-nh/release/nh $M "$P" 1 > $O/s_pp.txt 2>&1; stop
mon s_tg; PF=1 RUNS=2 timeout 60 target-nh/release/nh $M "$S" $TG > $O/s_tg.txt 2>&1; stop
mon l_pp; timeout 90 $LB -m $M -ngl 99 -fa 1 -p 512 -n 0 -r 8 > $O/l_pp.txt 2>&1; stop
mon l_tg; timeout 90 $LB -m $M -ngl 99 -fa 1 -p 0 -n $TG -r 2 > $O/l_tg.txt 2>&1; stop
echo bench done
