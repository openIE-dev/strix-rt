#!/bin/bash
# tools/comp/pf_load.sh: sustained 600-token prefills (serve_pf, 8 x 3 reps back to back) with GPU clock / power / temp
# sampled every 0.25 s (amdgpu hwmon) -> logs/serve/pf_load_q38.{log,csv}. Run under gpu_lock (perf_high).
cd ~/strix-rt; H=$(ls -d /sys/class/drm/card1/device/hwmon/hwmon* | head -1); C=logs/serve/pf_load_q38.csv
( echo "t,mhz,watts,temp_c"; while true; do echo "$(date +%s.%N | cut -c1-14),$(( $(cat $H/freq1_input) / 1000000 )),$(( $(cat $H/power1_input) / 1000000 )),$(( $(cat $H/temp1_input) / 1000 ))"; sleep 0.25; done ) > $C & SP=$!
trap "kill $SP" EXIT
sleep 3
ZEROCOPY=dmabuf timeout 300 target-serve/release/serve_pf $MODELS/Qwen3.8-27B-UD-Q4_K_M.gguf 600 600 600 600 600 600 600 600 2>&1 | while IFS= read -r l; do echo "$(date +%s.%N | cut -c1-14) $l"; done
sleep 5
