#!/bin/sh
# pf_gpumon.sh OUT.csv : samples sclk/fclk/mclk level, power (W), temp (C) every 100 ms until killed
D=/sys/class/drm/card1/device; H=$(ls -d $D/hwmon/hwmon* | head -1)
echo "t_ms,sclk_mhz,fclk,mclk,power_w,temp_c" > "$1"
while :; do
  echo "$(date +%s%3N),$(($(cat $H/freq1_input)/1000000)),$(grep '\*' $D/pp_dpm_fclk | awk '{print $2}'),$(grep '\*' $D/pp_dpm_mclk | awk '{print $2}'),$(($(cat $H/power1_input)/1000000)),$(($(cat $H/temp1_input)/1000))" >> "$1"
  sleep 0.1
done
