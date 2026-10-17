#!/bin/bash
D=/sys/class/drm/card1/device; echo high | sudo tee $D/power_dpm_force_performance_level >/dev/null
cd ~/strix-rt
for r in 1 2; do QI_NO_TF=1 /usr/bin/time -f "qi wall %e s" target/release/qi --golden ~/qi_ref/golden1024 --size 1024 --out logs/qi1024.png 2>&1 | grep -v "warm\|step "; sudo dmesg -T | grep timeout | tail -1; done
echo auto | sudo tee $D/power_dpm_force_performance_level >/dev/null
~/qi_ref/.venv/bin/python ~/qi_ref/cmp.py logs/qi1024.png.img.f32 ~/qi_ref/golden1024/image.npy
echo ALLDONE
