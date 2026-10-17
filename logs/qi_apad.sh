#!/bin/bash
cd ~/strix-rt
for e in "QI_APAD=0" "QI_APAD=64"; do echo "== $e"; logs/qi_prof.sh $e 2>&1 | grep -E "^(qkv|o |swiglu|down|flash)|full|rror"; done
echo "== qi 512 parity"; target/release/qi --golden ~/qi_ref/golden512 --size 512 --out /tmp/q5d.png 2>&1 | grep -E "step 39|PSNR|denoise"
