#!/bin/bash
# parity of prefill scan head-dim split: generated ids PS=1 vs PS=8/16, 5 prompts (PF=1 path), 48 new tokens
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
python3 - <<PY
import re
t=open("logs/nh/p512.ids").read(); import json
ids=[int(x) for x in re.findall(r"\d+",t)]
open("/tmp/nh_p200.ids","w").write(str(ids[:200])); open("/tmp/nh_p90.ids","w").write(str(ids[:90]))
PY
for f in logs/nh/p512.ids logs/nh/long_prompt.ids logs/nh/p30.ids logs/nh/div30.ids /tmp/nh_p200.ids /tmp/nh_p90.ids; do
 for ps in 1 8 16; do echo "$f PS=$ps $(NH_PSPLIT=$ps PF=1 timeout 60 target-nhp/release/nh $M "$(cat $f)" 48 2>&1 | grep ^ids | md5sum | cut -c1-8)"; done
done
echo "ref long:"; NH_PSPLIT=8 PF=1 timeout 60 target-nhp/release/nh $M "$(cat logs/nh/long_prompt.ids)" 48 2>&1 | grep ^ids | cut -c1-200; head -c 300 logs/nh/ref30b_long.ids
