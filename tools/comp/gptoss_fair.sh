#!/bin/bash
# gpt-oss-20b decode, same prompt/length/greedy for both: llama-server Vulkan no-draft (spec.sh) then strix-rt go_run, alternating, one lock session.
cd ~/strix-rt; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf; V=~/llama-latest/build-vulkan/bin
P="\"The history of the Roman Empire can be summarized as follows:\""
echo "strix-rt commit $(git rev-parse --short HEAD) (+go_run wall timer)"
for rep in 1 2; do
  tools/comp/spec.sh $V vk_gptoss_fair$rep $M "$P"
  RUNS=4 timeout 120 target-cmpgo/release/go_run $M @logs/comp/roman_ids.txt 128 2>&1 | grep -E "decode|rror|panic" | sed "s/^/STRIX rep$rep: /"
done
