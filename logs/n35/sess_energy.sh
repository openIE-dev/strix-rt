#!/bin/bash
# one lock session (perf high): differential decode energy (tools/comp/energy.sh, N=320 minus N=64) for $1 = llama | ours
cd ~/strix-rt; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf; free -g | head -2
P=$(python3 -c 'import json;print(",".join(map(str,json.load(open("logs/n35/ref_llama.json"))[0]["ids"])))')
case $1 in
 llama) tools/comp/energy.sh n35_llama_vk "timeout 200 $HOME/llama-n35/build-vulkan/bin/llama-bench -m $M -ngl 999 -fa on -p 0 -n @N@ -r 1" ;;
 ours) tools/comp/energy.sh n35_ours "ZEROCOPY=${ZC:-0} STEP_CHUNK=32 MAX_CTX=512 timeout 200 target/release/n35 $M $P @N@" ;;
esac
