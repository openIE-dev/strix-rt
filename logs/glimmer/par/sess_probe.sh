#!/bin/bash
# one lock session: teacher-forced top logprobs at mismatch positions on llama.cpp Vulkan(fa on/off), HIP, CPU
cd ~/strix-rt/logs/glimmer/par; M=$MODELS/new/glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
run() { tag=$1; bin=$2; shift 2
  $bin -m $M -c 4096 --port 18240 -np 1 "$@" > srv_$tag.log 2>&1 & SP=$!
  for i in $(seq 1 400); do kill -0 $SP 2>/dev/null || { echo "$tag server died"; tail -5 srv_$tag.log; return; }; curl -s --max-time 2 http://127.0.0.1:18240/health | grep -q ok && break; sleep 1; done
  python3 probe.py 18240 $tag; kill $SP; wait $SP 2>/dev/null; sleep 3; }
V=~/llama-latest/build-vulkan/bin/llama-server; H=~/llama.cpp/build-hip/bin/llama-server
run vk_fa $V -ngl 999 -fa on
run vk_nofa $V -ngl 999 -fa off
run hip $H -ngl 999 -fa on
run cpu $V -ngl 0 -fa off -t 32
