#!/bin/bash
# one lock session: llama-server (Vulkan) load + greedy refs for 6 prompts, then exit. Run: logs/wait_quiet.sh logs/gpu_lock.sh logs/n35/sess_llama_ref.sh
cd ~/strix-rt/logs/n35; M=$MODELS/new/nemotron35vl/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-Q4_0.gguf
free -g | tee -a sess.log
~/llama-n35/build-vulkan/bin/llama-server -m $M -ngl 999 -c 4096 -fa on --port 18235 -np 1 > srv_ref.log 2>&1 & SP=$!
trap "kill $SP 2>/dev/null; wait $SP 2>/dev/null" EXIT
for i in $(seq 1 150); do kill -0 $SP 2>/dev/null || { echo server died; tail -5 srv_ref.log; exit 1; }; curl -s --max-time 2 http://127.0.0.1:18235/health | grep -q ok && break; sleep 1; done
echo "ready after $i s" | tee -a sess.log
timeout 150 python3 ref_llama.py 18235 48 ref_llama.json | tee -a sess.log
