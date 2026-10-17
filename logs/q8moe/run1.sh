#!/bin/bash
# q8moe (run under gpu_lock): parity (text prompt, greedy vs llama.cpp Vulkan dump) + prefill/decode speed, Qwen3.6-35B-A3B Q8_0 via mcv
cd ~/strix-rt; B=target-q8/release; M=$MODELS/bench/Qwen3.6-35B-A3B-Q8_0.gguf
echo "== parity"; timeout 280 $B/mcv_ref $M ~/mcv_ref/ref_q36txt 128 2>&1 | tail -12
echo "== prefill L512"; PREFILL=1 REPS=3 MAX_CTX=8400 timeout 280 $B/mcv_ref $M logs/comp/mcvref/L512 129 2>&1 | grep -E "prefill|decode|rror|panic|MATCH"
echo DONE
