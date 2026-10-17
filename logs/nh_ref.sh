#!/bin/bash
# nh_ref.sh MODEL N "PROMPT" OUTPREFIX : llama.cpp Vulkan greedy reference (run under wait_quiet + gpu_lock)
B=~/llama-latest/build-vulkan/bin
timeout 240 $B/llama-simple -m "$1" -n "$2" -ngl 99 "$3" > "$4.txt" 2> "$4.err"
$B/llama-tokenize -m "$1" -f "$4.txt" --ids --log-disable 2>/dev/null | tail -1 > "$4.ids"
echo "ref done: $(wc -c < $4.txt) chars"
