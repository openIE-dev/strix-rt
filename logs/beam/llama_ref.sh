#!/bin/bash
cd ~/strix-rt
~/llama.cpp-beam/build-vk/bin/llama-completion -m $MODELS/new/beam/gguf/beam-nvfp4.gguf -ngl 0 --device none -c 256 -n 8 --temp 0 -p "The capital of France is" --no-display-prompt --verbose-prompt -t 16 > logs/beam/llama_ref.log 2>&1
echo "EXIT $?" >> logs/beam/llama_ref.log
