#!/bin/bash
# reference: llama.cpp (Vulkan) greedy, image + text prompts
M=$MODELS/minicpmv47
cd ~/mcv_ref
mkdir -p ref_img ref_txt
./mcvdump $M/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf $M/mmproj-model-f16.gguf fox.png prompt_img.txt ref_img 256 > ref_img/log.txt 2>&1
./mcvdump $M/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf $M/mmproj-model-f16.gguf - prompt_txt.txt ref_txt 128 > ref_txt/log.txt 2>&1
echo REFDONE
