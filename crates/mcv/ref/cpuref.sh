#!/bin/bash
cd ~/mcv_ref; M=$MODELS/minicpmv47
MTMD_CPU=1 timeout 900 ./mcvdump $M/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf $M/mmproj-model-f16.gguf chess.png prompt_img.txt ref_chess_cpu 1 > ref_chess_cpu/log.txt 2>&1
python3 ~/strix-rt/crates/mcv/ref/embcmp.py ref_chess/img_embd_1.f32 ref_chess_cpu/img_embd_1.f32
