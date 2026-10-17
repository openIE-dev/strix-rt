#!/bin/bash
cd ~/strix-rt
M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf
PREFILL=1 timeout 300 target-k6/release/mcv_ref $M ~/mcv_ref/ref_img 126 > logs/mcv/pf_img_refemb2.log 2>&1
PREFILL=1 timeout 300 target-k6/release/mcv_ref $M ~/mcv_ref/ref_txt 117 > logs/mcv/pf_txt2.log 2>&1
echo PFDONE
