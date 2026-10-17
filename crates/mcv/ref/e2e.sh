#!/bin/bash
cd ~/strix-rt
target-k6/release/mcv_ref $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/ref_img 126 > logs/mcv/img_refemb.log 2>&1
FORCED=1 target-k6/release/mcv_ref $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/ref_img > logs/mcv/img_refemb_forced.log 2>&1
OWN_VIS=~/mcv_ref/fox.ppm target-k6/release/mcv_ref $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/ref_img 126 > logs/mcv/img_own.log 2>&1
OWN_VIS=~/mcv_ref/fox.ppm FORCED=1 target-k6/release/mcv_ref $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/ref_img > logs/mcv/img_own_forced.log 2>&1
echo E2EDONE
