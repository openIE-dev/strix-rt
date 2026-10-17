#!/bin/bash
cd ~/strix-rt
PFCHK=1 timeout 300 target-k6/release/mcv_ref $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/ref_img > logs/mcv/pfchk.log 2>&1
echo CHKDONE
