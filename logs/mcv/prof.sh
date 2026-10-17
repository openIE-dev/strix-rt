#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf; B=target-k6/release/mcv_ref
for d in ref_img ref_chess; do for p in 7 1 0; do
  echo "== $d PF_PART=$p"; PF_PROF=1 PF_PART=$p PREFILL=1 REPS=3 timeout 120 $B $M ~/mcv_ref/$d 2 2>&1 | grep -E "chunk|prefill \(b"
done; done
echo PROFDONE
for d in ref_img ref_gold ref_chess; do for g in "16 16" "16 32" "32 32" "64 64"; do set -- $g
  echo "== $d GBM=$1 GBM2=$2"; PF_GBM=$1 PF_GBM2=$2 PREFILL=1 REPS=3 timeout 120 target-k6/release/mcv_ref $MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf ~/mcv_ref/$d 40 2>&1 | grep -E "prefill \(b|MATCH"
done; done
echo SWEEPDONE
