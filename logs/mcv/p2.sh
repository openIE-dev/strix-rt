#!/bin/bash
cd ~/strix-rt; M=$MODELS/minicpmv47/MiniCPM-V-4_7-35B-A3B-Q4_K_M.gguf; B=target-k6/release/mcv_ref
r() { PREFILL=1 REPS=3 timeout 120 $B $M ~/mcv_ref/$1 40 2>&1 | grep -E "submit|prefill \(b|MATCH" | tail -3; }
for d in ref_img ref_chess; do
  echo "== $d base"; PF_PROF=1 r $d
  for k in gdn_scan gdn_conv,gdn_convst attn gnorm,rmsnorm; do echo "== $d skip $k"; PF_PROF=1 PF_SKIPK=$k r $d; done
  for g in "16 16" "16 32" "64 64"; do set -- $g; echo "== $d GBM=$1 GBM2=$2"; PF_GBM=$1 PF_GBM2=$2 r $d; done
done
echo P2DONE
