#!/bin/bash
cd ~/strix-rt; M=$MODELS/new/nemotron3/nvidia_Nemotron-3-Nano-30B-A3B-Q4_K_M.gguf
for k in 0 256 0; do
 NH_SKIP=$k tools/energy/record.py record --hz 50 --json logs/nh/tel/skip$k.json -- bash -c "PF=1 RUNS=25 target-nhp/release/nh $M \"\$(cat logs/nh/p512.ids)\" 2 2>&1 | grep ^prefill | tail -3 | cut -c1-60" 
 python3 - logs/nh/tel/skip$k.json $k <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print("keys",list(d.keys())[:12]); 
ch=d["breakdown"]["channels"]; print([(c["fabric"],round(c["joules"],1), {k:v for k,v in c.items() if k not in("fabric","joules")}) for c in ch][:8])
PY
done
