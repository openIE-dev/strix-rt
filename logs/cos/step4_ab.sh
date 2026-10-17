#!/bin/sh
# Step 4 (denoise gap 262 vs 420 ms/eval): per-layer GPU time (COS_LTIME=1, 1 layer per submit) to see whether the gap is
# spread evenly (weight streaming / TLB on fresh layers, cf. COS_SAMEL=1 at ~300-390 ms) or concentrated in some layers / shards.
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
for v in "X=1" "COS_LTIME=1" "COS_SAMEL=1 COS_LTIME=1" "COS_LPS=36"; do echo "== $v"; env $v timeout 60 $B 2>&1 | grep -E "step gpu|layer ms" | head -4; done
COS_PROF=1 timeout 60 $B 2>&1 | grep -iE "prof|ms" | head -14
