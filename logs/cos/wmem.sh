#!/bin/sh
# Denoise gap: weight memory type A/B (QI_WMEM=<type>) on the gen tower step, perf level high
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
for v in "QI_ZC=0" "QI_WMEM=0" "QI_WMEM=2" "QI_WMEM=7" "QI_WMEM=0 COS_LTIME=1"; do echo "== $v"; env $v timeout 50 $B 2>&1 | grep -E "loaded|step gpu|layer ms|vel|rror" | head -5; done
