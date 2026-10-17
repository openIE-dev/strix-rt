#!/bin/sh
# gen step GPU time: default (2 layers/submit) vs one submit vs 1 layer/submit vs same-layer weights (locality test)
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
for v in "X=1" "COS_LPS=36" "COS_LPS=1" "COS_SAMEL=1" "QI_ZC=0"; do echo "== $v"; env $v timeout 100 $B 2>&1 | grep "step gpu"; done
