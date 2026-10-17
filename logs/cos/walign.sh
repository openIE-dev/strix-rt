#!/bin/sh
# QI_WALIGN=1 (4 KB aligned weight copies) vs default on the gen step; velocity parity must stay identical
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
for v in "X=1" "QI_WALIGN=1" "QI_WALIGN=1 COS_LTIME=1"; do echo "== $v"; env $v timeout 50 $B 2>&1 | grep -E "loaded|prefix gpu|step gpu|layer ms|vel" | head -5 | cut -c1-200; done
