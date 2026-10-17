#!/bin/sh
# Is layer L intrinsically slow (placement) or only slow when streamed after others? all gen layers use layer L's weights.
cd ~/strix-rt; B="target-cos/release/cos --golden $HOME/cosmos_ref/g256 --stage tower"
for L in 0 5 10 20 35; do echo "== COS_SAMEL=$L"; COS_SAMEL=$L COS_LTIME=1 timeout 45 $B 2>&1 | grep -E "step gpu|layer ms" | head -2 | cut -c1-120; done
