#!/bin/sh
# A/B: zero-copy page-cache weights vs device copies, gen-tower step time at 256x256x9 (192 tokens)
cd ~/strix-rt
echo "== zero-copy"; timeout 100 target-cos/release/cos --golden ~/cosmos_ref/g256 --stage tower 2>&1 | grep -v warm
echo "== QI_ZC=0"; QI_ZC=0 timeout 150 target-cos/release/cos --golden ~/cosmos_ref/g256 --stage tower 2>&1 | grep -v warm
