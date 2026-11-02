#!/bin/bash
# tools/cascade/chain.sh : accuracy passes, then energy windows (3 repeats), each as its own gated, GPU-locked session. Run with nohup.
cd ~/strix-rt; B=tools/cascade/driver.sh
$B acc kev08 kev4b kev9q8 laya; $B acc2
for r in 1 2 3; do for t in kev08 kev4b kev9q8 laya; do $B e1 $t $r; done; $B e2 $r; done
echo "$(date +%T) chain done" >> logs/cascade/session.log
