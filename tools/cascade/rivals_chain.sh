#!/bin/bash
# Nov 2026 open-weight rival decision models: accuracy (1000 items), energy e1 (batch 1, 3 reps), b8 floor (8 q/request vs 1, 2 reps).
# Every session goes through driver.sh (quiet gate + gpu_lock + perf high). Run detached: nohup bash tools/cascade/rivals_chain.sh &
cd ~/strix-rt; D=logs/cascade; R="d3edge d3lite d3flash"
until grep -q DLDONE $MODELS/rivals/dl.log; do sleep 30; done
echo "$(date +%T) rivals chain start" >> $D/session.log
bash tools/cascade/driver.sh acc $R
for rep in 1 2 3; do for t in $R; do bash tools/cascade/driver.sh e1 $t $rep; done; done
for rep in 1 2; do for t in kev08 $R kev9q8; do bash tools/cascade/driver.sh b8 $t $rep; done; done
echo "$(date +%T) RIVALS CHAIN DONE" >> $D/session.log
