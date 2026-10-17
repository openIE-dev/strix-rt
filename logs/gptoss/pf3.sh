#!/bin/bash
# GO_PF on a 762-token prompt: greedy ids must match GO_PF=0
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
for pf in 0 1; do GO_PF=$pf RUNS=1 timeout 120 target-gopf/go_run5 $M @$D/prompt600_ids.txt 64 2>&1 | grep -E "ids:|prompt|rror|panic" > /tmp/gopf$pf.txt; echo "== GO_PF=$pf"; cut -c1-200 /tmp/gopf$pf.txt; done
if [ "$(grep ids: /tmp/gopf0.txt)" == "$(grep ids: /tmp/gopf1.txt)" ]; then echo "IDS MATCH"; else echo "IDS DIFFER"; fi
echo DONE
