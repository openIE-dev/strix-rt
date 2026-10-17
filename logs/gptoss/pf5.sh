#!/bin/bash
# determinism: GO_PF=0 x2, GO_PF=1 NB=4 x2 on the 762-token prompt
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
i=0; for pf in 0 0 1 1; do i=$((i+1)); GO_PF=$pf RUNS=1 timeout 120 target-gopf/go_run5 $M @$D/prompt600_ids.txt 64 2>&1 | grep "ids:" > /tmp/gopf5_$i.txt; done
python3 -c "
L=[eval(open(f\"/tmp/gopf5_{i}.txt\").read()[5:]) for i in (1,2,3,4)]
d=lambda A,B: next((k for k,(x,y) in enumerate(zip(A,B)) if x!=y),\"none\")
print(\"pf0 vs pf0\", d(L[0],L[1]), \"| pf1 vs pf1\", d(L[2],L[3]), \"| pf0 vs pf1\", d(L[0],L[2]))"
echo DONE
