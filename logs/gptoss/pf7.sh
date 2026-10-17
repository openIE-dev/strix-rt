#!/bin/bash
# bisect: does decode itself depend on GO_NR (fused norm+router)? compare ids GO_PF=0 with GO_NR=0/1 and GO_PF=1 GO_NR=0
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf; G=target-gopf/go_run6
i=0; for c in "GO_PF=0 GO_NR=1" "GO_PF=0 GO_NR=0" "GO_PF=1 GO_NR=0" "GO_PF=1 GO_NR=1"; do i=$((i+1)); env $c timeout 120 $G $M @$D/pfp0.txt 64 2>/dev/null | grep "^ids:" > /tmp/pf7_$i.txt; done
python3 -c "
L=[eval(open(f\"/tmp/pf7_{i}.txt\").read()[5:]) for i in (1,2,3,4)]
d=lambda A,B: next((k for k,(x,y) in enumerate(zip(A,B)) if x!=y),\"none\")
print(\"pf0nr1 vs pf0nr0\", d(L[0],L[1]), \"| pf1nr0 vs pf0nr0\", d(L[2],L[1]), \"| pf1nr1 vs pf0nr1\", d(L[3],L[0]))"
echo DONE
