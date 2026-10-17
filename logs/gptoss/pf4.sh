#!/bin/bash
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf
GO_PF=0 RUNS=1 timeout 120 target-gopf/go_run5 $M @$D/prompt600_ids.txt 64 2>&1 | grep "ids:" > /tmp/gopf_ref.txt
for nb in 1 2 4; do GO_PF=1 GO_PF_NB=$nb RUNS=1 timeout 120 target-gopf/go_run5 $M @$D/prompt600_ids.txt 64 2>&1 | grep "ids:" > /tmp/gopf_nb.txt
python3 -c "
A=eval(open(\"/tmp/gopf_ref.txt\").read()[5:]);B=eval(open(\"/tmp/gopf_nb.txt\").read()[5:])
print(\"NB=$nb first diff\", next((k for k,(x,y) in enumerate(zip(A,B)) if x!=y),\"none\"), \"of\", len(A))"; done; echo DONE
