#!/bin/bash
# GO_PF acceptance: margin at token 54 (prompt600) + teacher-forced GO_PF=0 vs 1 on 5 prompts (762-1112 tokens)
cd ~/strix-rt; D=logs/gptoss; M=$MODELS/bench/gpt-oss-20b-MXFP4.gguf; G=target-gopf/go_run6
for i in 0 1 2 3 4; do
  GO_PF=0 timeout 120 $G $M @$D/pfp$i.txt 64 2>/dev/null | grep "^ids:" | sed "s/ids: //" > /tmp/pf6_ref$i.txt
  for pf in 0 1; do echo "== p$i GO_PF=$pf $(GO_PF=$pf PFORCED=1 MARGIN_AT=54 timeout 120 $G $M @$D/pfp$i.txt 64 @/tmp/pf6_ref$i.txt 2>&1 | grep -E "PFORCED|MARGIN|rror|panic" | tr "\n" " ")"; done
done; echo DONE
