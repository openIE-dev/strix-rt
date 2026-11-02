#!/bin/bash
# Follow-up: pplx-decider-flash v1.2 (re-converted with llama.cpp master convert_hf_to_gguf; the published GGUF lacks the qwen35.decision.type key -> 501).
cd ~/strix-rt; D=logs/cascade
until grep -q "RIVALS CHAIN DONE" $D/session.log; do sleep 60; done
echo "$(date +%T) rivals chain2 start" >> $D/session.log
bash tools/cascade/driver.sh acc pplxflash
for rep in 1 2 3; do bash tools/cascade/driver.sh e1 pplxflash $rep; done
for rep in 1 2; do bash tools/cascade/driver.sh b8 pplxflash $rep; done
echo "$(date +%T) RIVALS CHAIN2 DONE" >> $D/session.log
