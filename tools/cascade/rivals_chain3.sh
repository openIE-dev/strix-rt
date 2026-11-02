#!/bin/bash
# Follow-up 2: Drex Max v1 (nace-ai/llama.cpp fork drex-max-v1, Vulkan) and pplx-decider-v1.2 27B (converted with llama.cpp master, Q8_0).
cd ~/strix-rt; D=logs/cascade; R=""
until grep -q DL2DONE $MODELS/rivals/dl2.log; do sleep 60; done
cd ~/llama-master && PYTHONPATH=~/llama-master/gguf-py nice -n 19 ~/venv-conv/bin/python convert_hf_to_gguf.py $MODELS/rivals/pplx-decider-v1.2 --outtype q8_0 --outfile $MODELS/rivals/pplx-decider-v1.2-conv-Q8_0.gguf > $MODELS/rivals/conv_27.log 2>&1 && R=pplx27
cd ~/strix-rt
[ -x ~/llama-drex/build-vulkan/bin/llama-server ] && R="drexmax $R"
until grep -q "RIVALS CHAIN2 DONE" $D/session.log; do sleep 60; done
echo "$(date +%T) rivals chain3 start [$R]" >> $D/session.log
bash tools/cascade/driver.sh acc $R
for rep in 1 2 3; do for t in $R; do bash tools/cascade/driver.sh e1 $t $rep; done; done
for rep in 1 2; do for t in $R; do bash tools/cascade/driver.sh b8 $t $rep; done; done
echo "$(date +%T) RIVALS CHAIN3 DONE" >> $D/session.log
