#!/bin/bash
# dq2_prompt.sh TAG "user text": llama.cpp vk reference + ours FORCED with DQ2 0/1
cd ~/strix-rt; B=~/llama-latest/build-vulkan/bin; M=$MODELS/new/gemma4/26b-a4b/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf; T=$1
PR=$(printf "<|turn>user\n%s<turn|>\n<|turn>model\n" "$2")
$B/llama-tokenize -m $M -p "$PR" --ids --log-disable 2>/dev/null | tail -1 > logs/g4/$T.p.txt
timeout 150 $B/llama-completion -m $M -ngl 999 -no-cnv -p "$PR" -n 64 --temp 0 --ignore-eos -s 1 --no-display-prompt > logs/g4/$T.ref.txt 2>/dev/null
$B/llama-tokenize -m $M -f logs/g4/$T.ref.txt --ids --log-disable --no-bos 2>/dev/null | tail -1 > logs/g4/$T.r.txt
for d in 0 1; do echo "$T DQ2=$d $(G4_DQ2=$d FORCED=1 timeout 120 ./target-g4p/release/g4_run $M @logs/g4/$T.p.txt 64 @logs/g4/$T.r.txt 2>&1 | grep FORCED | cut -c1-260)"; done
