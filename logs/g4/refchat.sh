# Chat-templated llama.cpp reference: logs/g4/refchat.sh <tag> <model> [n]   (Vulkan + CPU, greedy)
cd ~/strix-rt; B=~/llama-latest/build-vulkan/bin; T=$1; M=$2; N=${3:-64}
PR=$(printf "<|turn>user\nThe history of the Roman Empire can be summarized as follows:<turn|>\n<|turn>model\n")
$B/llama-tokenize -m $M -p "$PR" --ids --log-disable 2>/dev/null | tail -1 > logs/g4/${T}c_prompt_ids.txt; cat logs/g4/${T}c_prompt_ids.txt
for be in vk cpu; do E=""; NG=999; [ $be = cpu ] && E="GGML_VK_VISIBLE_DEVICES=none" && NG=0
  env $E timeout 150 $B/llama-completion -m $M -ngl $NG -no-cnv -p "$PR" -n $N --temp 0 --ignore-eos -s 1 --no-display-prompt > logs/g4/${T}c_ref_$be.txt 2> logs/g4/${T}c_ref_$be.err
  echo "== $be"; cat logs/g4/${T}c_ref_$be.txt; echo
  $B/llama-tokenize -m $M -f logs/g4/${T}c_ref_$be.txt --ids --log-disable --no-bos 2>/dev/null | tail -1 > logs/g4/${T}c_${be}_ids.txt
done
